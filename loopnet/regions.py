"""Compile-time control-flow regions and liveness; never used to execute weights."""

from .language import ArrayAccess


CONTROL = {"if", "goto", "indirect", "return", "halt"}


def execution_regions(program, execution="instruction", indirect_targets=None):
    if execution not in {"instruction", "block"}:
        raise ValueError("execution must be 'instruction' or 'block'")
    instructions = program.instructions
    # An arbitrary computed goto can enter any instruction. Structured returns
    # supply their statically known return addresses; generic jumps stay safe.
    if execution == "instruction" or (indirect_targets is None
            and any(i.op == "indirect" for i in instructions)):
        return tuple((i, i + 1) for i in range(len(instructions)))
    leaders = {0}
    for index, instruction in enumerate(instructions):
        if instruction.op == "goto" and instruction.target == index + 1:
            continue  # inline leaf returns often jump directly to the next instruction
        if instruction.target is not None:
            leaders.add(instruction.target)
        if instruction.op in CONTROL and index + 1 < len(instructions):
            leaders.add(index + 1)
    leaders.update(indirect_targets or ())
    starts = sorted(leaders)
    return tuple(zip(starts, [*starts[1:], len(instructions)]))


def recurrent_variables(program, regions, *, eliminate=False, indirect_targets=None):
    """Keep user cells and compiler values live on entry to any region."""
    if not eliminate:
        return program.variables
    arrays = dict(program.arrays)

    def reads(address):
        if isinstance(address, ArrayAccess):
            return {address.index, *(f"{address.name}[{i}]" for i in range(arrays[address.name]))}
        return {address} if isinstance(address, str) else set()

    uses, definitions, successors = [], [], []
    entries = {start: index for index, (start, _) in enumerate(regions)}
    for start, end in regions:
        used, defined = set(), set()
        for instruction in program.instructions[start:end]:
            operands = reads(instruction.left) | reads(instruction.right)
            if isinstance(instruction.dest, ArrayAccess):
                operands.add(instruction.dest.index)
            used.update(operands - defined)
            if isinstance(instruction.dest, str):
                defined.add(instruction.dest)
        last = program.instructions[end - 1]
        if last.op in {"return", "halt"}:
            targets = ()
        elif last.op == "goto":
            targets = (last.target,)
        elif last.op == "if":
            targets = (last.target, end)
        elif last.op == "indirect":
            targets = indirect_targets if indirect_targets is not None else entries
        else:
            targets = (end,)
        uses.append(used)
        definitions.append(defined)
        successors.append({entries[target] for target in targets if target in entries})
    live = [set() for _ in regions]
    changed = True
    while changed:
        changed = False
        for i in reversed(range(len(regions))):
            outgoing = set().union(*(live[s] for s in successors[i]))
            incoming = uses[i] | (outgoing - definitions[i])
            if incoming != live[i]:
                live[i] = incoming
                changed = True
    crossing = set().union(*live)
    # A frame-local temporary bank can also be transient if every read in every
    # region follows a write to that same bank at the same, unchanged frame
    # pointer. Otherwise keep the entire bank, preserving recursive live values.
    banks = {name for name, size in arrays.items()
             if name.startswith(("__ln_tmp_", "__ln_result_", "__ln_call_"))
             and program.stack_pointer is not None and size == program.max_stack_depth + 1}
    retained_banks = set()
    for start, end in regions:
        locally_written = set()
        for instruction in program.instructions[start:end]:
            for operand in (instruction.left, instruction.right):
                if isinstance(operand, ArrayAccess) and operand.name in banks:
                    if operand.index != program.stack_pointer or operand.name not in locally_written:
                        retained_banks.add(operand.name)
                elif isinstance(operand, str) and "[" in operand and operand.split("[", 1)[0] in banks:
                    retained_banks.add(operand.split("[", 1)[0])
            dest = instruction.dest
            if isinstance(dest, ArrayAccess) and dest.name in banks:
                if dest.index == program.stack_pointer:
                    locally_written.add(dest.name)
                else:
                    retained_banks.add(dest.name)
            elif isinstance(dest, str) and "[" in dest and dest.split("[", 1)[0] in banks:
                retained_banks.add(dest.split("[", 1)[0])
            if dest == program.stack_pointer:
                locally_written.clear()
    transient_banks = banks - retained_banks
    return tuple(name for name in program.variables
                 if name.split("[", 1)[0] not in transient_banks
                 and (name in crossing or "[" in name
                      or not name.startswith(("__ln_tmp_", "__ln_result_", "__ln_index_", "__ln_call_"))))


def reference_step(reference, regions, max_steps):
    """Test/CLI oracle: step source instructions to each compiled boundary."""
    ends = dict(regions)
    if reference.terminated:
        return dict(reference.memory)
    if not reference.program.continuation:
        reference.pc = 0
        reference.terminated = False
    result = dict(reference.memory)
    # The interpreter's per-instruction entry point is continuation mode. Keep
    # its independent control/array/stack semantics while respecting function
    # mode's return boundary and overall execution-region budget.
    from dataclasses import replace
    from .language import ExecutionLimitError
    original = reference.program
    reference.program = replace(original, continuation=True)
    try:
        for _ in range(max_steps):
            if reference.terminated:
                break
            length = ends[reference.pc] - reference.pc
            result = reference.step(max_steps=length)
        if not original.continuation:
            ended = reference.terminated
            terminating = original.instructions[reference.pc].op == "halt"
            reference.terminated = ended and terminating
            if not ended:
                raise ExecutionLimitError(f"Function did not return within {max_steps} execution regions")
        return result
    finally:
        reference.program = original
