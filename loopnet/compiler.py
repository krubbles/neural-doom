"""Synthesize compact recurrent state and a weight-only transient ReGLU DAG."""

import math

from .language import ArrayAccess, Program, parse
from .network import Affine, Network, ResidualBlock, TransientBlock, Unit, channel
from .regions import execution_regions, recurrent_variables


_ONE = Affine(bias=1.0)


def _sum(*expressions):
    terms = {}
    for expression in expressions:
        for index, weight in expression.terms:
            terms[index] = terms.get(index, 0.0) + weight
    return Affine(tuple(sorted((i, w) for i, w in terms.items() if w)),
                  math.fsum(e.bias for e in expressions))


class _Branch:
    """Compile-time SSA activations, common subexpressions, and DAG scheduling."""
    def __init__(self, width):
        self.width = width
        self.nodes = []
        self.cache = {}

    def temporary(self, rows):
        rows = tuple(rows)
        if rows not in self.cache:
            index = self.width + len(self.nodes)
            self.cache[rows] = index
            self.nodes.append(rows)
        return channel(self.cache[rows])

    def finish(self, clear, write):
        needed = set()
        pending = [i for block in (clear, write) for unit in block.units
                   for expr in (unit.gate, unit.value) for i, _ in expr.terms if i >= self.width]
        while pending:
            index = pending.pop()
            if index in needed:
                continue
            needed.add(index)
            pending.extend(i for gate, value, _ in self.nodes[index - self.width]
                           for expr in (gate, value) for i, _ in expr.terms if i >= self.width)
        alive = sorted(needed)
        remap = {old: self.width + i for i, old in enumerate(alive)}

        def affine(expr):
            return Affine(tuple((remap.get(i, i), w) for i, w in expr.terms), expr.bias)

        def projection(block):
            return ResidualBlock(tuple(Unit(affine(u.gate), affine(u.value), u.outputs)
                                       for u in block.units), block.bias)

        depths, layers = {}, []
        for old in alive:
            rows = self.nodes[old - self.width]
            depth = max((depths.get(i, 0) for gate, value, _ in rows
                         for expr in (gate, value) for i, _ in expr.terms), default=0)
            while len(layers) <= depth:
                layers.append([])
            layers[depth].extend(Unit(affine(g), affine(v), ((remap[old], w),)) for g, v, w in rows)
            depths[old] = depth + 1
        return TransientBlock(len(alive), tuple(ResidualBlock(tuple(layer)) for layer in layers),
                              projection(clear), projection(write))


def compile_program(source: str | Program, *, max_steps: int = 64,
                    branch_margin: float = 1e-4, execution: str = "instruction",
                    indirect_targets: tuple[int, ...] | None = None,
                    optimize_temporaries: bool = False) -> Network:
    """Compile scalar control plus transient feedforward activations.

    max_steps counts primitive instructions in instruction mode, and basic blocks
    in block mode. Generic computed gotos conservatively use instruction regions;
    the structured frontend supplies its known dynamic return destinations and
    enables elimination of its reserved compiler-temporary names. Ordinary
    primitive globals are always retained, regardless of their spelling.
    """
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    branch_margin = float(branch_margin)
    if not math.isfinite(branch_margin) or branch_margin <= 0:
        raise ValueError("branch_margin must be positive and finite")
    if not math.isfinite(1.0 / branch_margin):
        raise ValueError("branch_margin is too small for float64")
    ramp = math.ldexp(1.0, math.frexp(branch_margin)[1] - 1)
    inverse_ramp = 1.0 / ramp
    if not math.isfinite(inverse_ramp):
        raise ValueError("branch_margin is too small for float64")
    program = parse(source) if isinstance(source, str) else source
    regions = execution_regions(program, execution, indirect_targets)
    variables = recurrent_variables(program, regions,
                                    eliminate=execution == "block" and optimize_temporaries,
                                    indirect_targets=indirect_targets)
    addresses = {name: i for i, name in enumerate(variables)}
    arrays = dict(program.arrays)
    transient_arrays = {name for name in arrays if f"{name}[0]" not in addresses}
    pc, done, margin_fault, index_fault, terminate, jump_fault, stack_fault = range(len(variables), len(variables) + 7)
    width = len(variables) + 7
    branch = _Branch(width)
    temp = branch.temporary
    write = []
    clear_masks = {}
    selectors_cache = {}
    entries = {start for start, _ in regions}

    def fault(target, value, gate=_ONE):
        write.append(Unit(gate, value, ((target, 1.0),)))

    def one_hot(size, value, enabled, diagnostic):
        key = (size, value, enabled, diagnostic)
        if key in selectors_cache:
            return selectors_cache[key]
        active = temp([(enabled, value, 1.0)])
        above = [temp([(_sum(active, enabled.scaled(-cell)), _ONE, 1.0)]) for cell in range(size)]
        below = [temp([(_sum(enabled.scaled(cell), active.scaled(-1)), _ONE, 1.0)]) for cell in range(size)]
        masks = [temp([(_sum(enabled, a.scaled(-1), b.scaled(-1)), _ONE, 1.0)])
                 for a, b in zip(above, below)]
        fault(diagnostic, below[0])
        fault(diagnostic, above[-1])
        for a, b in zip(above, below[1:]):
            excess = temp([(_sum(a, b.scaled(-1)), _ONE, 1.0)])
            fraction = temp([(enabled, _sum(a, excess.scaled(-1)), 1.0)])
            fault(diagnostic, fraction)
        selectors_cache[key] = masks
        return masks

    running = Affine(((done, -1.0),), 1.0)
    enables = []
    for start, end in regions:
        above = temp([(Affine(((pc, 1.0),), -float(start)), _ONE, 1.0)])
        below = temp([(Affine(((pc, -1.0),), float(start)), _ONE, 1.0)])
        enabled = temp([(_sum(_ONE, above.scaled(-1), below.scaled(-1)), running, 1.0)])
        enables.append(enabled)
        versions = {}
        dirty = set()

        def scalar(name):
            if name not in versions:
                if name not in addresses:
                    raise ValueError(f"Compiler value {name!r} was not defined on region entry")
                versions[name] = temp([(enabled, channel(addresses[name]), 1.0)])
            return versions[name]

        def read(address):
            if isinstance(address, float):
                return Affine(bias=address)
            if not isinstance(address, ArrayAccess):
                return scalar(address)
            if address.name in transient_arrays:
                return versions[address.name]
            masks = one_hot(arrays[address.name], scalar(address.index), enabled, index_fault)
            return temp([(mask, scalar(f"{address.name}[{cell}]"), 1.0)
                         for cell, mask in enumerate(masks)])

        def check_stack():
            if program.stack_pointer is not None:
                one_hot(program.max_stack_depth + 1, scalar(program.stack_pointer), enabled, stack_fault)

        check_stack()
        for instruction in program.instructions[start:end]:
            op = instruction.op
            if op in {"return", "halt"}:
                fault(done, _ONE, enabled)
                if op == "halt" or program.continuation:
                    fault(terminate, _ONE, enabled)
                next_pc = enabled.scaled(end - 1)
            elif op == "goto":
                next_pc = enabled.scaled(instruction.target)
            elif op == "indirect":
                target = read(instruction.left)
                masks = one_hot(len(program.instructions), target, enabled, jump_fault)
                for index, mask in enumerate(masks):
                    if index not in entries:
                        fault(jump_fault, mask)
                next_pc = target
            elif op == "if":
                condition = read(instruction.left)
                positive = temp([(condition, enabled, 1.0)])
                below = temp([(_sum(enabled.scaled(ramp), positive.scaled(-1)), _ONE, 1.0)])
                taken = temp([(enabled, _sum(enabled, below.scaled(-inverse_ramp)), 1.0)])
                next_pc = _sum(enabled.scaled(end), taken.scaled(instruction.target - end))
                gap = temp([(_sum(enabled.scaled(branch_margin), positive.scaled(-1)), _ONE, 1.0)])
                fault(margin_fault, positive, gap.scaled(1.0 / branch_margin))
            else:
                # Snapshot the destination index before any assignment effects.
                dest = instruction.dest
                transient = isinstance(dest, ArrayAccess) and dest.name in transient_arrays
                masks = (one_hot(arrays[dest.name], scalar(dest.index), enabled, index_fault)
                         if isinstance(dest, ArrayAccess) and not transient else None)
                if op == "set":
                    value = temp([(enabled, Affine(bias=instruction.right), 1.0)])
                else:
                    left, right = read(instruction.left), read(instruction.right)
                    if op == "add":
                        value = (left if instruction.right == 0.0 else
                                 temp([(enabled, left, 1.0), (enabled, right, 1.0)]))
                    elif op == "relu":
                        value = temp([(left, enabled, 1.0)])
                    elif isinstance(instruction.right, float):
                        value = temp([(enabled, left, instruction.right)])
                    else:
                        value = temp([(left, right, 1.0), (left.scaled(-1), right, -1.0)])
                if transient:
                    versions[dest.name] = value
                elif masks is None:
                    versions[dest] = value
                    dirty.add(dest)
                else:
                    for cell, mask in enumerate(masks):
                        name = f"{dest.name}[{cell}]"
                        old = scalar(name)
                        kept = temp([(_ONE, old, 1.0), (mask, old, -1.0)])
                        versions[name] = temp([(_ONE, kept, 1.0), (mask, value, 1.0)])
                        dirty.add(name)
                if dest == program.stack_pointer:
                    check_stack()
                next_pc = enabled.scaled(end)
        for name in sorted(dirty):
            if name in addresses:
                target = addresses[name]
                clear_masks.setdefault(target, []).append(enabled)
                saved = temp([(enabled, versions[name], 1.0)])
                write.append(Unit(enabled, saved, ((target, 1.0),)))
        # Snapshot control outputs just like data outputs: writeback may clear pc.
        saved_pc = temp([(enabled, next_pc, 1.0)])
        write.append(Unit(enabled, saved_pc, ((pc, 1.0),)))
    clear_masks[pc] = enables
    clear = ResidualBlock(tuple(Unit(_sum(*masks), channel(target), ((target, -1.0),))
                                for target, masks in sorted(clear_masks.items())))
    deep = branch.finish(clear, ResidualBlock(tuple(write)))
    reset_channels = [margin_fault, index_fault, jump_fault, stack_fault]
    reset_units = []
    if not program.continuation:
        reset_channels.extend((pc, done))
        reset_units.append(Unit(channel(terminate), channel(pc), ((pc, 1.0),)))
        reset_units.append(Unit(_ONE, channel(terminate), ((done, 1.0),)))
    reset_units.extend(Unit(_ONE, channel(index), ((index, -1.0),)) for index in reset_channels)
    reset = ResidualBlock(tuple(reset_units))
    return Network(width, variables, done, margin_fault, branch_margin, max_steps,
                   (reset, deep), (0,) + (1,) * max_steps,
                   arrays=tuple((name, size) for name, size in program.arrays if name not in transient_arrays),
                   index_channel=index_fault, terminate_channel=terminate,
                   continuation=program.continuation, jump_channel=jump_fault,
                   stack_channel=stack_fault, max_stack_depth=program.max_stack_depth,
                   pc_channel=pc, execution=execution, regions=regions)
