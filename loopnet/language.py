"""The source language and its exact, float64 reference interpreter."""

from dataclasses import dataclass
import math
import re
from typing import Mapping


class LanguageError(ValueError):
    pass


class ExecutionLimitError(RuntimeError):
    pass


class ArrayIndexError(RuntimeError):
    pass


class ControlFlowError(RuntimeError):
    pass


class StackError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArrayAccess:
    name: str
    index: str


Address = str | ArrayAccess


@dataclass(frozen=True)
class Instruction:
    op: str
    line: int
    dest: Address | None = None
    left: Address | None = None
    right: Address | float | None = None
    target: int | None = None


@dataclass(frozen=True)
class Program:
    instructions: tuple[Instruction, ...]
    variables: tuple[str, ...]
    arrays: tuple[tuple[str, int], ...] = ()
    continuation: bool = False
    stack_pointer: str | None = None
    max_stack_depth: int | None = None
    address_targets: tuple[int, ...] = ()

    def step(self, state: Mapping[str, float] | None = None,
             *, max_steps: int = 64) -> dict[str, float]:
        """Execute one complete invocation, preserving global memory."""
        if self.continuation:
            raise ValueError("Use Interpreter for continuation programs so control persists between calls")
        return Interpreter(self, state).step(max_steps=max_steps)


class Interpreter:
    """Reference execution with persistent control for continuation programs."""

    def __init__(self, program: Program, state: Mapping[str, float] | None = None):
        self.program = program
        supplied = dict(state or {})
        unknown = supplied.keys() - set(program.variables)
        if unknown:
            raise ValueError(f"Unknown globals: {', '.join(sorted(unknown))}")
        self.memory = {name: float(supplied.get(name, 0.0)) for name in program.variables}
        if not all(math.isfinite(value) for value in self.memory.values()):
            raise ValueError("Globals must be finite floats")
        self.pc = 0
        self.terminated = False

    def step(self, *, max_steps: int = 64) -> dict[str, float]:
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if self.terminated:
            return dict(self.memory)
        if not self.program.continuation:
            self.pc = 0
        memory = self.memory
        arrays = dict(self.program.arrays)

        def check_stack():
            if self.program.stack_pointer is not None:
                pointer = memory[self.program.stack_pointer]
                if not pointer.is_integer() or not 0 <= pointer <= self.program.max_stack_depth:
                    raise StackError(f"Call stack pointer must be an integer in [0, {self.program.max_stack_depth}]")

        check_stack()

        def resolve(address: Address, line: int) -> str:
            if isinstance(address, ArrayAccess):
                index = memory[address.index]
                if not index.is_integer() or not 0 <= index < arrays[address.name]:
                    raise ArrayIndexError(
                        f"Line {line}: {address.name}[{index:g}] needs an integer index "
                        f"in [0, {arrays[address.name]})")
                return f"{address.name}[{int(index)}]"
            return address

        for _ in range(max_steps):
            instruction = self.program.instructions[self.pc]
            if instruction.op in {"return", "halt"}:
                self.terminated = instruction.op == "halt" or self.program.continuation
                return dict(memory)
            if instruction.op == "goto":
                self.pc = instruction.target
                continue
            if instruction.op == "indirect":
                target = memory[resolve(instruction.left, instruction.line)]
                if not target.is_integer() or not 0 <= target < len(self.program.instructions):
                    raise ControlFlowError(f"Line {instruction.line}: invalid instruction address {target:g}")
                self.pc = int(target)
                continue
            if instruction.op == "if":
                condition = memory[resolve(instruction.left, instruction.line)]
                self.pc = instruction.target if condition > 0.0 else self.pc + 1
                continue
            dest = resolve(instruction.dest, instruction.line)
            right = (instruction.right if isinstance(instruction.right, float)
                     else memory[resolve(instruction.right, instruction.line)])
            if instruction.op == "set":
                value = right
            elif instruction.op == "add":
                value = memory[resolve(instruction.left, instruction.line)] + right
            elif instruction.op == "relu":
                value = max(0.0, memory[resolve(instruction.left, instruction.line)])
            else:
                value = memory[resolve(instruction.left, instruction.line)] * right
            if not math.isfinite(value):
                raise ArithmeticError(f"Line {instruction.line}: nonfinite arithmetic result")
            memory[dest] = value
            check_stack()
            self.pc += 1
        if self.program.continuation:
            return dict(memory)
        raise ExecutionLimitError(f"Function did not return within {max_steps} instructions")


_LABEL = r"[A-Za-z_][A-Za-z_0-9]*"
_SCALAR = rf"(?:{_LABEL}|\[\d+\])"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_INDEX = rf"(?:{_SCALAR}|{_NUMBER})"
_ARRAY_ACCESS = rf"({_LABEL})\s*\[\s*({_INDEX})\s*\]"
_ADDRESS = rf"(?:{_LABEL}\s*\[\s*{_INDEX}\s*\]|{_SCALAR})"
_DECLARATION = rf"array\s+({_LABEL})\s*\[\s*(\d+)\s*\]"
_STACK = rf"stack\s+({_SCALAR})\s+(\d+)"


def parse(source: str) -> Program:
    """Parse one function body. Missing globals start at zero; fallthrough returns."""
    instructions: list[Instruction] = []
    labels: dict[str, int] = {}
    unresolved: dict[int, str] = {}
    variables: dict[str, None] = {}
    arrays: dict[str, int] = {}
    continuation = False
    stack_pointer = None
    max_stack_depth = None
    label_values: dict[int, str] = {}
    lines = [(line_number, raw.split("#", 1)[0].strip())
             for line_number, raw in enumerate(source.splitlines(), 1)]

    # Declarations are static allocation, regardless of where execution jumps.
    # Collect them first so even an access before a declaration is unambiguous.
    for line_number, line in lines:
        if line == "continuation":
            if continuation:
                raise LanguageError(f"Line {line_number}: duplicate continuation directive")
            continuation = True
        stack = re.fullmatch(_STACK, line)
        if stack:
            if stack_pointer is not None or not 1 <= int(stack[2]) <= 2**53:
                raise LanguageError(f"Line {line_number}: stack needs one positive integer depth declaration")
            stack_pointer, max_stack_depth = stack[1], int(stack[2])
        declaration = re.fullmatch(_DECLARATION, line)
        if declaration:
            name, size = declaration[1], int(declaration[2])
            if name in arrays:
                raise LanguageError(f"Line {line_number}: duplicate array {name!r}")
            if not 1 <= size <= 2**53:
                raise LanguageError(f"Line {line_number}: array size must be a positive, exactly representable integer")
            arrays[name] = size
            for index in range(size):
                variables[f"{name}[{index}]"] = None
    if stack_pointer is not None:
        if stack_pointer in arrays:
            raise LanguageError("Stack pointer must be a scalar address")
        variables[stack_pointer] = None

    def address(token: str, line: int) -> Address:
        token = token.strip()
        access = re.fullmatch(_ARRAY_ACCESS, token)
        if access:
            name, index = access.groups()
            if name not in arrays:
                raise LanguageError(f"Line {line}: undeclared array {name!r}")
            if re.fullmatch(_SCALAR, index):
                if index in arrays:
                    raise LanguageError(f"Line {line}: an index must be a scalar address or a constant")
                variables[index] = None
                return ArrayAccess(name, index)
            value = number(index, line)
            if not value.is_integer() or not 0 <= value < arrays[name]:
                raise LanguageError(f"Line {line}: {name}[{index}] needs an integer index in [0, {arrays[name]})")
            return f"{name}[{int(value)}]"
        if token in arrays:
            raise LanguageError(f"Line {line}: array {token!r} needs an index")
        variables[token] = None
        return token

    def number(token: str, line: int) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise LanguageError(f"Line {line}: constants must be finite floats")
        return value

    for line_number, line in lines:
        if not line or line == "continuation" or re.fullmatch(_DECLARATION, line) or re.fullmatch(_STACK, line):
            continue
        label = re.fullmatch(rf"({_LABEL}):", line)
        if label:
            name = label[1]
            if name in labels:
                raise LanguageError(f"Line {line_number}: duplicate label {name!r}")
            labels[name] = len(instructions)
            continue
        if line in {"return", "halt"}:
            instructions.append(Instruction(line, line_number))
            continue
        indirect = re.fullmatch(rf"goto\s*\*\s*({_ADDRESS})", line)
        if indirect:
            instructions.append(Instruction("indirect", line_number, left=address(indirect[1], line_number)))
            continue
        branch = re.fullmatch(rf"if\s+({_ADDRESS})(?:\s*>\s*0)?\s+goto\s+({_LABEL})", line)
        jump = re.fullmatch(rf"goto\s+({_LABEL})", line)
        if branch or jump:
            target = branch[2] if branch else jump[1]
            unresolved[len(instructions)] = target
            left = address(branch[1], line_number) if branch else None
            instructions.append(Instruction("if" if branch else "goto", line_number, left=left))
            continue
        constant = re.fullmatch(rf"({_ADDRESS})\s*=\s*({_NUMBER})", line)
        label_value = re.fullmatch(rf"({_ADDRESS})\s*=\s*&({_LABEL})", line)
        binary = re.fullmatch(rf"({_ADDRESS})\s*=\s*({_ADDRESS})\s*([+*])\s*({_ADDRESS}|{_NUMBER})", line)
        copy = re.fullmatch(rf"({_ADDRESS})\s*=\s*({_ADDRESS})", line)
        relu = re.fullmatch(rf"({_ADDRESS})\s*=\s*relu\(\s*({_ADDRESS})\s*\)", line)
        if label_value:
            label_values[len(instructions)] = label_value[2]
            instructions.append(Instruction("set", line_number, address(label_value[1], line_number)))
        elif constant:
            instructions.append(Instruction("set", line_number, address(constant[1], line_number),
                                            right=number(constant[2], line_number)))
        elif relu:
            instructions.append(Instruction("relu", line_number, address(relu[1], line_number),
                                            address(relu[2], line_number), 0.0))
        elif binary:
            dest, left, operator, token = binary.groups()
            dest = address(dest, line_number)
            left = address(left, line_number)
            right = (address(token, line_number) if re.fullmatch(_ADDRESS, token)
                     else number(token, line_number))
            instructions.append(Instruction("add" if operator == "+" else "mul",
                                            line_number, dest, left, right))
        elif copy:
            instructions.append(Instruction("add", line_number, address(copy[1], line_number),
                                            address(copy[2], line_number), 0.0))
        else:
            raise LanguageError(f"Line {line_number}: invalid instruction: {line!r}")

    # Also gives labels at the end of a file a valid return destination.
    instructions.append(Instruction("return", len(source.splitlines()) + 1))
    for index, label in unresolved.items():
        old = instructions[index]
        if label not in labels:
            raise LanguageError(f"Line {old.line}: unknown label {label!r}")
        instructions[index] = Instruction(old.op, old.line, left=old.left, target=labels[label])
    for index, label in label_values.items():
        old = instructions[index]
        if label not in labels:
            raise LanguageError(f"Line {old.line}: unknown label {label!r}")
        instructions[index] = Instruction("set", old.line, old.dest, right=float(labels[label]))
    return Program(tuple(instructions), tuple(variables), tuple(arrays.items()), continuation,
                   stack_pointer, max_stack_depth,
                   tuple(sorted({labels[label] for label in label_values.values()})))
