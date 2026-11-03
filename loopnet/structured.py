"""Lower a restricted Python-like syntax to the float state-machine language.

This module parses Python ASTs but never evaluates or executes Python code.
Functions use runtime call frames; ``inline def`` opts into static expansion.
"""

import ast
from dataclasses import dataclass, field, replace
import math
import re

from .language import LanguageError, parse


class StructuredError(LanguageError):
    pass


@dataclass(frozen=True)
class _Context:
    names: dict[str, str]
    outputs: tuple[str, ...] = ()
    end: str | None = None
    loops: tuple[tuple[str, str], ...] = ()
    constants: dict[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class _Function:
    node: ast.FunctionDef
    parameters: tuple[str, ...]
    locals: tuple[str, ...]
    arity: int
    inline: bool = False


def _fails(node, message):
    raise StructuredError(f"Line {getattr(node, 'lineno', 1)}: {message}")


def _always_returns(body):
    for node in body:
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, ast.If) and _always_returns(node.body) and _always_returns(node.orelse):
            return True
    return False


class _Lowerer:
    def __init__(self, source, max_stack_depth, max_unrolled_iterations):
        if (not isinstance(max_stack_depth, int) or isinstance(max_stack_depth, bool)
                or not 1 <= max_stack_depth <= 2**53):
            raise ValueError("max_stack_depth must be a positive integer")
        self.max_stack_depth = max_stack_depth
        if (not isinstance(max_unrolled_iterations, int) or isinstance(max_unrolled_iterations, bool)
                or max_unrolled_iterations < 1):
            raise ValueError("max_unrolled_iterations must be a positive integer")
        self.max_unrolled_iterations = max_unrolled_iterations
        # Also accept the existing array declaration spelling, preserving lines.
        lines = []
        inline_lines = set()
        for line_number, line in enumerate(source.splitlines(), 1):
            inline = re.match(r"(\s*)inline\s+(def\s+.*)", line)
            if inline:
                inline_lines.add(line_number)
                line = inline[1] + inline[2]
            match = re.fullmatch(r"(\s*)array\s+([A-Za-z_][A-Za-z_0-9]*)\s*\[\s*(\d+)\s*\](\s*(?:#.*)?)", line)
            lines.append(f"{match[1]}{match[2]} = array({match[3]}){match[4]}" if match else line)
        try:
            self.module = ast.parse("\n".join(lines))
        except SyntaxError as error:
            raise StructuredError(f"Line {error.lineno}: {error.msg}") from error
        self.serial = 0
        self.lines = []
        self.arrays = {}
        self.declarations = set()
        self.functions = {}
        self.inline_stack = []
        self.internal_arrays = {}
        self.frame_function = None
        self.unrolled_iterations = 0
        # A called function may mutate a top-level induction variable. Such a
        # variable must retain ordinary runtime reads rather than substitution.
        self.written_globals = set()
        for node in ast.walk(self.module):
            names = ([node.id] if isinstance(node, ast.Name) else
                     [node.arg] if isinstance(node, ast.arg) else
                     [node.name] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else
                     node.names if isinstance(node, ast.Global) else [])
            for name in names:
                if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) or name.startswith("__ln_"):
                    _fails(node, "identifiers must be ASCII names; the __ln_ prefix is reserved")
        for node in self.module.body:
            if self.is_array(node):
                name = node.targets[0].id
                call = node.value
                if (len(call.args) != 1 or call.keywords or not isinstance(call.args[0], ast.Constant)
                        or type(call.args[0].value) not in (int, float)):
                    _fails(node, "array size must be a positive integer literal")
                size = call.args[0].value
                if not math.isfinite(size) or not float(size).is_integer() or not 1 <= size <= 2**53:
                    _fails(node, "array size must be a positive, exactly representable integer")
                if name in self.arrays:
                    _fails(node, f"duplicate array {name!r}")
                self.arrays[name] = int(size)
                self.declarations.add(id(node))
        for node in self.module.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name in self.functions or node.name in self.arrays or node.name in {"array", "relu"}:
                _fails(node, f"duplicate or reserved function name {node.name!r}")
            args = node.args
            parameters = tuple(arg.arg for arg in [*args.posonlyargs, *args.args])
            if (args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.kw_defaults
                    or node.decorator_list or node.returns
                    or any(arg.annotation for arg in [*args.posonlyargs, *args.args])):
                _fails(node, "functions support positional float parameters without defaults or annotations")
            if len(set(parameters)) != len(parameters):
                _fails(node, "duplicate function parameter")
            descendants = list(ast.walk(node))
            if any(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not node
                   for child in descendants):
                _fails(node, "nested function definitions are not supported")
            globals_ = {name for child in descendants if isinstance(child, ast.Global) for name in child.names}
            self.written_globals.update(globals_.intersection(
                child.id for child in descendants
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)))
            if globals_.intersection(parameters):
                _fails(node, "a parameter cannot also be declared global")
            locals_ = ({child.id for child in descendants if isinstance(child, ast.Name)
                        and isinstance(child.ctx, ast.Store)} | set(parameters)) - globals_
            if locals_.intersection(self.arrays):
                _fails(node, "a local or parameter cannot shadow a global array")
            self.functions[node.name] = _Function(node, parameters, tuple(sorted(locals_)), 0,
                                                  node.lineno in inline_lines)
        # A forwarding return equates two functions' output counts. Unifying
        # those constraints handles recursive and mutually recursive definitions.
        parents = {name: name for name in self.functions}

        def find(name):
            if parents[name] != name:
                parents[name] = find(parents[name])
            return parents[name]

        counts = {name: set() for name in self.functions}
        for name, function in self.functions.items():
            has_return = False
            for child in ast.walk(function.node):
                if not isinstance(child, ast.Return):
                    continue
                has_return = True
                value = child.value
                if value is None:
                    counts[name].add(0)
                elif isinstance(value, ast.Tuple):
                    counts[name].add(len(value.elts))
                elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in self.functions:
                    parents[find(name)] = find(value.func.id)
                else:
                    counts[name].add(1)
            if not has_return:
                counts[name].add(0)
        groups = {}
        for name in self.functions:
            groups.setdefault(find(name), set()).update(counts[name])
        for name, function in self.functions.items():
            choices = groups[find(name)]
            if len(choices) > 1:
                _fails(function.node, "all returns from a function must have the same number of outputs")
            count = next(iter(choices), 1)
            if count and not _always_returns(function.node.body):
                _fails(function.node, "a function with outputs must explicitly return them on every path")
            self.functions[name] = replace(function, arity=count)

    @staticmethod
    def is_array(node):
        return (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "array")

    def fresh(self, kind="tmp"):
        self.serial += 1
        name = f"__ln_{kind}_{self.serial}"
        return self.slot(name) if kind in {"tmp", "result"} else name

    def slot(self, name):
        if self.frame_function is not None:
            self.internal_arrays[name] = self.max_stack_depth + 1
            return f"{name}[__ln_sp]"
        return name

    def emit(self, text):
        # Frame-local indices live in arrays themselves. Load such an index into
        # a scalar immediately before each primitive use, keeping the primitive
        # grammar flat. Reloading after a call also preserves augmented-assignment
        # targets without leaving an unprotected scalar live across recursion.
        def index_register(match):
            register = self.fresh("index")
            self.lines.append(f"{register} = {match[2]}")
            return f"{match[1]}[{register}]"

        self.lines.append(re.sub(r"([A-Za-z_][A-Za-z_0-9]*)\[([A-Za-z_][A-Za-z_0-9]*\[__ln_sp\])\]",
                                 index_register, text))

    def number(self, value, node):
        try:
            result = float(value)
        except (OverflowError, ValueError):
            _fails(node, "constants must be finite floats")
        if not math.isfinite(result):
            _fails(node, "constants must be finite floats")
        return repr(result)

    def constant(self, node, context):
        """Fold pure arithmetic and unrolled induction variables; never eval."""
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            return context.constants.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.constant(node.operand, context)
            if value is not None:
                return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            left = self.constant(node.left, context)
            right = self.constant(node.right, context)
            if left is not None and right is not None:
                # Match the language's float arithmetic, including its rounding.
                left = float(self.number(left, node.left))
                right = float(self.number(right, node.right))
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                return left * right
        return None

    def constant_condition(self, node, context):
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self.constant_condition(node.operand, context)
            return None if value is None else not value
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for value in node.values:
                known = self.constant_condition(value, context)
                if known is None:
                    return None
                if known != isinstance(node.op, ast.And):
                    return known
            return isinstance(node.op, ast.And)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left = self.constant(node.left, context)
            right = self.constant(node.comparators[0], context)
            if left is not None and right is not None:
                left = float(self.number(left, node.left))
                right = float(self.number(right, node.comparators[0]))
                op = node.ops[0]
                if isinstance(op, ast.Gt):
                    return left > right
                if isinstance(op, ast.Lt):
                    return left < right
                if isinstance(op, ast.GtE):
                    return left >= right
                if isinstance(op, ast.LtE):
                    return left <= right
                if isinstance(op, ast.Eq):
                    return left == right
                if isinstance(op, ast.NotEq):
                    return left != right
            return None
        value = self.constant(node, context)
        return None if value is None else float(self.number(value, node)) > 0

    def for_loop(self, node, context):
        iterator = node.iter
        if (not isinstance(node.target, ast.Name) or not isinstance(iterator, ast.Call)
                or not isinstance(iterator.func, ast.Name) or iterator.func.id != "range"
                or iterator.keywords or not 1 <= len(iterator.args) <= 3):
            _fails(node, "for loops require a scalar target and range(stop), range(start, stop), or range(start, stop, step)")
        arguments = []
        for argument in iterator.args:
            value = self.constant(argument, context)
            if (value is None or abs(value) > 2**53 or not math.isfinite(value)
                    or not float(value).is_integer()):
                _fails(argument, "range bounds must be exactly representable compile-time integers")
            arguments.append(int(value))
        if len(arguments) == 3 and arguments[2] == 0:
            _fails(iterator, "range step cannot be zero")
        values = range(*arguments)
        self.unrolled_iterations += len(values)
        if self.unrolled_iterations > self.max_unrolled_iterations:
            _fails(node, f"program exceeds the {self.max_unrolled_iterations} unrolled-iteration limit")
        end = self.fresh("label")
        dest = self.target(node.target, context)
        name = node.target.id
        mutated = any(isinstance(child, ast.Name) and child.id == name
                      and isinstance(child.ctx, ast.Store)
                      for statement in node.body for child in ast.walk(statement))
        if name not in context.names and name in self.written_globals:
            mutated = True
        for value in values:
            onward = self.fresh("label")
            self.emit(f"{dest} = {self.number(value, node)}")
            constants = {**context.constants}
            constants.pop(name, None)
            if not mutated:
                constants[name] = value
            self.statements(node.body, replace(context, constants=constants,
                                               loops=(*context.loops, (onward, end))))
            self.emit(f"{onward}:")
        self.statements(node.orelse, context)
        self.emit(f"{end}:")

    def target(self, node, context):
        if isinstance(node, ast.Name):
            if node.id in self.arrays or node.id in self.functions or node.id == "array":
                _fails(node, "arrays and functions cannot be used as scalar addresses")
            return context.names.get(node.id, node.id)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            name = node.value.id
            if name not in self.arrays:
                _fails(node, f"undeclared array {name!r}")
            index = self.expression(node.slice, context)
            return f"{name}[{index}]"
        _fails(node, "expected a scalar address or an array cell")

    def subtract(self, left, right):
        try:
            constant = float(right)
        except ValueError:
            negative = self.fresh()
            self.emit(f"{negative} = {right} * -1")
            right = negative
        else:
            if constant == 0.0:
                return left
            right = repr(-constant)
        return self.binary(left, "+", right)

    def binary(self, left, operator, right):
        # Primitive binary instructions require an address on the left.
        try:
            float(left)
        except ValueError:
            pass
        else:
            saved = self.fresh()
            self.emit(f"{saved} = {left}")
            left = saved
        result = self.fresh()
        self.emit(f"{result} = {left} {operator} {right}")
        return result

    def expression(self, node, context):
        known = self.constant(node, context)
        if known is not None:
            return self.number(known, node)
        if isinstance(node, (ast.Name, ast.Subscript)):
            address = self.target(node, context)
            saved = self.fresh()
            self.emit(f"{saved} = {address}")
            return saved
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            left = self.expression(node.left, context)
            right = self.expression(node.right, context)
            if isinstance(node.op, ast.Sub):
                return self.subtract(left, right)
            return self.binary(left, "+" if isinstance(node.op, ast.Add) else "*", right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.expression(node.operand, context)
            if isinstance(node.op, ast.UAdd):
                return value
            try:
                return repr(-float(value))
            except ValueError:
                return self.binary(value, "*", "-1")
        if isinstance(node, ast.Call):
            values = self.call(node, context)
            if len(values) != 1:
                _fails(node, "an expression needs a function with exactly one output")
            return values[0]
        _fails(node, "float expressions support constants, addresses, +, -, *, and scalar function calls")

    def values(self, node, context):
        if node is None:
            return ()
        if isinstance(node, ast.Call):
            return self.call(node, context)
        if isinstance(node, ast.Tuple):
            return tuple(self.expression(item, context) for item in node.elts)
        return (self.expression(node, context),)

    def call(self, node, context):
        if isinstance(node.func, ast.Name) and node.func.id == "relu":
            if len(node.args) != 1 or node.keywords:
                _fails(node, "relu expects one float argument")
            value = self.expression(node.args[0], context)
            result = self.fresh()
            self.emit(f"{result} = relu({self.as_address(value)})")
            return (result,)
        if not isinstance(node.func, ast.Name) or node.func.id not in self.functions:
            _fails(node, "calls must name a defined function")
        name = node.func.id
        function = self.functions[name]
        if node.keywords or len(node.args) != len(function.parameters):
            _fails(node, f"{name} expects {len(function.parameters)} positional arguments")
        # Snapshot arguments before assigning parameter slots; every expression
        # read is materialized, preserving evaluation order and aliasing.
        arguments = tuple(self.expression(arg, context) for arg in node.args)
        if not function.inline:
            for index, value in enumerate(arguments):
                self.emit(f"__ln_arg_{index} = {value}")
            resume = self.fresh("label")
            self.emit("__ln_sp = __ln_sp + 1")
            self.emit(f"__ln_returns[__ln_sp] = &{resume}")
            self.emit(f"goto __ln_function_{name}")
            self.emit(f"{resume}:")
            outputs = tuple(self.fresh("result") for _ in range(function.arity))
            for index, output in enumerate(outputs):
                self.emit(f"{output} = __ln_return_{index}")
            return outputs
        if name in self.inline_stack:
            _fails(node, f"recursive inline expansion: {' -> '.join([*self.inline_stack, name])}")
        prefix = self.fresh("call")
        names = {local: self.slot(f"{prefix}_{local}") for local in function.locals}
        for local in function.locals:
            if local not in function.parameters:
                self.emit(f"{names[local]} = 0")
        for parameter, value in zip(function.parameters, arguments):
            self.emit(f"{names[parameter]} = {value}")
        outputs = tuple(self.fresh("result") for _ in range(function.arity))
        end = self.fresh("label")
        self.inline_stack.append(name)
        try:
            self.statements(function.node.body, _Context(names, outputs, end))
        finally:
            self.inline_stack.pop()
        self.emit(f"{end}:")
        return outputs

    def condition(self, node, yes, no, context):
        known = self.constant_condition(node, context)
        if known is not None:
            self.emit(f"goto {yes if known else no}")
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            self.condition(node.operand, no, yes, context)
            return
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for value in node.values[:-1]:
                onward = self.fresh("label")
                self.condition(value, onward if isinstance(node.op, ast.And) else yes,
                               no if isinstance(node.op, ast.And) else onward, context)
                self.emit(f"{onward}:")
            self.condition(node.values[-1], yes, no, context)
            return
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq)):
                _fails(node, "use one comparison at a time")
            left = self.expression(node.left, context)
            right = self.expression(node.comparators[0], context)
            op = node.ops[0]
            if isinstance(op, (ast.Lt, ast.GtE)):
                left, right = right, left
            if isinstance(op, (ast.GtE, ast.LtE)):
                yes, no = no, yes
            difference = self.subtract(left, right)
            if isinstance(op, (ast.Eq, ast.NotEq)):
                if isinstance(op, ast.NotEq):
                    yes, no = no, yes
                self.emit(f"if {self.as_address(difference)} goto {no}")
                negative = self.binary(difference, "*", "-1")
                self.emit(f"if {negative} goto {no}")
                self.emit(f"goto {yes}")
                return
            value = difference
        else:
            value = self.expression(node, context)
        self.emit(f"if {self.as_address(value)} goto {yes}")
        self.emit(f"goto {no}")

    def as_address(self, value):
        try:
            float(value)
        except ValueError:
            return value
        result = self.fresh()
        self.emit(f"{result} = {value}")
        return result

    def statements(self, body, context):
        for node in body:
            if self.is_array(node):
                if id(node) not in self.declarations:
                    _fails(node, "arrays must be declared at top level")
            elif isinstance(node, ast.FunctionDef):
                if context.end is not None or node not in self.module.body:
                    _fails(node, "functions must be defined at top level")
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    _fails(node, "chained assignment is unsupported")
                values = self.values(node.value, context)
                target = node.targets[0]
                targets = target.elts if isinstance(target, ast.Tuple) else [target]
                if len(targets) != len(values):
                    _fails(node, f"assignment has {len(targets)} targets and {len(values)} values")
                for dest, value in zip(targets, values):
                    self.emit(f"{self.target(dest, context)} = {value}")
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                dest = self.target(node.target, context)
                old = self.fresh()
                self.emit(f"{old} = {dest}")
                right = self.expression(node.value, context)
                result = (self.subtract(old, right) if isinstance(node.op, ast.Sub)
                          else self.binary(old, "+" if isinstance(node.op, ast.Add) else "*", right))
                self.emit(f"{dest} = {result}")
            elif isinstance(node, ast.While):
                head, body_label, otherwise, end = (self.fresh("label") for _ in range(4))
                self.emit(f"{head}:")
                self.condition(node.test, body_label, otherwise, context)
                self.emit(f"{body_label}:")
                self.statements(node.body, replace(context, loops=(*context.loops, (head, end))))
                self.emit(f"goto {head}")
                self.emit(f"{otherwise}:")
                self.statements(node.orelse, context)
                self.emit(f"{end}:")
            elif isinstance(node, ast.For):
                self.for_loop(node, context)
            elif isinstance(node, ast.If):
                known = self.constant_condition(node.test, context)
                if known is not None:
                    self.statements(node.body if known else node.orelse, context)
                    continue
                yes, no, end = (self.fresh("label") for _ in range(3))
                self.condition(node.test, yes, no, context)
                self.emit(f"{yes}:")
                self.statements(node.body, context)
                self.emit(f"goto {end}")
                self.emit(f"{no}:")
                self.statements(node.orelse, context)
                self.emit(f"{end}:")
            elif isinstance(node, (ast.Break, ast.Continue)):
                if not context.loops:
                    _fails(node, "break/continue requires a loop")
                self.emit(f"goto {context.loops[-1][1 if isinstance(node, ast.Break) else 0]}")
            elif isinstance(node, ast.Return):
                values = self.values(node.value, context)
                if len(values) != len(context.outputs):
                    _fails(node, "return output count does not match its function")
                if context.end is None:
                    self.emit("halt")
                else:
                    for dest, value in zip(context.outputs, values):
                        self.emit(f"{dest} = {value}")
                    self.emit(f"goto {context.end}")
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self.call(node.value, context)
            elif isinstance(node, (ast.Pass, ast.Global)):
                pass
            else:
                _fails(node, f"unsupported statement: {type(node).__name__}")

    def lower(self):
        self.statements(self.module.body, _Context({}))
        self.emit("halt")
        dynamic = [function for function in self.functions.values() if not function.inline]
        for function in dynamic:
            self.frame_function = function.node.name
            names = {name: self.slot(f"__ln_fn_{function.node.name}_{name}") for name in function.locals}
            self.emit(f"__ln_function_{function.node.name}:")
            for name in function.locals:
                if name not in function.parameters:
                    self.emit(f"{names[name]} = 0")
            for index, name in enumerate(function.parameters):
                self.emit(f"{names[name]} = __ln_arg_{index}")
            outputs = tuple(f"__ln_return_{index}" for index in range(function.arity))
            self.statements(function.node.body, _Context(names, outputs, "__ln_return"))
            self.emit("goto __ln_return")
        self.frame_function = None
        if dynamic:
            self.internal_arrays["__ln_returns"] = self.max_stack_depth + 1
            self.emit("__ln_return:")
            self.emit("__ln_ip = __ln_returns[__ln_sp]")
            self.emit("__ln_sp = __ln_sp + -1")
            self.emit("goto *__ln_ip")
        headers = ["continuation"]
        if dynamic:
            headers.append(f"stack __ln_sp {self.max_stack_depth}")
        result = "\n".join([*headers,
                            *(f"array {name}[{size}]" for name, size in {**self.arrays, **self.internal_arrays}.items()),
                            *self.lines, ""])
        parse(result)  # validate that the result belongs to the primitive language
        return result


def lower(source: str, *, max_stack_depth: int = 16, max_unrolled_iterations: int = 4096) -> str:
    """Return a reviewable primitive continuation program."""
    return _Lowerer(source, max_stack_depth, max_unrolled_iterations).lower()


def compile_structured(source: str, *, max_steps: int = 1, branch_margin: float = 1e-4,
                       max_stack_depth: int = 16, execution: str = "block",
                       max_unrolled_iterations: int = 4096):
    """Compile structured code; basic blocks run per call by default."""
    from .compiler import compile_program

    program = parse(lower(source, max_stack_depth=max_stack_depth,
                          max_unrolled_iterations=max_unrolled_iterations))
    return compile_program(program, max_steps=max_steps, branch_margin=branch_margin,
                           execution=execution, indirect_targets=program.address_targets,
                           optimize_temporaries=True)
