import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unittest

from loopnet import (ExecutionLimitError, Interpreter, Network, NetworkContractError,
                     StructuredError, compile_program, compile_structured, lower, parse)


ROOT = Path(__file__).resolve().parents[1]


class _FloatArray(list):
    """Python oracle adapter for the language's integer-valued float indices."""

    @staticmethod
    def index_value(index):
        if not float(index).is_integer():
            raise IndexError("fractional index")
        return int(index)

    def __getitem__(self, index):
        return super().__getitem__(self.index_value(index))

    def __setitem__(self, index, value):
        return super().__setitem__(self.index_value(index), value)


class StructuredTests(unittest.TestCase):
    def assertPythonResult(self, source, *, initial=None, max_steps=3, max_ticks=2000, max_stack_depth=16):
        # Only trusted test literals are executed here. The compiler itself never
        # executes source or Python ASTs. This provides an independent oracle.
        expected = {"__builtins__": {}, "range": range,
                    "array": lambda count: _FloatArray([0.0] * count), **(initial or {})}
        python_source = re.sub(r"^(\s*)inline\s+def\s", r"\1def ", source, flags=re.MULTILINE)
        exec(compile(python_source, "<trusted structured test>", "exec"), expected)
        flattened = {}
        for name, value in expected.items():
            if type(value) in (int, float):
                flattened[name] = float(value)
            elif isinstance(value, list):
                flattened.update({f"{name}[{i}]": float(item) for i, item in enumerate(value)})
        net = compile_structured(source, max_steps=max_steps, max_stack_depth=max_stack_depth)
        actual = net.run(initial, max_ticks=max_ticks)
        self.assertEqual(actual.keys(), flattened.keys())
        for name, value in actual.items():
            self.assertTrue(math.isclose(value, flattened[name], rel_tol=1e-12, abs_tol=1e-12),
                            (name, value, flattened[name]))
        return net, actual

    def test_tuple_function_and_while(self):
        source = (ROOT / "examples/functions.flow").read_text()
        net, values = self.assertPythonResult(source)
        self.assertTrue(net.continuation)
        self.assertEqual(values["total"], 38)
        self.assertEqual(values["remaining"], 0)

    def test_nested_calls_multiple_call_sites_and_shadowed_names(self):
        self.assertPythonResult("""
def pair(x, y):
    work = x + y
    return work, x * y

def outer(x, y):
    a, b = pair(x, y)
    return a + b, a * b

x = 2
y = 3
a, b = outer(x, y)
c, d = outer(y, x)
x, y = pair(y, x)
""")

    def test_return_forwarding_and_scalar_call_expressions(self):
        self.assertPythonResult("""
def pair(x):
    return x, x * x

def forward(x):
    return pair(x)

def twice(x):
    return x * 2

a, b = forward(3)
c = twice(twice(a)) + twice(b)
""")

    def test_return_in_nested_while_and_if(self):
        self.assertPythonResult("""
def first(n):
    while n > 0:
        if n == 2:
            return n, n * n
        n -= 1
    return -1, -2

a, b = first(4)
c, d = first(0)
""")

    def test_function_locals_reset_each_call(self):
        source = """
def count():
    local = local + 1
    return local

n = 3
total = 0
while n > 0:
    total += count()
    n -= 1
"""
        self.assertEqual(compile_structured(source).run()["total"], 3)

    def test_argument_snapshots_and_global_side_effects(self):
        self.assertPythonResult("""
def bump():
    global x
    x += 1
    return x

def pair(a, b):
    return a, b

x = 2
a, b = pair(x, bump())
y = x + bump()
""")

    def test_void_calls_and_global_array_access(self):
        self.assertPythonResult("""
bins = array(3)
def put(i, value):
    bins[i] = value
    return

def work(i):
    put(i, i * i)

i = 0
while i < 3:
    work(i)
    i += 1
""")

    def test_parallel_outputs_and_index_assignment_order(self):
        self.assertPythonResult("""
a = array(3)
def pair(x, y):
    return y, x

i = 0
x = 2
i, a[i] = pair(x, i)
a[0], a[2] = a[2], 7
""")

    def test_nested_loops_break_continue_and_while_else(self):
        self.assertPythonResult("""
i = 0
total = 0
while i < 3:
    i += 1
    j = 0
    while j < 4:
        j += 1
        if j == 2:
            continue
        if i == 2 and j == 3:
            break
        total += i * j
    else:
        total += 100
else:
    total += 1000
""", max_ticks=5000)

    def test_comparisons_and_short_circuit(self):
        for operator in [">", "<", ">=", "<=", "==", "!="]:
            for value in [-1, 0, 1]:
                self.assertPythonResult(f"x = {value}\nif x {operator} 0:\n    y = 2\nelse:\n    y = -2")
        self.assertPythonResult("""
a = array(1)
i = 4
x = 0
if True or a[i] > 0:
    x = 1
if False and a[i] > 0:
    x = 9
if not (x < 0):
    x += 1
""")

    def test_index_expressions_and_augmented_assignment(self):
        self.assertPythonResult("""
a = array(4)
i = 0
while i < 3:
    a[i + 1] += i * 2
    i += 1
x = a[a[1] + 2]
""")

    def test_declared_array_spelling(self):
        net = compile_structured((ROOT / "examples/array_loop.flow").read_text())
        values = net.run()
        self.assertEqual([values[f"squares[{i}]"] for i in range(4)], [0, 1, 4, 9])

    def test_unrolled_constant_indices_have_no_dynamic_selectors(self):
        source = """
a = array(8)
for i in range(4):
    a[i * 2] = i * i
    a[i * 2 + 1] = i + 10
"""
        net, values = self.assertPythonResult(source, max_steps=1)
        program = parse(lower(source))
        from loopnet.language import ArrayAccess
        self.assertFalse(any(isinstance(operand, ArrayAccess)
                             for instruction in program.instructions
                             for operand in (instruction.dest, instruction.left, instruction.right)))
        self.assertEqual([values[f"a[{i}]"] for i in range(8)], [0, 10, 1, 11, 4, 12, 9, 13])
        self.assertLess(len(net.schedule), 15)

    def test_unrolled_ranges_steps_empty_and_else(self):
        self.assertPythonResult("""
i = 9
total = 0
for i in range(0):
    total = -1
else:
    total += 100
for j in range(6, -2, -2):
    total += j
for k in range(-2, 4, 2):
    total += k
for unused in range(4, 1):
    total = -9
""")

    def test_unrolled_nested_ranges_break_continue_and_else(self):
        self.assertPythonResult("""
total = 0
for i in range(4):
    for j in range(i + 1):
        if j == 1:
            continue
        if i == 3 and j == 2:
            break
        total += i * 10 + j
    else:
        total += 100
else:
    total += 1000
for k in range(5):
    if k == 2:
        break
else:
    total = -1
""")

    def test_unrolled_loop_variable_mutation_and_reuse(self):
        self.assertPythonResult("""
total = 0
for i in range(3):
    i += 10
    total += i
for j in range(2):
    for j in range(3):
        total += j
    total += j
""")

    def test_unrolled_loop_variable_modified_by_called_function(self):
        self.assertPythonResult("""
def bump():
    global i
    i += 10

total = 0
for i in range(3):
    total += i
    bump()
    total += i
""")

    def test_unrolled_loops_in_shared_and_inline_functions(self):
        for prefix in ("", "inline "):
            with self.subTest(prefix=prefix):
                self.assertPythonResult(prefix + """def calculate(x):
    result = 0
    for i in range(4):
        j = 0
        while j < 3:
            j += 1
            if j == 2:
                continue
            if i == 2:
                break
            result += x * i + j
    return result, i

a, b = calculate(2)
c, d = calculate(3)
""", max_stack_depth=2)

    def test_invalid_or_excessive_unrolled_ranges(self):
        sources = ["for i in range(n):\n    pass", "for i in range(1.5):\n    pass",
                   "for i in range(True):\n    pass", "for i in range(0, 4, 0):\n    pass",
                   "for i in range():\n    pass", "for i in range(1, 2, 3, 4):\n    pass",
                   "for i in range(stop=4):\n    pass", "for i, j in range(4):\n    pass",
                   "for i in range(9007199254740993, 9007199254740994):\n    pass",
                   "for i in range(4097):\n    pass",
                   "for i in range(65):\n    for j in range(65):\n        pass"]
        for source in sources:
            with self.subTest(source=source), self.assertRaises(StructuredError):
                lower(source)

    def test_interpreter_agrees_at_every_recurrence_and_terminate_is_absorbing(self):
        source = (ROOT / "examples/functions.flow").read_text()
        for steps in [1, 4, 32]:
            net = compile_structured(source, max_steps=steps, execution="instruction")
            reference = Interpreter(parse(lower(source)))
            state = net.initial_state()
            self.assertEqual(state[net.terminate_channel], 0.0)
            for _ in range(200):
                state = net(state)
                net.assert_valid(state)
                self.assertEqual(net.globals(state, include_internal=True), reference.step(max_steps=steps))
                self.assertEqual(net.terminated(state), reference.terminated)
                if net.terminated(state):
                    self.assertEqual(state[net.terminate_channel], 1.0)
                    self.assertEqual(net(state), state)
                    break
            else:
                self.fail("Program did not finish")

    def test_empty_program_and_early_top_level_return(self):
        self.assertEqual(compile_structured("").run(), {})
        self.assertEqual(compile_structured("x = 3\nreturn\nx = 9").run()["x"], 3)

    def test_loop_can_exceed_one_forward_pass_budget(self):
        net = compile_structured("n = 30\nwhile n > 0:\n    n -= 1", max_steps=1)
        with self.assertRaises(ExecutionLimitError):
            net.run(max_ticks=2)
        self.assertEqual(net.run(max_ticks=1000)["n"], 0)
        with self.assertRaises(ValueError):
            net.step()
        with self.assertRaises(ValueError):
            parse(lower("x = 1")).step()

    def test_branch_margin_and_array_index_contracts_survive_lowering(self):
        with self.assertRaisesRegex(NetworkContractError, "margin"):
            compile_structured("if x > 0:\n    y = 1").run({"x": 1e-8})
        with self.assertRaisesRegex(NetworkContractError, "array index"):
            compile_structured("a = array(2)\nx = a[i]").run({"i": 0.5})

    def test_raw_positive_conditions_and_primitive_halt(self):
        self.assertEqual(compile_structured("x = -1\nwhile x:\n    x -= 1").run()["x"], -1)
        net = compile_program("x = x + 1\nhalt", max_steps=2)
        state = net(net.initial_state())
        net.assert_valid(state)
        self.assertTrue(net.terminated(state))
        again = net(state)
        net.assert_valid(again)
        self.assertEqual(net.globals(again), {"x": 1})
        self.assertTrue(net.terminated(again))

    def test_ordinary_function_return_does_not_terminate_recurrence(self):
        net = compile_program("x = x + 1\nreturn", max_steps=2)
        state = net.initial_state()
        for i in range(3):
            state = net(state)
            net.assert_valid(state)
            self.assertFalse(net.terminated(state))
            self.assertEqual(net.globals(state)["x"], i + 1)

    def test_unsupported_and_invalid_source(self):
        sources = ["import os", "for x in y:\n    pass", "x = 1 / 2", "x = 'text'",
                   "def f(x=2):\n    return x\nx = f()", "x = missing(2)",
                   "def f(x):\n    return x\na, b = f(1)",
                   "def f(x):\n    return x\na = f()",
                   "def f():\n    return 1, 2\na = 1 + f()",
                   "inline def f(x):\n    return f(x)\ny = f(2)",
                   "inline def f():\n    g()\ninline def g():\n    f()\nf()",
                   "def f(x):\n    if x > 0:\n        return x\ny = f(1)",
                   "def f(x):\n    if x > 0:\n        return x\n    return 1, 2\ny = f(1)",
                   "break", "continue", "return 1", "__ln_tmp_1 = 3",
                   "a = array(2)\ndef f(a):\n    return a\nx = f(1)",
                   "def f():\n    a = array(2)\nf()",
                   "x = array(n)", "x = array(0)", "x = True", "x = 1e999",
                   "def f(x):\n    def g():\n        return x\n    return g()"]
        for source in sources:
            with self.subTest(source=source), self.assertRaises(StructuredError):
                lower(source)

    def test_random_small_programs_against_python(self):
        rng = random.Random(347)
        for _ in range(12):
            count, scale = rng.randrange(1, 7), rng.choice([-1.5, -0.5, 0.5, 2.5])
            self.assertPythonResult(f"""
def pair(x, y):
    return x + y, x * y
n = {count}
total = 0
while n > 0:
    a, b = pair(n, {scale})
    total += a + b
    n -= 1
""")

    def test_weight_export_keeps_continuation_and_termination(self):
        net = compile_structured("x = 3\nwhile x > 0:\n    x -= 1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flow.json"
            net.save(path)
            loaded = Network.load(path)
            self.assertEqual(loaded.summary(), net.summary())
            self.assertEqual(loaded.run(), {"x": 0})
            data = json.loads(path.read_text())
            self.assertIsInstance(data["terminate_channel"], int)
            self.assertNotIn("instructions", data)

    def test_lower_run_and_export_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            lowered, weights = Path(directory) / "program.loop", Path(directory) / "network.json"
            commands = [
                ["lower", "examples/functions.flow", "-o", str(lowered)],
                ["run", str(lowered), "--max-steps", "8", "--compare"],
                ["compile", "examples/functions.flow", "-o", str(weights), "--max-steps", "8"],
                ["run-net", str(weights)],
                ["run", "examples/functions.flow", "--max-steps", "8", "--compare"],
            ]
            for command in commands:
                result = subprocess.run([sys.executable, "-m", "loopnet", *command], cwd=ROOT,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                if command[0] in {"run", "run-net"}:
                    lines = result.stdout.splitlines()
                    self.assertTrue(all(json.loads(line)["terminate"] == 0.0 for line in lines[:-1]))
                    record = json.loads(lines[-1])
                    self.assertEqual(record["terminate"], 1.0)
                    self.assertEqual(record["globals"]["total"], 38)
            failure = subprocess.run([sys.executable, "-m", "loopnet", "run", "examples/functions.flow",
                                      "--ticks", "1"], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(failure.returncode, 1)
            self.assertIn("did not terminate", failure.stderr)


if __name__ == "__main__":
    unittest.main()
