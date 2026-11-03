import tempfile
from pathlib import Path
import unittest

from loopnet import (ControlFlowError, Interpreter, LanguageError, Network, NetworkContractError,
                     StackError, StructuredError, compile_program, compile_structured, lower, parse)
import test_structured


FACTORIAL = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
"""


class CallTests(unittest.TestCase):
    def assertPythonResult(self, source, **kwargs):
        return test_structured.StructuredTests().assertPythonResult(source, **kwargs)

    def test_recursive_factorial_and_exact_stack_capacity(self):
        net, result = self.assertPythonResult(FACTORIAL, max_steps=8, max_stack_depth=5)
        self.assertEqual(result["result"], 120)
        self.assertEqual(net.max_stack_depth, 5)
        with self.assertRaisesRegex(NetworkContractError, "stack"):
            compile_structured(FACTORIAL, max_steps=8, max_stack_depth=4).run()

    def test_two_recursive_calls_preserve_expression_temporaries(self):
        self.assertPythonResult("""
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

result = fib(6)
""", max_steps=8, max_stack_depth=6)

    def test_recursive_tuple_outputs_preserve_locals(self):
        _, values = self.assertPythonResult("""
def stats(n):
    if n <= 0:
        return 0, 1
    total, product = stats(n - 1)
    return total + n, product * (n + 1)

a, b = stats(4)
""", max_steps=8, max_stack_depth=5)
        self.assertEqual((values["a"], values["b"]), (10, 120))

    def test_mutual_recursion_and_forwarded_returns(self):
        self.assertPythonResult("""
def even(n):
    if n <= 0:
        return 1
    return odd(n - 1)

def odd(n):
    if n <= 0:
        return 0
    return even(n - 1)

a = even(6)
b = odd(5)
""", max_steps=8, max_stack_depth=7)

    def test_recursive_void_function(self):
        self.assertPythonResult("""
def visit(n):
    global count
    count += 1
    if n > 0:
        visit(n - 1)

count = 0
visit(4)
""", max_steps=8, max_stack_depth=5)

    def test_inline_and_dynamic_functions_can_be_mixed(self):
        self.assertPythonResult("""
inline def twice(x):
    return x * 2

def walk(n):
    if n <= 0:
        return 0
    return twice(n) + walk(n - 1)

a = walk(4)
b = twice(3)
""", max_steps=8, max_stack_depth=5)

    def test_inline_function_can_call_dynamic_function_recursively(self):
        self.assertPythonResult("""
inline def descend(n):
    return worker(n)

def worker(n):
    if n <= 1:
        return 1
    return n * descend(n - 1)

result = descend(5)
""", max_steps=8, max_stack_depth=5)

    def test_expansion_cycles_are_rejected(self):
        for source in ["inline " + FACTORIAL.lstrip(), """
inline def a(x):
    return b(x)
inline def b(x):
    return a(x)
y = a(1)
"""]:
            with self.assertRaisesRegex(StructuredError, "inline expansion"):
                lower(source)

    def test_only_one_shared_body_is_emitted_and_inline_has_no_frame(self):
        body = "def pair(x, y):\n    return x + y, x * y\na, b = pair(2, 3)\nc, d = pair(4, 5)"
        dynamic = lower(body, max_stack_depth=2)
        self.assertEqual(dynamic.count("__ln_function_pair:"), 1)
        self.assertEqual(dynamic.count("goto __ln_function_pair"), 2)
        self.assertIn("goto *__ln_ip", dynamic)
        expanded = lower("inline " + body)
        self.assertNotIn("__ln_function_pair:", expanded)
        self.assertNotIn("stack __ln_sp", expanded)
        self.assertNotIn("goto *__ln_ip", expanded)
        net, _ = self.assertPythonResult("inline " + body)
        self.assertIsNone(net.max_stack_depth)

    def test_break_exits_only_the_innermost_loop_in_a_dynamic_function(self):
        self.assertPythonResult("""
def work(n):
    total = 0
    while n > 0:
        j = 0
        while j < 4:
            j += 1
            if j == 2:
                break
            total += 1
        else:
            total += 100
        n -= 1
    return total

i = 0
answer = 0
while i < 3:
    answer += work(i)
    i += 1
""", max_steps=8)

    def test_inline_break_and_continue(self):
        self.assertPythonResult("""
inline def work(n):
    total = 0
    while n > 0:
        n -= 1
        if n == 2:
            continue
        if n == 1:
            break
        total += n
    return total

a = work(5)
b = work(3)
""", max_steps=8)

    def test_recursion_and_break_keep_the_callers_loop_state(self):
        self.assertPythonResult("""
def work(n):
    if n <= 0:
        return 0
    while n > 0:
        result = work(n - 1)
        break
    return result + n

a = work(4)
""", max_steps=8, max_stack_depth=5)

    def test_reference_and_network_agree_through_recursive_returns(self):
        for steps in (1, 8):
            primitive = lower(FACTORIAL, max_stack_depth=5)
            reference = Interpreter(parse(primitive))
            net = compile_program(primitive, max_steps=steps)
            state = net.initial_state()
            for _ in range(400):
                state = net(state)
                net.assert_valid(state)
                self.assertEqual(net.globals(state, include_internal=True), reference.step(max_steps=steps))
                self.assertEqual(net.terminated(state), reference.terminated)
                if net.terminated(state):
                    self.assertEqual(net(state), state)
                    break
            else:
                self.fail("Recursive program did not terminate")

    def test_stack_errors_are_reported_before_index_errors(self):
        source = lower(FACTORIAL, max_stack_depth=2)
        reference = Interpreter(parse(source))
        with self.assertRaises(StackError):
            for _ in range(100):
                reference.step(max_steps=8)
        with self.assertRaisesRegex(NetworkContractError, "stack"):
            compile_program(source, max_steps=32).run()
        for pointer in (-1, 0.5, 3):
            source = f"continuation\nstack sp 2\nsp = {pointer}\nhalt"
            with self.assertRaises(StackError):
                Interpreter(parse(source)).step(max_steps=4)
            with self.assertRaisesRegex(NetworkContractError, "stack"):
                compile_program(source, max_steps=4).run()

    def test_label_address_constants_and_indirect_jumps(self):
        source = "continuation\nip = &finish\ngoto *ip\nx = 99\nfinish:\nx = 7\nhalt"
        self.assertEqual(compile_program(source, max_steps=4).run()["x"], 7)
        for target in (-1, 0.5, 100):
            source = "continuation\ngoto *ip"
            with self.assertRaises(ControlFlowError):
                Interpreter(parse(source), {"ip": target}).step()
            with self.assertRaisesRegex(NetworkContractError, "indirect jump"):
                compile_program(source, max_steps=1).run({"ip": target})
        for source in ("ip = &missing", "stack sp 0", "stack sp 2\nstack sp 3"):
            with self.assertRaises(LanguageError):
                parse(source)

    def test_stack_configuration_validation(self):
        for depth in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                lower("x = 1", max_stack_depth=depth)

    def test_weight_export_retains_runtime_frames_and_diagnostics(self):
        net = compile_structured(FACTORIAL, max_steps=8, max_stack_depth=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.json"
            net.save(path)
            loaded = Network.load(path)
            self.assertEqual(loaded.summary(), net.summary())
            self.assertEqual(loaded.run()["result"], 120)
            self.assertEqual(loaded.max_stack_depth, 5)
            self.assertIsNotNone(loaded.stack_channel)


if __name__ == "__main__":
    unittest.main()
