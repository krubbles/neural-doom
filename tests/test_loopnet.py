import itertools
import math
from pathlib import Path
import random
import tempfile
import unittest

from loopnet import (ExecutionLimitError, Interpreter, LanguageError, Network, NetworkContractError,
                     compile_program, parse)


class CompilerTests(unittest.TestCase):
    def assertMatches(self, source, initial=None, *, ticks=1, max_steps=64):
        program = parse(source)
        network = compile_program(program, max_steps=max_steps)
        state = network.initial_state(initial)
        expected = initial
        for _ in range(ticks):
            expected = program.step(expected, max_steps=max_steps)
            state = network(state)
            network.assert_valid(state)
            for name, actual in network.globals(state).items():
                self.assertTrue(math.isclose(actual, expected[name], abs_tol=1e-12, rel_tol=1e-12),
                                (name, actual, expected[name]))
        return network.globals(state)

    def test_all_five_assignments_and_numeric_addresses(self):
        self.assertMatches("""
            [0] = -2.5
            x = [0] + 4
            sum = x + [0]
            scaled = sum * -3
            product = x * scaled
        """)

    def test_signed_multiplication(self):
        net = compile_program("product = x * y", max_steps=2)
        numbers = [-17.25, -1.0, -1e-7, 0.0, 1e-7, 0.25, 21.5]
        for x, y in itertools.product(numbers, repeat=2):
            self.assertEqual(net.step({"x": x, "y": y})["product"], x * y)

    def test_self_multiplication_and_addition(self):
        self.assertMatches("x = x * x\nx = x + x\nx = x * -0.5", {"x": -1.5})

    def test_persistent_state_and_reset_control(self):
        path = Path(__file__).resolve().parents[1] / "examples/counter.loop"
        result = self.assertMatches(path.read_text(), ticks=100)
        self.assertEqual(result["cycles"], 16.0)
        self.assertEqual(result["count"], 4.0)

    def test_backward_jump_and_budget_counts_return(self):
        source = "n = 3\nagain:\nx = x + 1\nn = n + -1\nif n goto again\nreturn"
        result = self.assertMatches(source, ticks=5, max_steps=11)
        self.assertEqual(result["x"], 15.0)
        net = compile_program(source, max_steps=10)
        with self.assertRaises(NetworkContractError):
            net.step()
        with self.assertRaises(ExecutionLimitError):
            parse(source).step(max_steps=10)

    def test_unconditional_jump_and_trailing_label(self):
        result = self.assertMatches("goto skip\nx = 100\nskip:\nx = x + 1\ngoto end\nx = 0\nend:")
        self.assertEqual(result["x"], 1.0)

    def test_strict_positive_branch_and_boundaries(self):
        source = "if x > 0 goto yes\ny = -1\nreturn\nyes:\ny = 1"
        net = compile_program(source)
        for x in [-1e300, -1e-12, -0.0, 0.0]:
            self.assertEqual(net.step({"x": x})["y"], -1.0)
        for x in [1e-4, math.nextafter(1e-4, math.inf), 0.3, 1e300]:
            self.assertEqual(net.step({"x": x})["y"], 1.0)
        for x in [math.ulp(0.0), 1e-12, 5e-5, math.nextafter(1e-4, 0.0)]:
            with self.assertRaises(NetworkContractError, msg=str(x)):
                net.step({"x": x})

    def test_unreachable_branch_does_not_violate_margin(self):
        self.assertMatches("goto end\nif x goto end\nend:", {"x": 1e-8})

    def test_unreachable_multiplication_does_not_overflow(self):
        self.assertMatches("goto end\ny = x * x\nz = x * 1e300\nend:", {"x": 1e300})

    def test_small_overwrite_of_huge_previous_value(self):
        result = self.assertMatches("x = 3", {"x": 1e300})
        self.assertEqual(result["x"], 3.0)

    def test_logistic_recurrence(self):
        path = Path(__file__).resolve().parents[1] / "examples/logistic.loop"
        self.assertMatches(path.read_text(), {"x": 0.25}, ticks=200)

    def test_random_programs_and_branch_paths(self):
        rng = random.Random(812)
        for _ in range(30):
            lines = []
            for _ in range(8):
                target = rng.choice(["a", "b", "c"])
                left = rng.choice(["a", "b", "c"])
                right = rng.choice(["a", "b", "c", "-0.5", "0.25"])
                lines.append(f"{target} = {left} {rng.choice(['+', '*'])} {right}")
            lines += ["if a goto positive", "b = -1", "goto end", "positive:", "b = 1", "end:"]
            self.assertMatches("\n".join(lines), {"a": 0.5, "b": -0.5, "c": 0.25})

    def test_export_without_source(self):
        net = compile_program("x = x + 1\ny = x * x")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            net.save(path)
            loaded = Network.load(path)
            self.assertEqual(loaded.summary(), net.summary())
            self.assertEqual(loaded.step({"x": -3}), net.step({"x": -3}))
            self.assertNotIn('"op"', path.read_text())

    def test_parallel_depth_does_not_grow_with_instruction_count(self):
        short = "continuation\nx = x * y\nif x goto end\nend:\nhalt"
        long = "continuation\n" + "x = x * y\n" * 100 + "if x goto end\nend:\nhalt"
        small = compile_program(short, max_steps=1)
        large = compile_program(long, max_steps=1)
        self.assertEqual(len(small.schedule), len(large.schedule))
        self.assertLess(len(large.schedule), 15)

    def test_parallel_circuits_preserve_every_instruction_boundary(self):
        program = parse("""
continuation
array a[3]
x = 3
i = 1
a[i] = a[j] * x
x = a[i]
if x goto positive
x = -1
goto end
positive:
a[j] = x + a[i]
end:
halt
""")
        for steps in (1, 2, 8):
            network = compile_program(program, max_steps=steps)
            for initial in ({"x": 1e300, "j": 2, "a[1]": 1e300, "a[2]": -2},
                            {"x": 1e300, "j": 2, "a[1]": 1e300, "a[2]": 2}):
                reference = Interpreter(program, initial)
                state = network.initial_state(initial)
                for _ in range(30):
                    state = network(state)
                    network.assert_valid(state)
                    self.assertEqual(network.globals(state), reference.step(max_steps=steps))
                    self.assertEqual(network.terminated(state), reference.terminated)
                    if reference.terminated:
                        self.assertEqual(network(state), state)
                        break
                else:
                    self.fail("Parallel network did not terminate")

    def test_empty_function_and_immediate_return(self):
        self.assertMatches("")
        self.assertMatches("return\nx = 100", {"x": 2.0})

    def test_compile_validation(self):
        for value in [0, -1, math.inf, math.nan, 1e-320]:
            with self.assertRaises(ValueError):
                compile_program("", branch_margin=value)
        for value in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                compile_program("", max_steps=value)


class ParserTests(unittest.TestCase):
    def test_invalid_source(self):
        for source in ["goto missing", "same:\nsame:", "x = y / 2", "x = 1e999", "x = y - 1",
                       "if x > 1 goto end\nend:", "x = __import__('os')"]:
            with self.assertRaises(LanguageError):
                parse(source)

    def test_comments_and_scientific_notation(self):
        self.assertEqual(parse("# comment\nx = +.2e-2 # float").step()["x"], 0.002)

    def test_initial_value_validation(self):
        net = compile_program("x = x + 1")
        for state in [{"unknown": 1}, {"x": math.nan}, {"x": math.inf}]:
            with self.assertRaises(ValueError):
                net.initial_state(state)


if __name__ == "__main__":
    unittest.main()
