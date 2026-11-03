import importlib.util
import math
import unittest

from loopnet import Interpreter, StructuredError, compile_program, compile_structured, lower, parse


class RendererPrimitiveTests(unittest.TestCase):
    def test_relu_is_exact_without_a_branch_gap(self):
        net = compile_program("array a[2]\na[i] = relu(x)\ny = relu(a[i])", max_steps=3)
        for x in (-1e300, -1e-12, 0, math.ulp(0.0), 1e-12, 2, 1e300):
            initial = {"i": 1, "x": x, "a[1]": 1e300}
            expected = parse("array a[2]\na[i] = relu(x)\ny = relu(a[i])").step(initial, max_steps=3)
            self.assertEqual(net.step(initial), expected)

    def test_inline_leaf_helpers_do_not_split_regions_or_keep_parameter_slots(self):
        source = "inline def positive(x):\n    return relu(x)\ny = positive(v) + positive(v*2)"
        net = compile_structured(source)
        self.assertEqual(net.width, 9)  # v, y, seven control channels
        self.assertEqual(len(net.regions), 2)
        for v in (-2, 0, 1e-12, 2):
            self.assertEqual(net.run({"v": v})["y"], max(0, v) + max(0, v * 2))

    def test_configurable_unroll_budget_and_relu_validation(self):
        with self.assertRaises(StructuredError):
            lower("for i in range(21):\n    x=i", max_unrolled_iterations=20)
        self.assertIn("20.0", lower("for i in range(21):\n    x=i", max_unrolled_iterations=21))
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                lower("", max_unrolled_iterations=value)
        for source in ("x=relu()", "x=relu(1,2)", "def relu(x):\n    return x"):
            with self.assertRaises(StructuredError):
                lower(source)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch not installed")
    def test_relu_pytorch_values_and_gradients(self):
        import torch
        net = compile_structured("y=relu(x)")
        model = net.to_torch()
        for x in (-2, 2):
            state = torch.tensor(net.initial_state({"x": x}), dtype=torch.float64, requires_grad=True)
            output = model(state)
            output[net.variables.index("y")].backward()
            self.assertEqual(state.grad[net.variables.index("x")].item(), int(x > 0))


if __name__ == "__main__":
    unittest.main()
