import importlib.util
import math
from pathlib import Path
import unittest

from loopnet import NetworkContractError, compile_program, compile_structured


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch not installed")
class TorchExportTests(unittest.TestCase):
    def test_recurrence_and_batch(self):
        import torch

        net = compile_program("x = x + 1\nif x goto yes\ny = -1\nreturn\nyes:\ny = x * x",
                              max_steps=5)
        model = net.to_torch()
        initial = [{"x": -3}, {"x": 0}, {"x": 0.25}]
        states = [net.initial_state(values) for values in initial]
        tensor = torch.tensor(states, dtype=torch.float64)
        with torch.no_grad():
            for _ in range(10):
                states = [net(state) for state in states]
                tensor = model(tensor)
                for expected, actual in zip(states, tensor.tolist()):
                    net.assert_valid(actual)
                    for left, right in zip(expected, actual):
                        self.assertTrue(math.isclose(left, right, abs_tol=1e-12, rel_tol=1e-12))

    def test_overwrite_and_inactive_overflow(self):
        import torch

        net = compile_program("goto end\ny = x * x\ny = x * 1e300\nend:\nx = 3", max_steps=3)
        model = net.to_torch()
        state = torch.tensor(net.initial_state({"x": 1e300}), dtype=torch.float64)
        with torch.no_grad():
            result = model(state).tolist()
        net.assert_valid(result)
        self.assertEqual(net.globals(result)["x"], 3.0)

    def test_signed_product_gradient(self):
        import torch

        net = compile_program("product = x * y", max_steps=2)
        model = net.to_torch()
        self.assertTrue(all(parameter.dtype == torch.float64 for parameter in model.parameters()))
        state = torch.tensor(net.initial_state({"x": -2.5, "y": 3.25}),
                             dtype=torch.float64, requires_grad=True)
        output = model(state)
        output[net.variables.index("product")].backward()
        self.assertEqual(state.grad[net.variables.index("x")].item(), 3.25)
        self.assertEqual(state.grad[net.variables.index("y")].item(), -2.5)

    def test_array_recurrence_and_batch(self):
        import torch

        source = (Path(__file__).resolve().parents[1] / "examples/arrays.loop").read_text()
        net = compile_program(source, max_steps=7)
        states = [net.initial_state({"i": i, "bins[0]": 5}) for i in (0, 1, 3)]
        tensor = torch.tensor(states, dtype=torch.float64)
        model = net.to_torch()
        with torch.no_grad():
            for _ in range(12):
                states = [net(state) for state in states]
                tensor = model(tensor)
                for expected, actual in zip(states, tensor.tolist()):
                    net.assert_valid(actual)
                    self.assertEqual(net.globals(actual), net.globals(expected))

    def test_array_index_diagnostics_and_overwrites(self):
        import torch

        net = compile_program("array a[3]\na[i] = a[j]", max_steps=2)
        model = net.to_torch()
        with torch.no_grad():
            state = net.initial_state({"i": 1, "j": 2, "a[1]": 1e300, "a[2]": 3})
            result = model(torch.tensor(state, dtype=torch.float64)).tolist()
            net.assert_valid(result)
            self.assertEqual(net.globals(result)["a[1]"], 3)
            for index in [-1e300, -math.ulp(0.0), math.ulp(0.0), 0.5,
                          math.nextafter(1.0, 0.0), math.nextafter(1.0, math.inf),
                          math.nextafter(2.0, math.inf), 3, 1e300]:
                state = net.initial_state({"i": index})
                result = model(torch.tensor(state, dtype=torch.float64)).tolist()
                with self.assertRaisesRegex(NetworkContractError, "array index"):
                    net.assert_valid(result)

    def test_array_gather_scatter_gradient(self):
        import torch

        net = compile_program("array a[3]\na[i] = a[j] * x\ny = a[i]", max_steps=3)
        model = net.to_torch()
        state = torch.tensor(net.initial_state({"i": 1, "j": 2, "a[1]": 8, "a[2]": -3, "x": 2}),
                             dtype=torch.float64, requires_grad=True)
        model(state)[net.variables.index("y")].backward()
        self.assertEqual(state.grad[net.variables.index("a[2]")].item(), 2)
        self.assertEqual(state.grad[net.variables.index("a[1]")].item(), 0)
        self.assertEqual(state.grad[net.variables.index("x")].item(), -3)

    def test_structured_batch_termination_and_fixed_points(self):
        import torch

        source = """
def pair(x):
    return x + 1, x * x
total = 0
while n > 0:
    a, b = pair(n)
    total += a + b
    n -= 1
"""
        net = compile_structured(source, max_steps=8)
        states = [net.initial_state({"n": n}) for n in (1, 3)]
        tensor = torch.tensor(states, dtype=torch.float64)
        model = net.to_torch()
        self.assertEqual(model.terminate_channel, net.terminate_channel)
        self.assertTrue(model.continuation)
        with torch.no_grad():
            for _ in range(100):
                states = [net(state) for state in states]
                tensor = model(tensor)
                for expected, actual in zip(states, tensor.tolist()):
                    net.assert_valid(actual)
                    self.assertEqual(net.globals(actual), net.globals(expected))
                    self.assertEqual(net.terminated(actual), net.terminated(expected))
                if all(net.terminated(state) for state in states):
                    self.assertTrue(torch.equal(model(tensor), tensor))
                    break
            else:
                self.fail("Batch did not terminate")
        self.assertEqual([net.globals(state)["total"] for state in states], [3, 23])

    def test_structured_function_gradient_across_recurrence(self):
        import torch

        net = compile_structured("def square(x):\n    return x * x\nresult = square(x)", max_steps=4)
        model = net.to_torch()
        initial = torch.tensor(net.initial_state({"x": -3}), dtype=torch.float64, requires_grad=True)
        state = initial
        for _ in range(30):
            state = model(state)
            if state[net.terminate_channel].item() == 1:
                break
        else:
            self.fail("Function did not terminate")
        self.assertEqual(state[net.variables.index("result")].item(), 9)
        state[net.variables.index("result")].backward()
        self.assertEqual(initial.grad[net.variables.index("x")].item(), -6)

    def test_unrolled_static_arrays_and_gradient(self):
        import torch

        net = compile_structured("""
array a[3]
for i in range(3):
    a[i] = x * (i + 1)
total = a[0] + a[1] + a[2]
""", max_steps=4)
        model = net.to_torch()
        initial = torch.tensor(net.initial_state({"x": -2}), dtype=torch.float64, requires_grad=True)
        state = initial
        for _ in range(30):
            state = model(state)
            if state[model.terminate_channel].item() == 1:
                break
        else:
            self.fail("Unrolled program did not terminate")
        net.assert_valid(state.tolist())
        self.assertEqual(net.globals(state.tolist())["total"], -12)
        state[net.variables.index("total")].backward()
        self.assertEqual(initial.grad[net.variables.index("x")].item(), 6)

    def test_inline_keyword_matches_shared_runtime_calls(self):
        import torch

        source = "def pair(x, y):\n    return x + y, x * y\na, b = pair(2, 3)\nc, d = pair(4, 5)"
        for prefix in ("", "inline "):
            net = compile_structured(prefix + source, max_steps=8, max_stack_depth=2)
            model = net.to_torch()
            state = torch.tensor(net.initial_state(), dtype=torch.float64)
            with torch.no_grad():
                for _ in range(30):
                    state = model(state)
                    values = state.tolist()
                    net.assert_valid(values)
                    if net.terminated(values):
                        self.assertTrue(torch.equal(model(state), state))
                        break
                else:
                    self.fail("Program did not terminate")
            self.assertEqual(net.globals(state.tolist()), {"a": 5, "b": 6, "c": 9, "d": 20})


if __name__ == "__main__":
    unittest.main()
