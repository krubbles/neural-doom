import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from loopnet import Interpreter, Network, NetworkContractError, compile_program, compile_structured, lower, parse
from loopnet.regions import reference_step


ROOT = Path(__file__).resolve().parents[1]


class CompactStateTests(unittest.TestCase):
    def assertRegionTrace(self, source, initial=None, *, structured=True, steps=1, stack=4):
        primitive = lower(source, max_stack_depth=stack) if structured else source
        program = parse(primitive)
        net = (compile_structured(source, max_steps=steps, max_stack_depth=stack) if structured else
               compile_program(program, max_steps=steps, execution="block"))
        reference = Interpreter(program, initial)
        state = net.initial_state(initial)
        for _ in range(1000):
            state = net(state)
            net.assert_valid(state)
            expected = reference_step(reference, net.regions, steps)
            self.assertEqual(net.globals(state, include_internal=True),
                             {name: expected[name] for name in net.variables})
            self.assertEqual(net.terminated(state), reference.terminated)
            if net.terminated(state):
                self.assertEqual(net(state), state)
                return net, state
        self.fail("Basic-block program did not terminate")

    def test_independent_unrolled_work_grows_hidden_width_not_control_or_scratch_state(self):
        nets = []
        for size in (8, 64):
            source = f"array a[{size}]\nfor i in range({size}):\n    a[i] = x * (i + 1)"
            net, state = self.assertRegionTrace(source, {"x": -2})
            self.assertEqual(net.width, size + 2 + 7)  # array, i/x, seven control cells
            self.assertEqual(net.globals(state)[f"a[{size-1}]"], -2 * size)
            self.assertEqual(len(net.regions), 2)  # one whole straight-line region plus dead fallthrough
            nets.append(net)
        self.assertEqual(nets[0].summary()["feedforward_layers"], nets[1].summary()["feedforward_layers"])
        self.assertGreater(nets[1].summary()["peak_hidden_width"], nets[0].summary()["peak_hidden_width"])

    def test_block_array_aliases_and_tiny_overwrites_of_huge_values(self):
        source = """
array a[3]
a[i] = 3
first = a[i]
a[j] = a[i] * 2
a[i] = a[j] + 1
last = a[i] + a[j]
"""
        for j in (0, 1):
            net, state = self.assertRegionTrace(source, {"i": 0, "j": j, "a[0]": 1e300, "a[1]": 1e300})
            values = net.globals(state)
            self.assertEqual(values["first"], 3)
            self.assertEqual(values["last"], 14 if j == 0 else 13)

    def test_recursive_live_values_stay_banked_but_local_temporaries_do_not(self):
        source = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
result = factorial(4)
"""
        for steps in (1, 4):
            net, state = self.assertRegionTrace(source, steps=steps)
            self.assertEqual(net.globals(state)["result"], 24)
            instruction = compile_structured(source, execution="instruction", max_stack_depth=4)
            self.assertLess(net.width, instruction.width)
            self.assertTrue(any(name.startswith("__ln_tmp_") for name, _ in net.arrays))

    def test_short_circuit_loop_and_multiple_call_boundaries(self):
        self.assertRegionTrace("""
array a[3]
def change(x):
    a[0] += x
    return a[0], x * x

n = 3
total = 0
while n > 0:
    x, y = change(n)
    if n == 2 or (x > 0 and y > 0):
        total += x + y
    n -= 1
""", steps=3)

    def test_generic_numeric_indirect_jumps_can_enter_any_instruction(self):
        source = "continuation\nip = 2\ngoto *ip\nx = 7\nhalt"
        net, state = self.assertRegionTrace(source, structured=False)
        self.assertEqual(len(net.regions), len(parse(source).instructions))
        self.assertEqual(net.globals(state)["x"], 7)

    def test_function_mode_halt_is_a_full_state_fixed_point(self):
        for execution in ("instruction", "block"):
            net = compile_program("x = x + 1\nhalt", execution=execution, max_steps=3)
            state = net(net.initial_state())
            net.assert_valid(state)
            self.assertTrue(net.terminated(state))
            self.assertEqual(net(state), state)

    def test_dead_load_still_checks_indices_and_invalid_stack_cannot_be_repaired_silently(self):
        indexed = compile_program("continuation\narray a[2]\nx = a[i]\nx = 3\nhalt", execution="block", max_steps=1)
        for index in (-1, 0.5, 2):
            with self.assertRaisesRegex(NetworkContractError, "array index"):
                indexed.run({"i": index})
        stack = compile_program("continuation\nstack sp 2\nsp = 3\nsp = 0\nhalt", execution="block", max_steps=1)
        with self.assertRaisesRegex(NetworkContractError, "stack"):
            stack.run()

    def test_user_arrays_resembling_temporary_names_keep_their_bounds_checks(self):
        source = "continuation\nstack sp 3\narray __ln_tmp_user[1]\nsp = 2\n__ln_tmp_user[sp] = 3\nx = __ln_tmp_user[sp]\nhalt"
        net = compile_program(source, execution="block", max_steps=1, optimize_temporaries=True)
        with self.assertRaisesRegex(NetworkContractError, "array index"):
            net.run()

    def test_primitive_global_names_are_preserved_even_if_they_resemble_frontend_names(self):
        net = compile_program("__ln_tmp_user = 3\nx = 2", execution="block", max_steps=2)
        self.assertEqual(net.step(), {"__ln_tmp_user": 3, "x": 2})

    def test_export_retains_architecture_and_regions_without_source_operations(self):
        net = compile_structured("x = 2\ny = x*x\nif y > 0:\n    z = y+1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            net.save(path)
            data = json.loads(path.read_text())
            self.assertEqual(data["format"], "loopnet.reglu.v5")
            self.assertNotIn("instructions", data)
            self.assertEqual(Network.load(path).summary(), net.summary())
            self.assertEqual(Network.load(path).run()["z"], 5)
        self.assertEqual(net.width, len(net.variables) + 7)
        self.assertGreater(net.summary()["transient_channels"], 0)
        self.assertGreater(net.summary()["feedforward_layers"], net.summary()["residual_blocks"])

    @unittest.skipUnless(shutil.which("node"), "Node is needed for the browser activation-lifetime check")
    def test_browser_hidden_buffers_can_be_reused_across_unrelated_states(self):
        net = compile_structured("array a[3]\na[i] = x * x\ny = a[i] + 1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            net.save(path)
            script = """
import fs from 'node:fs';
import {ResidualNetwork} from './loopnet/web/neural.mjs';
const n = new ResidualNetwork(JSON.parse(fs.readFileSync(process.argv[1], 'utf8')));
const values = [];
for (const x of [-3, 0, 2, -1]) {
  n.work.fill(NaN);
  const state = n.initialState({i:1, x});
  n.run(state);
  if (state.length !== n.model.width) throw new Error('Hidden state leaked');
  values.push(n.read(state,'y'));
}
console.log(JSON.stringify(values));
"""
            result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script, str(path)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [10, 1, 5, 2])


if __name__ == "__main__":
    unittest.main()
