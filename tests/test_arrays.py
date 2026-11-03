import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from loopnet import ArrayIndexError, LanguageError, Network, NetworkContractError, compile_program, parse


ROOT = Path(__file__).resolve().parents[1]


class ArrayTests(unittest.TestCase):
    def assertMatches(self, program, net, initial=None, *, ticks=1):
        expected = initial
        state = net.initial_state(initial)
        for _ in range(ticks):
            expected = program.step(expected, max_steps=net.max_steps)
            state = net(state)
            net.assert_valid(state)
            self.assertEqual(net.globals(state), expected)
        return net.globals(state)

    def test_declaration_allocates_zeroed_persistent_cells_without_instructions(self):
        program = parse("goto end\narray a[3]\nend:")
        self.assertEqual(len(program.instructions), 2)
        self.assertEqual(program.arrays, (("a", 3),))
        self.assertEqual(program.variables, ("a[0]", "a[1]", "a[2]"))
        net = compile_program(program, max_steps=2)
        self.assertEqual(net.step(), {"a[0]": 0, "a[1]": 0, "a[2]": 0})
        self.assertMatches(program, net, {"a[1]": 7}, ticks=3)

    def test_constant_indices_all_assignment_forms_and_copy(self):
        program = parse("""
            array a[5]
            a[0] = -2
            a[1.0] = a[0] + 3
            a[+2e0] = a[ 1 ] + a[0]
            a[3] = a[2] * -2
            a[4] = a[3] * a[0]
            x = a[4]
            y = x
        """)
        net = compile_program(program, max_steps=8)
        result = self.assertMatches(program, net)
        self.assertEqual([result[f"a[{i}]"] for i in range(5)], [-2, 1, -1, 2, -4])
        self.assertEqual(result["y"], -4)

    def test_dynamic_reads_and_writes_with_aliases(self):
        program = parse("array a[4]\na[i] = a[j] + a[k]\nresult = a[i] * a[j]")
        net = compile_program(program, max_steps=3)
        for i, j, k in itertools.product(range(4), repeat=3):
            initial = {"i": i, "j": j, "k": k, **{f"a[{n}]": n - 1.5 for n in range(4)}}
            self.assertMatches(program, net, initial)

    def test_dynamic_set_copy_and_scalar_multiplication(self):
        program = parse("""
            array a[3]
            a[i] = 2.5
            a[j] = a[i] * -2
            a[k] = a[j]
            x = a[k]
        """)
        net = compile_program(program, max_steps=5)
        for i, j, k in itertools.product(range(3), repeat=3):
            self.assertMatches(program, net, {"i": i, "j": j, "k": k})

    def test_different_arrays_and_numeric_global_index(self):
        program = parse("array a[2]\narray b[3]\nb[[0]] = a[i] * x\ny = b[j]")
        net = compile_program(program, max_steps=3)
        result = self.assertMatches(program, net, {"[0]": 2, "i": 1, "j": 2, "a[1]": -3, "x": 4})
        self.assertEqual(result["y"], -12)

    def test_index_uses_pre_assignment_memory(self):
        program = parse("array a[3]\ni = a[i]\na[i] = i + 1\ni = a[i] + -1")
        net = compile_program(program, max_steps=4)
        result = self.assertMatches(program, net, {"i": 1, "a[1]": 2})
        self.assertEqual(result["a[2]"], 3)
        self.assertEqual(result["i"], 2)

    def test_array_branch_conditions_keep_strict_positive_semantics(self):
        program = parse("array a[2]\nif a[i] > 0 goto yes\nx = -1\nreturn\nyes:\nx = 1")
        net = compile_program(program, max_steps=4)
        for i in range(2):
            for value in [-1e300, -1e-12, 0.0, 1e-4, 1e300]:
                self.assertMatches(program, net, {"i": i, f"a[{i}]": value})
            with self.assertRaisesRegex(NetworkContractError, "margin"):
                net.step({"i": i, f"a[{i}]": 1e-8})

    def test_recurrence(self):
        program = parse((ROOT / "examples/arrays.loop").read_text())
        net = compile_program(program, max_steps=7)
        result = self.assertMatches(program, net, ticks=17)
        self.assertEqual([result[f"bins[{i}]"] for i in range(4)], [5, 4, 4, 4])
        self.assertEqual(result["i"], 1)

    def test_backward_loop_fills_array(self):
        program = parse("""
            array a[4]
            i = 0
            again:
            a[i] = i * 2
            i = i + 1
            remaining = i + -3
            if remaining goto end
            goto again
            end:
        """)
        result = self.assertMatches(program, compile_program(program, max_steps=21))
        self.assertEqual([result[f"a[{i}]"] for i in range(4)], [0, 2, 4, 6])

    def test_invalid_dynamic_index_in_every_operand_position(self):
        for instruction in ["a[i] = 3", "a[i] = x", "x = a[i]", "x = a[i] + 2",
                            "x = y + a[i]", "x = a[i] * -2", "x = y * a[i]",
                            "if a[i] goto end\nend:"]:
            program = parse("array a[3]\n" + instruction)
            net = compile_program(program, max_steps=2)
            for i in [-1e300, -1, -math.ulp(0.0), math.ulp(0.0), 0.5,
                      math.nextafter(1.0, 0.0), math.nextafter(1.0, math.inf),
                      math.nextafter(2.0, math.inf), 3, 1e300]:
                initial = {"i": i}
                with self.subTest(instruction=instruction, index=i):
                    with self.assertRaises(ArrayIndexError):
                        program.step(initial)
                    with self.assertRaisesRegex(NetworkContractError, "array index"):
                        net.step(initial)

    def test_inactive_invalid_indices_are_ignored(self):
        program = parse("""
            array a[3]
            goto end
            a[i] = a[j] * a[i]
            x = a[i] * 1e300
            if a[i] goto end
            end:
        """)
        net = compile_program(program, max_steps=2)
        self.assertMatches(program, net, {"i": -1e300, "j": 0.5, "a[0]": 1e300}, ticks=2)

    def test_single_element_array_and_large_overwrite(self):
        program = parse("array a[1]\na[i] = 3")
        net = compile_program(program, max_steps=2)
        self.assertMatches(program, net, {"a[0]": 1e300})
        for i in [-math.ulp(0.0), math.ulp(0.0), 0.5, 1]:
            with self.assertRaises(NetworkContractError):
                net.step({"i": i})

    def test_compile_time_index_and_declaration_errors(self):
        sources = ["array a[0]", "array a[-1]", "array a[1.5]", "array a[n]",
                   "array a[2]\narray a[3]", "x = a[0]", "a[0] = 2",
                   "array a[2]\na = 1", "array a[2]\nx = a + 1",
                   "array a[2]\nx = a[-1]", "array a[2]\nx = a[2]",
                   "array a[2]\nx = a[0.5]", "array a[2]\nx = a[1e999]",
                   "array a[2]\nx = a[i+1]", "array a[2]\nx = a[a[0]]",
                   "array a[2]\nx = a[a]", "a = 1\narray a[2]"]
        for source in sources:
            with self.subTest(source=source), self.assertRaises(LanguageError):
                parse(source)

    def test_export_retains_array_layout_and_diagnostics(self):
        net = compile_program("array a[3]\na[i] = a[j] * 2", max_steps=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arrays.json"
            net.save(path)
            loaded = Network.load(path)
            self.assertEqual(loaded.summary(), net.summary())
            self.assertEqual(loaded.step({"i": 1, "j": 2, "a[2]": -3})["a[1]"], -6)
            with self.assertRaisesRegex(NetworkContractError, "array index"):
                loaded.step({"i": 0.5})

    def test_old_scalar_network_format_still_loads(self):
        data = {"format": "loopnet.reglu.v1", "width": 3, "variables": ["x"],
                "done_channel": 1, "margin_channel": 2, "branch_margin": 1e-4,
                "max_steps": 1, "blocks": [{"units": [], "bias": [[0, 1], [1, 1]]}],
                "schedule": [0]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps(data))
            net = Network.load(path)
            self.assertEqual(net.step({"x": 2}), {"x": 3})

    def test_cli_array_cells_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arrays.json"
            commands = [
                ["compile", "examples/arrays.loop", "-o", str(path), "--max-steps", "7"],
                ["run-net", str(path), "--ticks", "2", "--set", "bins[0]=5"],
                ["run", "examples/arrays.loop", "--ticks", "2", "--set", "bins[0]=5", "--compare"],
            ]
            for command in commands:
                result = subprocess.run([sys.executable, "-m", "loopnet", *command], cwd=ROOT,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                if command[0] != "compile":
                    state = json.loads(result.stdout.splitlines()[-1])["globals"]
                    self.assertEqual(state["bins[0]"], 6)
                    self.assertEqual(state["bins[1]"], 1)


if __name__ == "__main__":
    unittest.main()
