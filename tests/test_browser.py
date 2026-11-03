"""Cross-language checks for the compiler shipped to static-site visitors."""

import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from loopnet.browser import browser_program, manifest_bytes
from loopnet.compiler import compile_program
from loopnet.language import parse
from loopnet.structured import lower


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node is needed for the browser compiler")
class BrowserCompilerTests(unittest.TestCase):
    def compare(self, program, *, max_steps=1, initial=None):
        manifest, code = browser_program(program, max_steps=max_steps)
        expected = compile_program(program, max_steps=max_steps, execution="block",
                                   indirect_targets=program.address_targets, optimize_temporaries=True)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "program.json").write_bytes(manifest_bytes(manifest))
            (directory / "program.bin").write_bytes(gzip.decompress(code))
            expected.save(directory / "network.json")
            result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", """
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {compileProgram} from './loopnet/web/compiler.mjs';
import {ResidualNetwork} from './loopnet/web/neural.mjs';
const input=JSON.parse(fs.readFileSync(0,'utf8')), dir=input.directory;
const manifest=JSON.parse(fs.readFileSync(`${dir}/program.json`,'utf8'));
const bytes=fs.readFileSync(`${dir}/program.bin`);
const code=new Uint32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
const local=new ResidualNetwork(compileProgram(manifest,code));
const reference=new ResidualNetwork(JSON.parse(fs.readFileSync(`${dir}/network.json`,'utf8')));
const clean=block=>block.kind==='transient'?{...block,layers:block.layers.map(clean),clear:clean(block.clear),write:clean(block.write)}:
 Object.fromEntries(Object.entries(block).filter(([key])=>key!=='packed'));
// Every float64 weight, connection, bias, layer boundary, and ordering must match.
assert.deepEqual(local.blocks.map(clean),reference.blocks);
for(const key of Object.keys(reference.model))if(key!=='blocks')assert.deepEqual(local.model[key],reference.model[key],key);
const a=local.initialState(input.initial),b=reference.initialState(input.initial);
for(let tick=0;tick<1000;tick++) {
 local.forward(a);reference.forward(b);assert.deepEqual(a,b);
 local.assertValid(a);
 if(a[local.model.terminate_channel]===1)break;
}
if(manifest.continuation)assert.equal(a[local.model.terminate_channel],1);
"""], cwd=ROOT, input=json.dumps({"directory": str(directory), "initial": initial or {}}),
                                    text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_arithmetic_and_conditional_jumps(self):
        self.compare(parse("x = x * y\nz = x + -2\nz = relu(z)\nif z goto yes\nx = -4\nreturn\nyes:\nx = z * -3\nreturn"),
                     max_steps=8, initial={"x": 2, "y": 3})

    def test_indirect_arrays_and_bounds_diagnostics(self):
        self.compare(parse((ROOT / "examples/arrays.loop").read_text()), max_steps=8)

    def test_functions_while_and_dynamic_returns(self):
        self.compare(parse(lower((ROOT / "examples/functions.flow").read_text(), max_stack_depth=3)))

    def test_recursion_and_transient_call_banks(self):
        self.compare(parse(lower((ROOT / "examples/recursive.flow").read_text(), max_stack_depth=6)))

    def test_unrolling_and_break(self):
        self.compare(parse(lower("""array squares[8]
for i in range(8):
    squares[i] = i * i
total = 0
while 1:
    total += squares[3]
    if total > 10:
        break
""")))


if __name__ == "__main__":
    unittest.main()
