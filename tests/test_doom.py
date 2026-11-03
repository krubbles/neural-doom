import gzip
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from threading import Thread
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen

from loopnet.doom import (Handler, HEIGHT, INITIAL, PALETTE, SOURCE, VIEW_HEIGHT, WIDTH,
                          browser_assets, compiled_game, export_site, model_bytes)


ROOT = Path(__file__).resolve().parents[1]


class NeuralShooterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = compiled_game()

    def test_architecture_and_pixels_match_independent_trusted_source_execution(self):
        # Execute only our checked-in test fixture as an independent oracle.
        source = re.sub(r"^array \w+\[\d+\]\n", "", SOURCE.read_text(), flags=re.MULTILINE)
        source = re.sub(r"^inline def ", "def ", source, flags=re.MULTILINE)
        expected = {"__builtins__": {"range": range}, "relu": lambda x: max(0.0, x),
                    "frame": [0.0] * (WIDTH * HEIGHT), "alive": [1.0] * 3,
                    **{k: v for k, v in INITIAL.items() if "[" not in k},
                    "forward": 0, "strafe": 0, "turn": 0, "fire": 0}
        exec(compile(source, "<trusted shooter fixture>", "exec"), expected)
        state = self.net(self.net.initial_state(INITIAL))
        self.net.assert_valid(state)
        self.assertTrue(self.net.terminated(state))
        actual = self.net.globals(state)
        for i, value in enumerate(expected["frame"]):
            self.assertAlmostEqual(actual[f"frame[{i}]"], value, places=8)
        self.assertEqual(actual["ammo"], 60)
        self.assertEqual(actual["kills"], 0)
        self.assertLess(self.net.width, WIDTH * HEIGHT + 100)
        self.assertEqual(len(self.net.regions), 2)

    def test_close_wall_contour_matches_perspective_within_one_pixel(self):
        # These rays all meet the right room wall, away from the two pillars.
        # Derive its silhouette analytically, independently of the ray marcher.
        direction_x, direction_y = 0.5, -math.sqrt(0.75)
        state = self.net(self.net.initial_state({**INITIAL, "player_x": 6.1,
                         "dir_x": direction_x, "dir_y": direction_y,
                         "alive[0]": 0, "alive[1]": 0, "alive[2]": 0}))
        self.net.assert_valid(state)
        pixels = self.net.globals(state)
        tops = []
        for col in range(40, WIDTH):
            ray_x = direction_x - direction_y * ((2 * col - 63) * 0.0103125)
            distance = (7 - 6.1) / ray_x
            expected_top = math.ceil((VIEW_HEIGHT - 1) / 2 - 16 / distance)
            actual_top = next(row for row in range(VIEW_HEIGHT // 2)
                              if 3 < pixels[f"frame[{row * WIDTH + col}]"] < 12)
            self.assertLessEqual(abs(actual_top - expected_top), 1, f"column {col}")
            tops.append(actual_top)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(tops, tops[1:])), 1)

    @unittest.skipUnless(shutil.which("node"), "Node is needed for the browser game checks")
    def test_neural_controls_collision_occlusion_cooldown_and_all_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "doom.json"
            path.write_bytes(model_bytes())
            manifest, program = browser_assets()
            (Path(directory) / "program.json").write_bytes(manifest)
            (Path(directory) / "program.bin").write_bytes(gzip.decompress(program))
            script = """
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {compileProgram} from './loopnet/web/compiler.mjs';
import {ResidualNetwork} from './loopnet/web/neural.mjs';
import {captureActivations} from './loopnet/web/activation-trace.mjs';
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const manifest=JSON.parse(fs.readFileSync(input.manifest,'utf8')), bytes=fs.readFileSync(input.program);
const code=new Uint32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
const n=new ResidualNetwork(compileProgram(manifest,code));
const reference=new ResidualNetwork(JSON.parse(fs.readFileSync(input.model,'utf8')));
const clean=b=>b.kind==='transient'?{...b,layers:b.layers.map(clean),clear:clean(b.clear),write:clean(b.write)}:
 Object.fromEntries(Object.entries(b).filter(([key])=>key!=='packed'));
assert.deepEqual(n.blocks.map(clean),reference.blocks); // every locally generated Doom weight
const before=n.initialState(input.initial), expected=before.slice();
n.run(expected);
const trace=captureActivations(n,before);
assert.equal(trace.calls,1);
assert.equal(trace.stages.length,input.depth+1);
assert.deepEqual(trace.residuals.at(-1),expected);
assert.equal(before[n.model.terminate_channel],0); // capture does not mutate the frame inputs
// Independent bitmaps check the actual framebuffer, including unlit cells.
const digits=['111101101101111','001001001001001','111001111100111',
 '111001111001111','101101111001001','111100111001111',
 '111100111101111','111001001001001','111101111101111','111101111001111'];
function hud(s) {
 const ammo=n.read(s,'ammo'), enemies=3-n.read(s,'kills'), won=n.read(s,'cleared')===1;
 const bar=Array(8*64).fill(21);bar.fill(22,0,64);
 function glyph(col,bitmap,ink,width=3) {
  for(let i=0;i<bitmap.length;i++)if(bitmap[i]==='1')bar[(2+Math.floor(i/width))*64+col+i%width]=ink;
 }
 if(won) {
  glyph(23,['10001','10001','10101','10101','01010'].join(''),25,5);
  glyph(30,'111010010010111',25);
  glyph(35,['10001','11001','10101','10011','10001'].join(''),25,5);
 } else {
  glyph(2,'111101111101101',19); // A
  glyph(6,'101111101101101',19); // M
  glyph(10,'101111101101101',19); // M
  glyph(14,'111101101101111',19); // O
  glyph(22,digits[Math.floor(ammo/10)],23);
  glyph(26,digits[ammo%10],23);
  glyph(34,'111100111100111',24); // E
  glyph(38,'101111111111101',24); // N
  glyph(42,'111100111100111',24); // E
  glyph(46,'101111101101101',24); // M
  glyph(50,'101101010010010',24); // Y
  glyph(59,digits[enemies],24);
 }
 // Checking every bar pixel also proves victory clears all previous text.
 for(let i=0;i<bar.length;i++)assert.ok(Math.abs(n.read(s,`frame[${40*64+i}]`)-bar[i])<1e-8,
  `HUD pixel ${i}, ammo ${ammo}, enemies ${enemies}`);
}
for(const [ammo,enemies] of [[0,0],[9,1],[47,2],[60,3]]) {
 const s=n.initialState({...input.initial,ammo,'alive[0]':Number(enemies>0),
  'alive[1]':Number(enemies>1),'alive[2]':Number(enemies>2)});
 n.run(s); hud(s);
}
let memory={...input.initial};
function tick(keys={}) {
 const s=n.initialState({...memory,...keys});
 assert.equal(n.run(s),1);
 memory=Object.fromEntries(Object.keys(input.initial).map(k=>[k,n.read(s,k)]));
 for(let i=0;i<input.pixels;i++)assert.ok(n.read(s,`frame[${i}]`)>=0&&n.read(s,`frame[${i}]`)<=input.paletteSize-1+1e-6);
 hud(s);
 return Object.fromEntries(['ammo','kills','cleared','shot','player_x','player_y'].map(k=>[k,n.read(s,k)]));
}
assert.equal(tick().kills,0);
assert.equal(tick({fire:1}).kills,1);
assert.equal(tick({fire:1}).ammo,59); // cooldown suppresses the second shot
for(let i=0;i<14;i++)tick({turn:1});
assert.equal(tick({fire:1}).kills,2);
for(let i=0;i<21;i++)tick({turn:-1});
assert.equal(tick({fire:1}).kills,2); // the left pillar occludes target three
for(let i=0;i<7;i++)tick({turn:1});
for(let i=0;i<18;i++)tick({strafe:-1});
assert.equal(tick({fire:1}).kills,3);
assert.equal(tick().cleared,1);
for(let i=0;i<80;i++)tick({strafe:-1});
assert.ok(memory.player_x>=1.16&&memory.player_x<1.4);
for(let i=0;i<80;i++)tick({forward:1});
assert.ok(memory.player_y>=1.16&&memory.player_y<1.4);
memory={...input.initial,player_x:2.8};
for(let i=0;i<80;i++)tick({forward:1});
assert.ok(memory.player_y>=4.26&&memory.player_y<4.5); // internal pillar collision
console.log('All targets cleared; occlusion, cooldown and both collision types verified.');
"""
            result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script], cwd=ROOT,
                                    input=json.dumps({"model": str(path), "initial": INITIAL, "pixels": WIDTH * HEIGHT,
                                                      "manifest": str(Path(directory) / "program.json"),
                                                      "program": str(Path(directory) / "program.bin"),
                                                      "depth": self.net.summary()["feedforward_layers"], "paletteSize": len(PALETTE)}),
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_game_assets_and_config(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            for path in ("/", "/app.mjs", "/style.css", "/game-worker.mjs", "/neural.mjs",
                         "/compiler.mjs", "/program.json", "/program.bin.gz", "/doom.flow.txt",
                         "/activation-trace.mjs", "/activation-atlas.mjs", "/atlas-viewer.mjs",
                         "/activation-viewer.mjs", "/source", "/health"):
                with urlopen(base + path) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.read())
            with urlopen(base + "/api/config") as response:
                config = json.load(response)
                self.assertEqual((config["width"], config["height"]), (64, 48))
                self.assertEqual(config["summary"]["state_width"], self.net.width)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)

    @unittest.skipUnless(shutil.which("node"), "Node is needed for static-site checks")
    def test_static_export_boots_from_a_nested_path_without_api_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / "nested-site"
            size = export_site(site)
            self.assertLess(size, len(model_bytes()) // 5)
            self.assertTrue((site / ".nojekyll").exists())
            self.assertEqual((site / "doom.flow.txt").read_text(), SOURCE.read_text())
            manifest = json.loads((site / "program.json").read_text())
            self.assertNotIn("blocks", manifest)
            requested = []

            class StaticHandler(SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=directory, **kwargs)

                def do_GET(self):
                    requested.append(self.path)
                    super().do_GET()

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), StaticHandler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/nested-site/program.json"
                result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", """
import assert from 'node:assert/strict';
import {loadProgram} from './loopnet/web/compiler.mjs';
import {ResidualNetwork} from './loopnet/web/neural.mjs';
const {model,config}=await loadProgram(process.argv[1]);
const n=new ResidualNetwork(model),state=n.initialState(config.initial);
assert.equal(n.run(state),1);
assert.equal(n.read(state,'ammo'),60);assert.equal(n.read(state,'kills'),0);
assert.equal(n.model.width,3144);
assert.equal(n.read(state,'frame[2710]'),23); // the neural ammo counter
""", url], cwd=ROOT, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(requested, ["/nested-site/program.json", "/nested-site/program.bin.gz"])
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
