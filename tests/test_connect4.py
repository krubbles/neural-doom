import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from loopnet import Interpreter, parse
from loopnet.connect4 import Handler, SOURCE, compiled_game, model_bytes
from loopnet.structured import lower
from http.server import ThreadingHTTPServer


ROOT = Path(__file__).resolve().parents[1]
DRAW = [3, 6, 6, 3, 0, 1, 3, 0, 0, 5, 1, 1, 1, 2, 0, 6, 0, 3, 2, 2, 3,
        4, 2, 3, 2, 4, 4, 0, 2, 1, 6, 5, 6, 4, 5, 6, 4, 5, 4, 5, 1, 5]


def oracle(moves):
    """Independent whole-board rules oracle; never used by the shipped GUI."""
    board, heights = [0] * 42, [0] * 7
    player, count, winner, finished = 1, 0, 0, 0
    results = []
    for column in moves:
        error = 3 if finished else 1 if not 0 <= column < 7 else 2 if heights[column] == 6 else 0
        if not error:
            board[column * 6 + heights[column]] = player
            heights[column] += 1
            count += 1
            for c in range(7):
                for r in range(6):
                    if not board[c * 6 + r]:
                        continue
                    for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                        cells = [(c + k * dc, r + k * dr) for k in range(4)]
                        if all(0 <= x < 7 and 0 <= y < 6 and board[x * 6 + y] == player for x, y in cells):
                            winner = player
            finished = int(winner > 0 or count == 42)
            if not finished:
                player = 3 - player
        results.append({"board": list(board), "heights": list(heights), "moves": count,
                        "player": player, "winner": winner, "finished": finished, "error": error})
    return results


class ConnectFourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program = parse(lower(SOURCE.read_text()))

    def test_source_rule_transaction_and_full_column(self):
        memory = {"player": 1.0}
        for expected, column in zip(oracle([0] * 7), [0] * 7):
            machine = Interpreter(self.program, {**memory, "column": column})
            result = machine.step(max_steps=5000)
            self.assertTrue(machine.terminated)
            self.assertEqual(result["error"], expected["error"])
            self.assertEqual(result["moves"], expected["moves"])
            for index, value in enumerate(expected["board"]):
                self.assertEqual(result[f"board[{index}]"], value)
            memory = result

    @unittest.skipUnless(shutil.which("node"), "Node is needed for the browser neural-runtime checks")
    def test_browser_weights_against_all_win_directions_draw_and_random_games(self):
        cases = [
            [0, 1, 0, 1, 0, 1, 0, 2],
            [0, 6, 1, 6, 2, 5, 3],
            [0, 1, 1, 2, 4, 2, 2, 3, 4, 3, 5, 3, 3],
            [6, 5, 5, 4, 2, 4, 4, 3, 2, 3, 1, 3, 3],
            [0, 1, 0, 1, 2, 1, 2, 1],
            [0] * 7 + [1],
            [-1, 7, 3],
            DRAW,
        ]
        rng = random.Random(782)
        for _ in range(16):
            moves, heights = [], [0] * 7
            for _ in range(42):
                legal = [c for c in range(7) if heights[c] < 6]
                if not legal:
                    break
                column = rng.choice(legal)
                moves.append(column)
                heights[column] += 1
                if oracle(moves)[-1]["finished"]:
                    break
            cases.append(moves)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "network.json"
            model_path.write_bytes(model_bytes())
            script = """
import fs from 'node:fs';
import { ResidualNetwork } from './loopnet/web/neural.mjs';
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const network = new ResidualNetwork(JSON.parse(fs.readFileSync(input.model, 'utf8')));
const persisted = network.model.variables.filter(name => /^(board\\[|heights\\[|winning\\[|player$|moves$|finished$|winner$|last_column$|last_row$)/.test(name));
const output = [];
for (const moves of input.cases) {
  let state = network.initialState({player: 1});
  const records = [];
  for (const column of moves) {
    const values = Object.fromEntries(persisted.map(name => [name, network.read(state, name)]));
    const next = network.initialState({...values, column});
    const calls = network.run(next);
    const read = name => network.read(next, name);
    records.push({board: Array.from({length:42}, (_,i)=>read(`board[${i}]`)),
      heights: Array.from({length:7}, (_,i)=>read(`heights[${i}]`)),
      winning: Array.from({length:42}, (_,i)=>read(`winning[${i}]`)),
      moves:read('moves'),player:read('player'),winner:read('winner'),finished:read('finished'),error:read('error'),calls});
    if (read('accepted')) state = next;
  }
  output.push(records);
}
console.log(JSON.stringify(output));
"""
            result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script], cwd=ROOT,
                                    input=json.dumps({"model": str(model_path), "cases": cases}),
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual_cases = json.loads(result.stdout)
        for case_index, (sequence, actuals) in enumerate(zip(cases, actual_cases)):
            for move_index, (expected, actual) in enumerate(zip(oracle(sequence), actuals)):
                with self.subTest(case=case_index, move=move_index):
                    for key, value in expected.items():
                        self.assertEqual(actual[key], value, key)
                    self.assertGreater(actual["calls"], 0)
                    marked = [i for i, value in enumerate(actual["winning"]) if value]
                    if actual["winner"]:
                        self.assertGreaterEqual(len(marked), 4)
                        self.assertTrue(all(actual["board"][i] == actual["winner"] for i in marked))
                    else:
                        self.assertEqual(marked, [])
        self.assertEqual(actual_cases[7][-1]["finished"], 1)
        self.assertEqual(actual_cases[7][-1]["winner"], 0)
        self.assertEqual(actual_cases[7][-1]["moves"], 42)

    def test_local_server_assets_source_and_network(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            for path in ("/", "/style.css", "/app.mjs", "/game-worker.mjs", "/neural.mjs", "/source", "/health"):
                with urlopen(base + path) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.read())
            with urlopen(base + "/api/network") as response:
                model = json.load(response)
                self.assertEqual(model["width"], compiled_game().width)
                self.assertIsInstance(model["terminate_channel"], int)
                self.assertNotIn("instructions", model)
            for path in ("/missing", "/../README.md"):
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + path)
                self.assertEqual(error.exception.code, 404)
                error.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
