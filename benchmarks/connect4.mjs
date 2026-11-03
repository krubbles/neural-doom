/** Benchmark one or more exported models on the same seven-move game.
 * python3 -m loopnet compile examples/connect4.flow -o /tmp/connect4.json
 * node benchmarks/connect4.mjs /tmp/connect4.json [older-model.json ...]
 */
import fs from 'node:fs';
import path from 'node:path';
import {performance} from 'node:perf_hooks';
import {ResidualNetwork} from '../loopnet/web/neural.mjs';

const files = process.argv.slice(2);
if (!files.length) {
  console.error('Usage: node benchmarks/connect4.mjs model.json [model.json ...]');
  process.exit(1);
}
const models = files.map(file => {
  const net = new ResidualNetwork(JSON.parse(fs.readFileSync(file, 'utf8')));
  const persistent = net.model.variables.filter(name =>
    /^(board\[|heights\[|winning\[|player$|moves$|finished$|winner$|last_column$|last_row$)/.test(name));
  return {file, net, persistent, samples: []};
});

function game({net, persistent}) {
  let state = net.initialState({player: 1});
  let calls = 0;
  const start = performance.now();
  for (const column of [3, 2, 3, 2, 3, 2, 3]) {
    const values = Object.fromEntries(persistent.map(name => [name, net.read(state, name)]));
    state = net.initialState({...values, column});
    calls += net.run(state);
  }
  if (net.read(state, 'winner') !== 1 || net.read(state, 'moves') !== 7) throw new Error('Incorrect game result');
  return {milliseconds: performance.now() - start, calls};
}

for (const model of models) game(model); // warm-up
for (let round = 0; round < 5; round++) {
  for (let i = 0; i < models.length; i++) {
    const model = models[(round + i) % models.length];
    model.samples.push(game(model));
  }
}
for (const {file, net, samples} of models) {
  samples.sort((a, b) => a.milliseconds - b.milliseconds);
  const median = samples[2];
  const depth = net.model.schedule.reduce((sum, id) => {
    const block = net.model.blocks[id];
    return sum + (block.kind === 'transient' ? block.layers.length + 2 : 1);
  }, 0);
  console.log(JSON.stringify({model: path.basename(file), state_width: net.model.width,
    feedforward_layers: depth, calls: median.calls,
    median_ms_per_seven_moves: Math.round(median.milliseconds * 10) / 10}));
}
