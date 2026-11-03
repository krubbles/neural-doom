import { ResidualNetwork } from './neural.mjs';

let network, state;
let history = [];
const persisted = ['player', 'moves', 'finished', 'winner', 'last_column', 'last_row'];

function snapshot(metrics = {}) {
  const read = name => network.read(state, name);
  return {
    board: Array.from({ length: 42 }, (_, i) => read(`board[${i}]`)),
    heights: Array.from({ length: 7 }, (_, i) => read(`heights[${i}]`)),
    winning: Array.from({ length: 42 }, (_, i) => read(`winning[${i}]`)),
    player: read('player'), moves: read('moves'), finished: read('finished'), winner: read('winner'),
    lastColumn: read('last_column'), lastRow: read('last_row'), canUndo: history.length > 0,
    ...metrics,
  };
}

function freshGame() {
  history = [];
  state = network.initialState({ player: 1, last_column: -1, last_row: -1 });
  return snapshot();
}

function move(column) {
  if (!Number.isInteger(column) || column < 0 || column > 6) throw new Error('Choose a column from 1 to 7.');
  const before = state;
  const values = Object.fromEntries(persisted.map(name => [name, network.read(before, name)]));
  for (const prefix of ['board', 'heights', 'winning']) {
    const size = prefix === 'heights' ? 7 : 42;
    for (let i = 0; i < size; i++) values[`${prefix}[${i}]`] = network.read(before, `${prefix}[${i}]`);
  }
  values.column = column;
  const next = network.initialState(values);
  const start = performance.now();
  const calls = network.run(next);
  const milliseconds = performance.now() - start;
  const error = network.read(next, 'error');
  if (error > 0) {
    const messages = { 1: 'Choose a column from 1 to 7.', 2: 'That column is full. Try another.', 3: 'This game is over. Start a new one.' };
    throw new Error(messages[error] || 'The move was rejected.');
  }
  history.push(before);
  state = next;
  return snapshot({ milliseconds, calls });
}

self.onmessage = async event => {
  const { type, id, column } = event.data;
  try {
    if (type === 'init') {
      const response = await fetch('/api/network');
      if (!response.ok) throw new Error('Could not load the compiled game.');
      network = new ResidualNetwork(await response.json());
      self.postMessage({ id, type: 'ready', game: freshGame() });
    } else if (!network) {
      throw new Error('The game is still loading.');
    } else if (type === 'move') {
      self.postMessage({ id, type: 'state', game: move(column) });
    } else if (type === 'reset') {
      self.postMessage({ id, type: 'state', game: freshGame() });
    } else if (type === 'undo') {
      if (history.length) state = history.pop();
      self.postMessage({ id, type: 'state', game: snapshot() });
    }
  } catch (error) {
    self.postMessage({ id, type: 'error', message: error.message });
  }
};
