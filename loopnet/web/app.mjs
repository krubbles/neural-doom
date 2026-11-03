const $ = selector => document.querySelector(selector);
const worker = new Worker('/game-worker.mjs', { type: 'module' });
const cells = [], buttons = [], previews = [];
let game = null, busy = true, selected = 3, requestId = 0;

for (let displayRow = 5; displayRow >= 0; displayRow--) {
  for (let column = 0; column < 7; column++) {
    const index = column * 6 + displayRow;
    const cell = document.createElement('div');
    cell.className = 'slot'; cell.dataset.player = '0'; cell.dataset.index = index;
    cell.style.setProperty('--fall', `${-(displayRow + 1) * 100}%`);
    const disc = document.createElement('span'); disc.className = 'disc'; cell.append(disc);
    $('#cells').append(cell); cells[index] = cell;
  }
}

for (let column = 0; column < 7; column++) {
  const preview = document.createElement('span'); preview.className = 'preview';
  $('#previews').append(preview); previews.push(preview);
  const button = document.createElement('button');
  button.className = 'column-button'; button.dataset.column = column;
  button.disabled = true; button.setAttribute('aria-label', `Drop disc in column ${column + 1}`);
  button.addEventListener('click', () => move(column));
  button.addEventListener('mouseenter', () => { selected = column; renderPreview(); });
  button.addEventListener('focus', () => { selected = column; renderPreview(); });
  button.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === 'ArrowRight' ? 1 : -1;
    for (let offset = 1; offset < 8; offset++) {
      const next = (column + offset * direction + 14) % 7;
      if (!buttons[next].disabled) { buttons[next].focus(); break; }
    }
  });
  $('#columns').append(button); buttons.push(button);
}

function renderPreview() {
  previews.forEach((preview, index) => preview.classList.toggle('active',
    index === selected && !busy && game && !game.finished && game.heights[index] < 6));
}

function render(animate = false) {
  document.body.classList.toggle('busy', busy);
  if (!game) return;
  document.body.dataset.player = game.winner || game.player;
  const last = game.lastColumn * 6 + game.lastRow;
  cells.forEach((cell, index) => {
    cell.dataset.player = game.board[index];
    cell.classList.toggle('winning', game.winning[index] > 0);
    cell.classList.toggle('last', index === last && game.moves > 0);
    cell.classList.remove('drop');
    if (animate && index === last && !game.winning[index]) {
      void cell.offsetWidth;
      cell.classList.add('drop');
    }
  });
  buttons.forEach((button, index) => {
    button.disabled = busy || game.finished > 0 || game.heights[index] >= 6;
    button.setAttribute('aria-label', `Drop disc in column ${index + 1}${game.heights[index] >= 6 ? ', full' : `, ${6 - game.heights[index]} spaces left`}`);
  });
  $('#board').classList.toggle('over', game.winner > 0);
  $('.control-panel').classList.toggle('celebrating', game.finished > 0);
  for (let player = 1; player <= 2; player++) {
    const active = game.winner === player || (!game.finished && game.player === player);
    $(`#player-${player}`).classList.toggle('active', active);
    $(`#player-${player}`).setAttribute('aria-label', `Player ${player}${active ? game.finished ? ', winner' : ', current turn' : ''}`);
  }
  if (game.winner) {
    $('#status-label').textContent = 'A CONNECTION WORTH CELEBRATING';
    $('#headline').textContent = `Player ${game.winner} wins!`;
    $('#detail').textContent = 'Four connected. Well played. Ready for a rematch?';
  } else if (game.finished) {
    $('#status-label').textContent = 'AN EVEN MATCH';
    $('#headline').textContent = 'It’s a draw.';
    $('#detail').textContent = 'A full board, and two worthy opponents. Go again?';
  } else {
    $('#status-label').textContent = busy ? 'ONE GOOD MOVE…' : 'THE NEXT MOVE';
    $('#headline').textContent = `Player ${game.player}’s turn`;
    $('#detail').textContent = game.moves ? 'Keep an eye on those diagonals. Choose your next column.' : 'Coral goes first. Pick a column to get things rolling.';
  }
  $('#move-count').textContent = game.moves;
  $('#progress').style.width = `${game.moves / 42 * 100}%`;
  $('#reset').disabled = busy;
  $('#undo').disabled = busy || !game.canUndo;
  renderPreview();
}

function send(type, extra = {}) {
  if (busy && type !== 'init') return;
  busy = true;
  $('#feedback').textContent = '';
  render();
  worker.postMessage({ type, id: ++requestId, ...extra });
}

function move(column) {
  if (busy || !game || game.finished || game.heights[column] >= 6) return;
  selected = column;
  send('move', { column });
}

worker.onmessage = event => {
  const response = event.data;
  if (response.id !== requestId) return;
  busy = false;
  if (response.type === 'error') {
    $('#feedback').textContent = response.message;
    if (!game) { $('#headline').textContent = 'Couldn’t set up the game.'; $('#detail').textContent = 'Refresh the page to try again.'; }
  } else {
    const oldMoves = game?.moves || 0;
    game = response.game;
    render(game.moves > oldMoves);
  }
  renderPreview();
  if (response.type === 'error') render();
};
worker.onerror = () => {
  busy = false;
  $('#feedback').textContent = 'The game could not start. Refresh the page to try again.';
  $('#headline').textContent = 'Something went wrong.';
  document.body.classList.remove('busy');
};

$('#reset').addEventListener('click', () => send('reset'));
$('#undo').addEventListener('click', () => send('undo'));
$('#help').addEventListener('click', () => $('#rules').showModal());
$('#close-help').addEventListener('click', () => $('#rules').close());
$('#lets-play').addEventListener('click', () => $('#rules').close());
$('#rules').addEventListener('click', event => {
  if (event.target !== $('#rules')) return;
  const rect = $('#rules').getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $('#rules').close();
});
document.addEventListener('keydown', event => {
  if ($('#rules').open || event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
  if (/^[1-7]$/.test(event.key)) { event.preventDefault(); move(Number(event.key) - 1); }
  if (event.key.toLowerCase() === 'u' && !busy && game?.canUndo) { event.preventDefault(); send('undo'); }
});

send('init');
