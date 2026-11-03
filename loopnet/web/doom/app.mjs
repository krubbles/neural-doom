import {AtlasViewer} from './atlas-viewer.mjs';

const $ = selector => document.querySelector(selector);
const canvas = $('#game'), context = canvas.getContext('2d', {alpha: false});
const worker = new Worker(new URL('./game-worker.mjs', import.meta.url), {type: 'module'});
const held = new Set(), pressed = new Set(), pending = new Set();
const description = $('#description');
new ResizeObserver(() => document.body.style.setProperty('--description-height', `${description.getBoundingClientRect().height}px`)).observe(description);
let config, image, ready = false, failed = false, twitterMode = false, serial = 0, sentAt = 0, timer;
const neurons = new AtlasViewer($('#neurons'),
  request => worker.postMessage({type: 'atlas-value', ...request}),
  buffer => worker.postMessage({type: 'atlas-buffer', buffer}, [buffer]));

function post(message) {
  const id = ++serial; pending.add(id);
  worker.postMessage({...message, id});
}

function clearInputs() {
  held.clear(); pressed.clear();
  for (const button of document.querySelectorAll('.held')) button.classList.remove('held');
}

function schedule() {
  clearTimeout(timer);
  if (ready && !failed && !pending.size && !document.hidden) {
    timer = setTimeout(tick, Math.max(0, 50 - (performance.now() - sentAt)));
  }
}

function tick() {
  if (!ready || failed || pending.size || document.hidden) return;
  const has = (...codes) => Number(codes.some(code => held.has(code) || pressed.has(code)));
  sentAt = performance.now();
  post({type: 'tick', input: {
    forward: has('KeyW', 'ArrowUp') - has('KeyS', 'ArrowDown'),
    strafe: has('KeyD') - has('KeyA'),
    turn: has('KeyE', 'ArrowRight') - has('KeyQ', 'ArrowLeft'),
    fire: has('Space', 'MouseFire')
  }});
  pressed.clear();
}

function reset() {
  if (!ready || failed) return;
  clearInputs(); clearTimeout(timer);
  post({type: 'reset'});
  canvas.focus({preventScroll: true});
}

function setTwitterMode(enabled) {
  if (!ready || failed || twitterMode === enabled) return;
  clearInputs(); clearTimeout(timer);
  twitterMode = enabled;
  document.body.classList.toggle('twitter-mode', enabled);
  $('#twitter-mode').setAttribute('aria-pressed', String(enabled));
  neurons.setVertical(enabled);
  post({type: 'atlas-limit', neuronLimit: enabled ? 100 : 60, layerStride: enabled ? 3 : 1});
  canvas.focus({preventScroll: true});
}

function draw(data) {
  for (let index = 0; index < data.frame.length; index++) {
    const value = Math.max(0, Math.min(config.palette.length - 1, data.frame[index]));
    const low = Math.floor(value), blend = value - low;
    const a = config.palette[low], b = config.palette[Math.min(low + 1, config.palette.length - 1)], at = index * 4;
    for (let channel = 0; channel < 3; channel++) image.data[at + channel] = a[channel] + (b[channel] - a[channel]) * blend;
    image.data[at + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  canvas.dataset.frame = data.frameNumber;
  neurons.draw(data.atlas, data.frameNumber);
}

function error(message) {
  failed = true; clearTimeout(timer); clearInputs();
  canvas.setAttribute('aria-busy', 'false');
  $('#loading').hidden = true;
  $('#error').hidden = false; $('#error').textContent = message;
  for (const button of document.querySelectorAll('[data-code], #reset, #twitter-mode')) button.disabled = true;
}

worker.onmessage = ({data}) => {
  if (data.type === 'atlas-value') {neurons.inspected(data); return;}
  pending.delete(data.id);
  if (data.type === 'error') {error(data.message); return;}
  if (data.type === 'atlas-layout') {
    neurons.ready(data.atlasLayout); neurons.draw(data.atlas, data.frameNumber);
    schedule(); return;
  }
  if (data.type === 'ready') {
    config = data.config; canvas.width = config.width; canvas.height = config.height;
    image = context.createImageData(config.width, config.height);
    neurons.ready(data.atlasLayout);
    $('#parameter-count').textContent = data.statistics.parameters.toLocaleString('en-US');
    $('#layer-count').textContent = data.statistics.layers.toLocaleString('en-US');
    ready = true; canvas.setAttribute('aria-busy', 'false');
    $('#loading').hidden = true;
    for (const button of document.querySelectorAll('[data-code], #reset, #twitter-mode')) button.disabled = false;
  }
  draw(data); schedule();
};
worker.onerror = event => error(event.message || 'The neural worker could not start.');

$('#reset').addEventListener('click', reset);
$('#twitter-mode').addEventListener('click', () => setTwitterMode(true));

canvas.addEventListener('pointerdown', event => {
  if (!ready || event.button !== 0) return;
  canvas.focus({preventScroll: true});
  held.add('MouseFire'); pressed.add('MouseFire'); canvas.setPointerCapture(event.pointerId);
});
for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(event, () => held.delete('MouseFire'));
const keys = new Set(['KeyW', 'KeyS', 'KeyA', 'KeyD', 'KeyQ', 'KeyE', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space']);
window.addEventListener('keydown', event => {
  if (event.code === 'Escape' && twitterMode) {event.preventDefault(); setTwitterMode(false); return;}
  if (!ready || event.ctrlKey || event.metaKey || event.altKey || event.target.closest?.('input, select, textarea')) return;
  if (event.code === 'KeyR' && !event.repeat) {event.preventDefault(); reset(); return;}
  if (!keys.has(event.code) || (event.code === 'Space' && event.target.closest?.('button, a'))) return;
  event.preventDefault(); held.add(event.code); pressed.add(event.code);
});
window.addEventListener('keyup', event => held.delete(event.code));
window.addEventListener('blur', clearInputs);
document.addEventListener('visibilitychange', () => {clearInputs(); schedule();});
for (const button of document.querySelectorAll('[data-code]')) {
  button.addEventListener('pointerdown', event => {
    if (!ready || event.button !== 0) return;
    event.preventDefault();
    held.add(button.dataset.code); pressed.add(button.dataset.code); button.classList.add('held');
    button.setPointerCapture(event.pointerId);
  });
  for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) button.addEventListener(event, () => {
    held.delete(button.dataset.code); button.classList.remove('held');
  });
  // Keyboard and assistive-technology activation still produce a single input.
  button.addEventListener('click', event => {if (ready && event.detail === 0) pressed.add(button.dataset.code);});
}
post({type: 'init'});
