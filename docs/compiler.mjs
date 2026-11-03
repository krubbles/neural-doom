/** Source-to-weight compiler. Called once at startup, never during game frames. */

class Affine {
  constructor(terms = [], bias = 0) {this.terms = terms; this.bias = bias;}
  scaled(scale) {return new Affine(this.terms.map(([i, w]) => [i, w * scale]), this.bias * scale);}
  key() {return `${this.bias}:${this.terms.map(([i, w]) => `${i},${w}`).join(',')}`;}
}
const ONE = new Affine([], 1);
const channel = index => new Affine([[index, 1]]);

// Faithful summation for compile-time affine biases, as in the Python compiler.
function fsum(values) {
  const partials = [];
  for (let x of values) {
    let used = 0;
    for (let y of partials) {
      if (Math.abs(x) < Math.abs(y)) [x, y] = [y, x];
      const hi = x + y, lo = y - (hi - x);
      if (lo) partials[used++] = lo;
      x = hi;
    }
    partials.length = used; partials.push(x);
  }
  let hi = 0, lo = 0, n = partials.length;
  if (n) {
    hi = partials[--n];
    while (n) {
      const x = hi, y = partials[--n];
      hi = x + y; lo = y - (hi - x);
      if (lo) break;
    }
    if (n && ((lo < 0 && partials[n - 1] < 0) || (lo > 0 && partials[n - 1] > 0))) {
      const y = lo * 2, x = hi + y;
      if (y === x - hi) hi = x;
    }
  }
  return hi;
}

function sum(...expressions) {
  const terms = new Map();
  for (const expression of expressions) for (const [i, w] of expression.terms) terms.set(i, (terms.get(i) || 0) + w);
  return new Affine([...terms].filter(([, w]) => w !== 0).sort((a, b) => a[0] - b[0]), fsum(expressions.map(e => e.bias)));
}

// Generate typed sparse matrices directly, avoiding a second, object-heavy model.
function pack(units, remap = index => index) {
  let count = 0, gates = 0, values = 0, outputs = 0;
  for (const [g, v, o] of units()) {count++; gates += g.terms.length; values += v.terms.length; outputs += o.length;}
  const rows = length => ({pointers: new Int32Array(count + 1), indices: new Int32Array(length),
    weights: new Float64Array(length), biases: new Float64Array(count)});
  const gate = rows(gates), value = rows(values);
  const output = {pointers: new Int32Array(count + 1), indices: new Int32Array(outputs), weights: new Float64Array(outputs)};
  const targets = new Set();
  let index = 0, gi = 0, vi = 0, oi = 0;
  for (const [g, v, out] of units()) {
    gate.biases[index] = g.bias; value.biases[index] = v.bias;
    for (const [i, w] of g.terms) {gate.indices[gi] = remap(i); gate.weights[gi++] = w;}
    for (const [i, w] of v.terms) {value.indices[vi] = remap(i); value.weights[vi++] = w;}
    for (const [i, w] of out) {output.indices[oi] = i; output.weights[oi++] = w; targets.add(i);}
    index++; gate.pointers[index] = gi; value.pointers[index] = vi; output.pointers[index] = oi;
  }
  return {packed: true, count, gate, value, output, targets: [...targets], bias: []};
}

class Branch {
  constructor(width) {this.width = width; this.nodes = []; this.cache = new Map();}
  temporary(rows) {
    const key = rows.map(([g, v, w]) => `${g.key()}|${v.key()}|${w}`).join(';');
    let index = this.cache.get(key);
    if (index === undefined) {
      index = this.width + this.nodes.length;
      this.nodes.push(rows); this.cache.set(key, index);
    }
    return channel(index);
  }
  finish(clear, write) {
    this.cache.clear();
    const needed = new Uint8Array(this.nodes.length), pending = [];
    const inputs = (g, v) => {
      for (const expr of [g, v]) for (const [i] of expr.terms) if (i >= this.width && !needed[i - this.width]) {
        needed[i - this.width] = 1; pending.push(i - this.width);
      }
    };
    for (const block of [clear, write]) for (const [g, v] of block) inputs(g, v);
    while (pending.length) for (const [g, v] of this.nodes[pending.pop()]) inputs(g, v);
    const indices = new Int32Array(this.nodes.length), depths = new Int32Array(this.nodes.length), layers = [];
    let count = 0;
    for (let old = 0; old < needed.length; old++) if (needed[old]) {
      indices[old] = this.width + count++;
      let depth = 0;
      for (const [g, v] of this.nodes[old]) for (const expr of [g, v]) for (const [i] of expr.terms) {
        if (i >= this.width) depth = Math.max(depth, depths[i - this.width]);
      }
      (layers[depth] ||= []).push(old); depths[old] = depth + 1;
    }
    const remap = i => i < this.width ? i : indices[i - this.width], nodes = this.nodes;
    const packedLayers = layers.map(layer => pack(function* () {
      for (const old of layer) for (const [g, v, w] of nodes[old]) yield [g, v, [[indices[old], w]]];
    }, remap));
    return {kind: 'transient', transient_width: count, layers: packedLayers,
      clear: pack(() => clear.values(), remap), write: pack(() => write.values(), remap)};
  }
}

export function compileProgram(program, code) {
  if (program.format !== 'loopnet.program.v1') throw new Error('Unsupported source program format');
  if (!(code instanceof Uint32Array) || code.length !== program.word_count) throw new Error('Invalid source program data');
  const margin = program.branch_margin;
  if (!(margin > 0) || !Number.isFinite(1 / margin)) throw new Error('Invalid branch margin');
  const ramp = 2 ** Math.floor(Math.log2(margin)), inverseRamp = 1 / ramp;
  if (!Number.isFinite(inverseRamp)) throw new Error('Branch margin is too small for float64');
  const variables = program.globals.map(([name]) => name);
  const addresses = new Map(program.globals.map(([, id], i) => [id, i]));
  const [pc, done, marginFault, indexFault, terminate, jumpFault, stackFault] = Array.from({length: 7}, (_, i) => variables.length + i);
  const width = variables.length + 7, branch = new Branch(width), temp = rows => branch.temporary(rows);
  const write = [], clearMasks = new Map(), selectors = new Map(), enables = [];
  const entries = new Set(program.regions.map(([start]) => start));
  const transient = program.arrays.map(a => !addresses.has(a.cells[0]));
  const arrayKey = id => program.variable_count + program.accesses.length + id;
  const fault = (target, value, gate = ONE) => write.push([gate, value, [[target, 1]]]);
  const oneHot = (size, value, enabled, diagnostic) => {
    const key = `${size}|${value.key()}|${enabled.key()}|${diagnostic}`;
    if (selectors.has(key)) return selectors.get(key);
    const active = temp([[enabled, value, 1]]), above = [], below = [];
    for (let cell = 0; cell < size; cell++) above.push(temp([[sum(active, enabled.scaled(-cell)), ONE, 1]]));
    for (let cell = 0; cell < size; cell++) below.push(temp([[sum(enabled.scaled(cell), active.scaled(-1)), ONE, 1]]));
    const masks = above.map((a, i) => temp([[sum(enabled, a.scaled(-1), below[i].scaled(-1)), ONE, 1]]));
    fault(diagnostic, below[0]); fault(diagnostic, above.at(-1));
    for (let i = 0; i + 1 < size; i++) {
      const excess = temp([[sum(above[i], below[i + 1].scaled(-1)), ONE, 1]]);
      fault(diagnostic, temp([[enabled, sum(above[i], excess.scaled(-1)), 1]]));
    }
    selectors.set(key, masks); return masks;
  };
  const running = new Affine([[done, -1]], 1);
  // Sorting only retained globals is equivalent to sorting all dirty source names.
  const sortedGlobals = program.globals.slice().sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
  for (const [start, end, wordStart, wordEnd] of program.regions) {
    const above = temp([[new Affine([[pc, 1]], -start), ONE, 1]]);
    const below = temp([[new Affine([[pc, -1]], start), ONE, 1]]);
    const enabled = temp([[sum(ONE, above.scaled(-1), below.scaled(-1)), running, 1]]);
    enables.push(enabled);
    const versions = new Map(), dirty = new Set();
    const scalar = id => {
      if (!versions.has(id)) {
        if (!addresses.has(id)) throw new Error(`Source temporary ${id} was not defined on region entry`);
        versions.set(id, temp([[enabled, channel(addresses.get(id)), 1]]));
      }
      return versions.get(id);
    };
    const read = id => {
      if (id < program.variable_count) return scalar(id);
      const [array, index] = program.accesses[id - program.variable_count];
      if (transient[array]) return scalar(arrayKey(array));
      const cells = program.arrays[array].cells, masks = oneHot(cells.length, scalar(index), enabled, indexFault);
      return temp(masks.map((mask, cell) => [mask, scalar(cells[cell]), 1]));
    };
    const checkStack = () => {
      if (program.stack_pointer !== null) oneHot(program.max_stack_depth + 1, scalar(program.stack_pointer), enabled, stackFault);
    };
    checkStack();
    let at = wordStart, nextPc;
    while (at < wordEnd) {
      const op = code[at++];
      if (op === 9 || op === 10) {
        fault(done, ONE, enabled);
        if (op === 10 || program.continuation) fault(terminate, ONE, enabled);
        nextPc = enabled.scaled(end - 1);
      } else if (op === 6) nextPc = enabled.scaled(code[at++]);
      else if (op === 8) {
        const target = read(code[at++]), masks = oneHot(program.instruction_count, target, enabled, jumpFault);
        masks.forEach((mask, i) => {if (!entries.has(i)) fault(jumpFault, mask);});
        nextPc = target;
      } else if (op === 7) {
        const condition = read(code[at++]), target = code[at++];
        const positive = temp([[condition, enabled, 1]]);
        const low = temp([[sum(enabled.scaled(ramp), positive.scaled(-1)), ONE, 1]]);
        const taken = temp([[enabled, sum(enabled, low.scaled(-inverseRamp)), 1]]);
        nextPc = sum(enabled.scaled(end), taken.scaled(target - end));
        const gap = temp([[sum(enabled.scaled(margin), positive.scaled(-1)), ONE, 1]]);
        fault(marginFault, positive, gap.scaled(1 / margin));
      } else {
        const dest = code[at++], access = dest >= program.variable_count ? program.accesses[dest - program.variable_count] : null;
        const localArray = access && transient[access[0]];
        const cells = access ? program.arrays[access[0]].cells : null;
        const masks = access && !localArray ? oneHot(cells.length, scalar(access[1]), enabled, indexFault) : null;
        let value;
        if (op === 0) value = temp([[enabled, new Affine([], program.constants[code[at++]]), 1]]);
        else {
          const left = read(code[at++]);
          if (op === 11) value = left;
          else if (op === 5) value = temp([[left, enabled, 1]]);
          else {
            const operand = code[at++], immediate = op === 1 || op === 3;
            const right = immediate ? new Affine([], program.constants[operand]) : read(operand);
            if (op === 1 || op === 2) value = temp([[enabled, left, 1], [enabled, right, 1]]);
            else if (op === 3) value = temp([[enabled, left, program.constants[operand]]]);
            else if (op === 4) value = temp([[left, right, 1], [left.scaled(-1), right, -1]]);
            else throw new Error(`Unsupported source instruction ${op}`);
          }
        }
        if (localArray) versions.set(arrayKey(access[0]), value);
        else if (!masks) {versions.set(dest, value); dirty.add(dest);}
        else masks.forEach((mask, cell) => {
          const id = cells[cell], old = scalar(id), kept = temp([[ONE, old, 1], [mask, old, -1]]);
          versions.set(id, temp([[ONE, kept, 1], [mask, value, 1]])); dirty.add(id);
        });
        if (dest === program.stack_pointer) checkStack();
        nextPc = enabled.scaled(end);
      }
    }
    for (const [, id] of sortedGlobals) if (dirty.has(id)) {
      const target = addresses.get(id);
      if (!clearMasks.has(target)) clearMasks.set(target, []);
      clearMasks.get(target).push(enabled);
      const saved = temp([[enabled, versions.get(id), 1]]);
      write.push([enabled, saved, [[target, 1]]]);
    }
    write.push([enabled, temp([[enabled, nextPc, 1]]), [[pc, 1]]]);
  }
  clearMasks.set(pc, enables);
  const clear = [...clearMasks].sort((a, b) => a[0] - b[0]).map(([target, masks]) => [sum(...masks), channel(target), [[target, -1]]]);
  selectors.clear();
  const deep = branch.finish(clear, write), resetChannels = [marginFault, indexFault, jumpFault, stackFault], reset = [];
  if (!program.continuation) {
    resetChannels.push(pc, done);
    reset.push([channel(terminate), channel(pc), [[pc, 1]]], [ONE, channel(terminate), [[done, 1]]]);
  }
  for (const index of resetChannels) reset.push([ONE, channel(index), [[index, -1]]]);
  return {format: 'loopnet.reglu.v5', width, variables, done_channel: done, margin_channel: marginFault,
    branch_margin: margin, max_steps: program.max_steps, blocks: [pack(() => reset.values()), deep],
    schedule: [0, ...Array(program.max_steps).fill(1)], arrays: Object.fromEntries(program.arrays.filter((_, i) => !transient[i]).map(a => [a.name, a.cells.length])),
    index_channel: indexFault, terminate_channel: terminate, continuation: program.continuation,
    jump_channel: jumpFault, stack_channel: stackFault, max_stack_depth: program.max_stack_depth,
    pc_channel: pc, execution: program.execution, regions: program.regions.map(([start, end]) => [start, end])};
}

export async function loadProgram(url = new URL('./program.json', import.meta.url)) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('Source program unavailable');
  const program = await response.json();
  const source = await fetch(new URL(program.code, response.url));
  if (!source.ok) throw new Error('Source program data unavailable');
  if (typeof DecompressionStream === 'undefined') throw new Error('This browser needs support for gzip DecompressionStream');
  const buffer = await new Response(source.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  if (buffer.byteLength % 4) throw new Error('Invalid source program length');
  const code = new Uint32Array(buffer);
  if (new Uint8Array(Uint32Array.of(1).buffer)[0] !== 1) {
    const view = new DataView(buffer);
    for (let i = 0; i < code.length; i++) code[i] = view.getUint32(i * 4, true);
  }
  return {model: compileProgram(program, code), config: program.game};
}
