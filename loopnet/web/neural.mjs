/** Weight-only float64 residual ReGLU evaluator. No source-language opcodes. */
export class ResidualNetwork {
  constructor(model) {
    if (!['loopnet.reglu.v4', 'loopnet.reglu.v5'].includes(model.format)) throw new Error('Unsupported network format');
    this.model = model;
    this.addresses = new Map(model.variables.map((name, index) => [name, index]));
    this.blocks = model.blocks.map(block => block.kind === 'transient'
      ? { kind: 'transient', transientWidth: block.transient_width,
          layers: block.layers.map(layer => this.pack(layer)),
          clear: this.pack(block.clear), write: this.pack(block.write) }
      : this.pack(block));
    const workWidth = model.width + Math.max(0, ...this.blocks.map(block => block.transientWidth || 0));
    this.work = new Float64Array(workWidth);
    this.delta = new Float64Array(workWidth);
    this.tags = new Uint32Array(workWidth);
    this.touched = new Int32Array(workWidth);
    this.epoch = 0;
  }

  pack(block) {
    // Browser synthesis already produces typed sparse matrices locally.
    if (block.packed === true) return block;
    const units = block.units;
    const packRows = key => {
      const pointers = [0], indices = [], weights = [], biases = [];
      for (const unit of units) {
        const row = unit[key];
        biases.push(row.bias);
        for (const [index, weight] of row.terms) { indices.push(index); weights.push(weight); }
        pointers.push(indices.length);
      }
      return { pointers: Int32Array.from(pointers), indices: Int32Array.from(indices),
        weights: Float64Array.from(weights), biases: Float64Array.from(biases) };
    };
    const pointers = [0], indices = [], weights = [];
    for (const unit of units) {
      for (const [index, weight] of unit.outputs) { indices.push(index); weights.push(weight); }
      pointers.push(indices.length);
    }
    const targets = [...new Set([...indices, ...block.bias.map(([index]) => index)])];
    return { count: units.length, gate: packRows('gate'), value: packRows('value'), targets,
      output: { pointers: Int32Array.from(pointers), indices: Int32Array.from(indices),
        weights: Float64Array.from(weights) }, bias: block.bias };
  }

  initialState(values = {}) {
    const state = new Float64Array(this.model.width);
    for (const [name, value] of Object.entries(values)) {
      if (!this.addresses.has(name) || !Number.isFinite(value)) throw new Error(`Invalid global: ${name}`);
      state[this.addresses.get(name)] = value;
    }
    return state;
  }

  read(state, name) { return state[this.addresses.get(name)]; }

  statistics() {
    // Parameters belong to unique modules; recurrent schedule reuse adds depth,
    // not new weights. Count the sparse coefficient and bias slots we store.
    let parameters = 0, layers = 0;
    for (const block of this.blocks) {
      const stages = block.kind === 'transient' ? [...block.layers, block.clear, block.write] : [block];
      for (const stage of stages) parameters += stage.gate.weights.length + stage.gate.biases.length
        + stage.value.weights.length + stage.value.biases.length + stage.output.weights.length + stage.bias.length;
    }
    for (const id of this.model.schedule) {
      const block = this.blocks[id];
      layers += block.kind === 'transient' ? block.layers.length + 2 : 1;
    }
    return {parameters, layers};
  }

  // onStage receives the actual ReGLU outputs and a read-only view of the
  // residual stream. A function receives fresh snapshots; a recording object
  // can supply bufferFor(metadata) and record(stage) to reuse live buffers.
  forward(state, onStage = null, recurrence = 1) {
    if (state.length !== this.model.width) throw new Error('Wrong residual-stream width');
    const observer = typeof onStage === 'function' ? {record: onStage} : onStage;
    for (let slot = 0; slot < this.model.schedule.length; slot++) {
      const id = this.model.schedule[slot];
      const block = this.blocks[id];
      if (block.kind === 'transient') {
        // Only the persistent input is copied. Every hidden output is freshly
        // assigned by its feedforward layer and is never returned as state.
        const work = this.work;
        work.set(state);
        for (let index = 0; index < block.layers.length; index++) {
          const layer = block.layers[index], values = observer ? this.stageBuffer(observer, 'hidden', index, id, recurrence, layer.count, slot) : null;
          this.activate(layer, work, values);
          if (observer) observer.record({phase: 'hidden', layer: index, block: id, slot, recurrence, values, state});
        }
        for (const phase of ['clear', 'write']) {
          const layer = block[phase], values = observer ? this.stageBuffer(observer, phase, null, id, recurrence, layer.count, slot) : null;
          this.residual(layer, work, state, values);
          if (observer) observer.record({phase, layer: null, block: id, slot, recurrence, values, state});
        }
      } else {
        const values = observer ? this.stageBuffer(observer, 'residual', null, id, recurrence, block.count, slot) : null;
        this.residual(block, state, state, values);
        if (observer) observer.record({phase: 'residual', layer: null, block: id, slot, recurrence, values, state});
      }
    }
    return state;
  }

  stageBuffer(observer, phase, layer, block, recurrence, count, slot) {
    if (!observer.bufferFor) return new Float64Array(count);
    const values = observer.bufferFor({phase, layer, block, slot, recurrence, count});
    if (!(values instanceof Float64Array) || values.length !== count) throw new Error('Wrong activation recording buffer');
    values.fill(0); // Inactive gates must not retain values from the previous frame.
    return values;
  }

  activate(block, work, activations = null) {
    for (const index of block.targets) work[index] = 0;
    const gate = block.gate, value = block.value, output = block.output;
    for (let unit = 0; unit < block.count; unit++) {
      let g = gate.biases[unit];
      for (let j = gate.pointers[unit]; j < gate.pointers[unit + 1]; j++) {
        g += gate.weights[j] * work[gate.indices[j]];
      }
      if (g <= 0) continue;
      let v = value.biases[unit];
      for (let j = value.pointers[unit]; j < value.pointers[unit + 1]; j++) {
        v += value.weights[j] * work[value.indices[j]];
      }
      const hidden = g * v;
      if (activations) activations[unit] = hidden;
      if (hidden === 0) continue;
      for (let j = output.pointers[unit]; j < output.pointers[unit + 1]; j++) {
        work[output.indices[j]] += output.weights[j] * hidden;
      }
    }
    for (const [index, bias] of block.bias) work[index] += bias;
    for (const index of block.targets) {
      if (!Number.isFinite(work[index])) throw new Error('Nonfinite neural activation');
    }
  }

  residual(block, input, state, activations = null) {
    const { delta, tags, touched } = this;
    let epoch = ++this.epoch;
    if (epoch >= 0xffffffff) { tags.fill(0); this.epoch = epoch = 1; }
    let used = 0;
    const gate = block.gate, value = block.value, output = block.output;
    for (let unit = 0; unit < block.count; unit++) {
      let g = gate.biases[unit];
      for (let j = gate.pointers[unit]; j < gate.pointers[unit + 1]; j++) {
        g += gate.weights[j] * input[gate.indices[j]];
      }
      // Activation sparsity, independent of source operations or the pc.
      if (g <= 0) continue;
      let v = value.biases[unit];
      for (let j = value.pointers[unit]; j < value.pointers[unit + 1]; j++) {
        v += value.weights[j] * input[value.indices[j]];
      }
      const hidden = g * v;
      if (activations) activations[unit] = hidden;
      if (hidden === 0) continue;
      for (let j = output.pointers[unit]; j < output.pointers[unit + 1]; j++) {
        const index = output.indices[j];
        if (tags[index] !== epoch) { tags[index] = epoch; delta[index] = 0; touched[used++] = index; }
        delta[index] += output.weights[j] * hidden;
      }
    }
    for (const [index, bias] of block.bias) {
      if (tags[index] !== epoch) { tags[index] = epoch; delta[index] = 0; touched[used++] = index; }
      delta[index] += bias;
    }
    // Commit only after every hidden unit has read the previous state.
    for (let j = 0; j < used; j++) { const index = touched[j]; state[index] += delta[index]; }
  }

  assertValid(state) {
    const m = this.model;
    for (const value of state) if (!Number.isFinite(value)) throw new Error('Nonfinite neural state');
    if (state[m.stack_channel] > 0) throw new Error('Call-stack limit exceeded');
    if (state[m.jump_channel] > 0) throw new Error('Invalid instruction pointer');
    if (state[m.index_channel] > 0) throw new Error('Invalid array index');
    if (state[m.margin_channel] > 0) throw new Error('Branch condition violated its margin');
    if (![0, 1].includes(state[m.terminate_channel])) throw new Error('Invalid termination output');
  }

  run(state, maxCalls = 5000, onStage = null) {
    let calls = 0;
    while (state[this.model.terminate_channel] !== 1) {
      if (calls >= maxCalls) throw new Error('Move exceeded its recurrence budget');
      this.forward(state, onStage, calls + 1);
      this.assertValid(state);
      calls++;
    }
    return calls;
  }
}
