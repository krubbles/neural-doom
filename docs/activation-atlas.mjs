export const ATLAS_NEURON_LIMIT = 60;

/** One column per layer, clipping its displayed neurons in original order. */
export function layoutActivationColumns(stages, {gap = 1, neuronLimit = ATLAS_NEURON_LIMIT, layerStride = 1} = {}) {
  if (!Number.isInteger(neuronLimit) || neuronLimit < 1) throw new Error('Neuron limit must be a positive integer');
  if (!Number.isInteger(layerStride) || layerStride < 1) throw new Error('Layer stride must be a positive integer');
  if (!stages.length) return {width: 1, height: 1, count: 0, visibleCount: 0, tiles: []};
  let x = gap, height = 1, offset = 0, visibleCount = 0;
  const tiles = stages.map((stage, index) => {
    const included = index % layerStride === 0;
    const shown = included ? Math.min(stage.count, neuronLimit) : 0;
    const tile = {...stage, stage: index, offset, visibleCount: shown, x, y: gap, width: included ? 1 : 0, height: Math.max(1, shown)};
    offset += stage.count; visibleCount += shown;
    if (included) x += 1 + gap;
    height = Math.max(height, tile.height);
    return tile;
  });
  return {width: x, height: height + 2 * gap, count: offset, visibleCount, tiles};
}

/** Hit-test visible neurons only; omitted tails have no image coordinates. */
export function atlasNeuronAt(layout, x, y) {
  if (!Number.isInteger(x) || !Number.isInteger(y)) return null;
  const tile = layout.tiles.find(item => x >= item.x && y >= item.y && x < item.x + item.width && y < item.y + item.height);
  if (!tile) return null;
  const neuron = y - tile.y;
  return neuron < tile.visibleCount ? {tile, neuron} : null;
}

const key = ({slot, phase, layer}) => `${slot}:${phase}:${layer}`;

export class ActivationAtlas {
  constructor(network, {neuronLimit = ATLAS_NEURON_LIMIT, layerStride = 1} = {}) {
    const stages = [{phase: 'input', slot: null, block: null, layer: null, count: network.model.width}];
    for (const [slot, block] of network.model.schedule.entries()) {
      const item = network.blocks[block];
      if (item.kind === 'transient') {
        item.layers.forEach((layer, index) => stages.push({phase: 'hidden', slot, block, layer: index, count: layer.count}));
        for (const phase of ['clear', 'write']) stages.push({phase, slot, block, layer: null, count: item[phase].count});
      } else stages.push({phase: 'residual', slot, block, layer: null, count: item.count});
    }
    this.layout = layoutActivationColumns(stages);
    this.values = new Float64Array(this.layout.count);
    this.buffers = new Map(this.layout.tiles.map(tile => [key(tile), this.values.subarray(tile.offset, tile.offset + tile.count)]));
    this.setNeuronLimit(neuronLimit, layerStride);
    // Store RGBA colors through byte views so packed-word writes work on either endianness.
    const colors = new Uint8ClampedArray(512 * 4);
    for (let sign = 0; sign < 2; sign++) {
      const end = sign ? [240, 70, 70] : [80, 220, 220];
      for (let level = 0; level < 256; level++) {
        const at = (sign * 256 + level) * 4, strength = level / 255;
        colors[at] = end[0] * strength;
        colors[at + 1] = end[1] * strength;
        colors[at + 2] = end[2] * strength;
        colors[at + 3] = 255;
      }
    }
    this.colors = new Uint32Array(colors.buffer);
    this.background = new Uint32Array(Uint8Array.from([0, 0, 0, 255]).buffer)[0];
    this.maximum = 0;
  }

  setNeuronLimit(neuronLimit, layerStride = 1) {
    this.layout = layoutActivationColumns(this.layout.tiles, {neuronLimit, layerStride});
    this.visibleIndices = new Uint32Array(this.layout.visibleCount);
    this.locations = new Uint32Array(this.layout.visibleCount);
    let visible = 0;
    for (const tile of this.layout.tiles) {
      for (let unit = 0; unit < tile.visibleCount; unit++, visible++) {
        this.visibleIndices[visible] = tile.offset + unit;
        this.locations[visible] = (tile.y + unit) * this.layout.width + tile.x;
      }
    }
    this.maximum = 0;
  }

  begin(state) {
    this.values.set(state, this.layout.tiles[0].offset);
    this.recurrence = 0;
  }

  bufferFor(stage) {return this.buffers.get(key(stage));}

  record(stage) {this.recurrence = stage.recurrence;}

  inspect(stage, unit) {
    const tile = this.layout.tiles[stage];
    return tile && Number.isInteger(unit) && unit >= 0 && unit < tile.visibleCount ? this.values[tile.offset + unit] : null;
  }

  render(recycled = null) {
    const {width, height} = this.layout, bytes = width * height * 4;
    const data = new Uint8ClampedArray(recycled?.byteLength === bytes ? recycled : new ArrayBuffer(bytes));
    const words = new Uint32Array(data.buffer); words.fill(this.background);
    let active = 0, visible = 0;
    this.maximum = 0;
    for (const tile of this.layout.tiles) {
      let maximum = 0;
      for (let unit = 0; unit < tile.visibleCount; unit++) {
        const value = this.values[tile.offset + unit];
        maximum = Math.max(maximum, Math.abs(value));
        if (value !== 0) active++;
      }
      this.maximum = Math.max(this.maximum, maximum);
      const denominator = Math.log1p(maximum);
      // Each layer's current visible values set its own logarithmic color range.
      // Only pixels change; recorded float64 activations remain untouched.
      for (let unit = 0; unit < tile.visibleCount; unit++, visible++) {
        const value = this.values[tile.offset + unit];
        const level = denominator > 0 ? Math.min(255, Math.round(Math.log1p(Math.abs(value)) / denominator * 255)) : 0;
        words[this.locations[visible]] = this.colors[(value < 0 ? 256 : 0) + level];
      }
    }
    return {width, height, data, maximum: this.maximum, active, count: this.layout.visibleCount};
  }
}
