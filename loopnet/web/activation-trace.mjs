/** Capture real neural outputs; the source program is never interpreted here. */
export function activationStatistics(values) {
  let minimum = Infinity, maximum = -Infinity, active = 0;
  for (const value of values) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    if (value !== 0) active++;
  }
  return {minimum: values.length ? minimum : 0, maximum: values.length ? maximum : 0,
    active, count: values.length};
}

export function residualChannels(network) {
  const channels = Array.from({length: network.model.width}, (_, index) =>
    network.model.variables[index] || `channel ${index}`);
  for (const key of ['pc', 'done', 'terminate', 'stack', 'jump', 'index', 'margin']) {
    const index = network.model[`${key}_channel`];
    if (Number.isInteger(index)) channels[index] = key;
  }
  return channels;
}

export function captureActivations(network, input, {maxCalls = 5000} = {}) {
  const state = input.slice(), residuals = [state.slice()];
  const residualStats = [activationStatistics(state)];
  const stages = [{phase: 'input', recurrence: 0, block: null, layer: null,
    values: residuals[0], residual: 0, stats: residualStats[0]}];
  let maximum = Math.max(Math.abs(residualStats[0].minimum), Math.abs(residualStats[0].maximum));
  let residual = 0;
  const include = stats => {maximum = Math.max(maximum, Math.abs(stats.minimum), Math.abs(stats.maximum));};
  const calls = network.run(state, maxCalls, stage => {
    if (stage.phase !== 'hidden') {
      residual = residuals.length;
      residuals.push(stage.state.slice());
      residualStats.push(activationStatistics(stage.state));
      include(residualStats.at(-1));
    }
    const stats = activationStatistics(stage.values);
    include(stats);
    stages.push({phase: stage.phase, recurrence: stage.recurrence, block: stage.block,
      layer: stage.layer, values: stage.values, residual, stats});
  });
  return {stages, residuals, residualStats, maximum, channels: residualChannels(network), calls,
    variableCount: network.model.variables.length};
}

/** Keep small program values visible even when the instruction pointer is huge.
 * The state range is fixed over the capture; larger control values saturate.
 */
export function residualOverview(trace) {
  const pixels = new Set(trace.pixels || []);
  const indices = Uint32Array.from(trace.channels.flatMap((_, index) => pixels.has(index) ? [] : [index]));
  let maximum = 1;
  for (const state of trace.residuals) {
    for (let index = 0; index < (trace.variableCount ?? state.length); index++) {
      maximum = Math.max(maximum, Math.abs(state[index]));
    }
  }
  let writeback = trace.stages.findLastIndex(stage => stage.phase === 'write');
  if (writeback < 0) writeback = trace.stages.length - 1;
  return {indices, maximum, writeback};
}

/** A fixed signed logarithmic color scale, shared by every image in a capture. */
export function activationImage(values, maximum) {
  const width = Math.max(1, Math.min(256, Math.ceil(Math.sqrt(values.length))));
  const height = Math.max(1, Math.ceil(values.length / width));
  const data = new Uint8ClampedArray(width * height * 4);
  const denominator = Math.log1p(maximum || 1);
  for (let index = 0; index < values.length; index++) {
    const value = values[index], strength = Math.min(1, Math.log1p(Math.abs(value)) / denominator);
    const end = value < 0 ? [240, 139, 105] : [195, 237, 140], at = index * 4;
    data[at] = 11 + (end[0] - 11) * strength;
    data[at + 1] = 16 + (end[1] - 16) * strength;
    data[at + 2] = 15 + (end[2] - 15) * strength;
    data[at + 3] = 255;
  }
  return {width, height, data};
}
