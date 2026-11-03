import {ResidualNetwork} from './neural.mjs';
import {ActivationAtlas} from './activation-atlas.mjs';
import {loadProgram} from './compiler.mjs';

let network, config, state, pixels, atlas, atlasBuffer, frameNumber = 0;

function render(input = {}) {
  const values = Object.fromEntries(Object.keys(config.initial).map(name => [name, network.read(state, name)]));
  const frameValues = {...values,
    forward: input.forward || 0, strafe: input.strafe || 0,
    turn: input.turn || 0, fire: input.fire || 0};
  const next = network.initialState(frameValues);
  atlas.begin(next);
  const start = performance.now();
  const calls = network.run(next, 5000, atlas);
  const milliseconds = performance.now() - start;
  state = next;
  frameNumber++;
  const frame = Float64Array.from(pixels, index => state[index]);
  const atlasStart = performance.now();
  const atlasImage = atlas.render(atlasBuffer);
  const atlasMilliseconds = performance.now() - atlasStart;
  atlasBuffer = null;
  return {frame, frameNumber, milliseconds, calls, atlas: atlasImage, atlasMilliseconds};
}

const transfers = result => [result.frame.buffer, result.atlas.data.buffer];

self.onmessage = async ({data}) => {
  try {
    if (data.type === 'init') {
      const {model, config: settings} = await loadProgram();
      config = settings;
      network = new ResidualNetwork(model);
      atlas = new ActivationAtlas(network);
      const statistics = network.statistics();
      pixels = Int32Array.from({length: config.width * config.height}, (_, i) => network.addresses.get(`frame[${i}]`));
      state = network.initialState(config.initial);
      const result = render();
      self.postMessage({type: 'ready', id: data.id, config, statistics, atlasLayout: atlas.layout, ...result}, transfers(result));
    } else if (data.type === 'reset') {
      state = network.initialState(config.initial);
      frameNumber = 0;
      const result = render();
      self.postMessage({type: 'frame', id: data.id, ...result}, transfers(result));
    } else if (data.type === 'tick') {
      const result = render(data.input);
      self.postMessage({type: 'frame', id: data.id, ...result}, transfers(result));
    } else if (data.type === 'atlas-limit') {
      atlas.setNeuronLimit(data.neuronLimit, data.layerStride ?? 1);
      const image = atlas.render(atlasBuffer);
      atlasBuffer = null;
      self.postMessage({type: 'atlas-layout', id: data.id, atlasLayout: atlas.layout,
        atlas: image, frameNumber}, [image.data.buffer]);
    } else if (data.type === 'atlas-buffer') {
      atlasBuffer = data.buffer;
    } else if (data.type === 'atlas-value') {
      self.postMessage({type: 'atlas-value', requestId: data.requestId,
        value: atlas.inspect(data.stage, data.neuron), frameNumber});
    }
  } catch (error) {
    self.postMessage({type: 'error', id: data.id, message: error.message});
  }
};
