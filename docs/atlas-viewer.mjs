import {atlasNeuronAt} from './activation-atlas.mjs';

export class AtlasViewer {
  constructor(root, inspect, recycle) {
    this.root = root;
    this.canvas = root.querySelector('canvas');
    this.context = this.canvas.getContext('2d', {alpha: false});
    this.staging = document.createElement('canvas');
    this.stagingContext = this.staging.getContext('2d', {alpha: false});
    this.vertical = false;
    this.inspect = inspect; this.recycle = recycle; this.serial = 0;
    this.canvas.addEventListener('pointermove', event => {
      this.pointer = {clientX: event.clientX, clientY: event.clientY};
      this.hover();
    });
    this.canvas.addEventListener('pointerleave', () => {
      this.pointer = null; this.target = null; this.serial++; this.canvas.title = '';
    });
  }

  ready(layout) {
    this.layout = layout;
    this.canvas.width = this.vertical ? layout.height : layout.width;
    this.canvas.height = this.vertical ? layout.width : layout.height;
    const limit = Math.max(0, ...layout.tiles.map(tile => tile.visibleCount));
    this.canvas.dataset.neuronLimit = limit;
    this.canvas.dataset.layerCount = layout.tiles.filter(tile => tile.width > 0).length;
    this.canvas.setAttribute('aria-label', `Live neurons: the first ${limit} activations of each displayed layer`);
  }

  setVertical(vertical) {
    this.vertical = vertical;
    this.pointer = null; this.target = null; this.serial++; this.canvas.title = '';
    if (this.layout) this.ready(this.layout);
  }

  draw(atlas, frameNumber) {
    this.frameNumber = frameNumber;
    const image = new ImageData(atlas.data, atlas.width, atlas.height);
    if (this.vertical) {
      if (this.staging.width !== atlas.width || this.staging.height !== atlas.height) {
        this.staging.width = atlas.width; this.staging.height = atlas.height;
      }
      this.stagingContext.putImageData(image, 0, 0);
      this.context.save();
      this.context.setTransform(0, 1, -1, 0, atlas.height, 0);
      this.context.drawImage(this.staging, 0, 0);
      this.context.restore();
    } else this.context.putImageData(image, 0, 0);
    this.root.dataset.frame = frameNumber;
    this.recycle(atlas.data.buffer);
    if (this.pointer) this.hover();
  }

  hover() {
    if (this.vertical || !this.layout || !this.pointer) return;
    const box = this.canvas.getBoundingClientRect();
    const x = Math.floor((this.pointer.clientX - box.left) / box.width * this.layout.width);
    const y = Math.floor((this.pointer.clientY - box.top) / box.height * this.layout.height);
    const hit = atlasNeuronAt(this.layout, x, y);
    if (!hit) {this.target = null; this.serial++; this.canvas.title = ''; return;}
    const key = `${this.frameNumber}:${hit.tile.stage}:${hit.neuron}`;
    if (key === this.target) return;
    this.target = key; this.tile = hit.tile; this.neuron = hit.neuron;
    this.inspect({stage: hit.tile.stage, neuron: hit.neuron, requestId: ++this.serial});
  }

  inspected(data) {
    if (data.requestId !== this.serial || !this.target) return;
    this.canvas.title = `Layer ${this.tile.stage}, neuron ${this.neuron} = ${data.value}`;
  }
}
