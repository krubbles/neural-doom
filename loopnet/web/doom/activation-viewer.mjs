import {activationImage, residualOverview} from '/activation-trace.mjs';

/** The completed, non-pixel residual channels from the actual game frame. */
export class ResidualViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d', {alpha: false});
    canvas.addEventListener('pointermove', event => {
      this.pointer = {clientX: event.clientX, clientY: event.clientY};
      this.inspect();
    });
    canvas.addEventListener('pointerleave', () => {this.pointer = null; canvas.title = '';});
  }

  ready(layout) {
    this.layout = layout;
    this.overview = residualOverview({...layout, residuals: [], stages: []});
    this.maximum = 1;
    this.canvas.setAttribute('aria-label', `Residual state: ${this.overview.indices.length} globals and control channels; pixel outputs excluded`);
  }

  draw({residual, frameNumber}) {
    this.values = Float64Array.from(this.overview.indices, index => residual[index]);
    for (let unit = 0; unit < this.values.length; unit++) {
      if (this.overview.indices[unit] < this.layout.variableCount) this.maximum = Math.max(this.maximum, Math.abs(this.values[unit]));
    }
    const image = activationImage(this.values, this.maximum);
    for (let unit = 0; unit < this.values.length; unit++) {
      if (this.values[unit] === 0) image.data.fill(0, unit * 4, unit * 4 + 3);
    }
    if (this.canvas.width !== image.width || this.canvas.height !== image.height) {
      this.canvas.width = image.width; this.canvas.height = image.height;
    }
    this.context.putImageData(new ImageData(image.data, image.width, image.height), 0, 0);
    this.canvas.dataset.frame = frameNumber;
    if (this.pointer) this.inspect();
  }

  inspect() {
    if (!this.values || !this.pointer) return;
    const box = this.canvas.getBoundingClientRect();
    const x = Math.floor((this.pointer.clientX - box.left) / box.width * this.canvas.width);
    const y = Math.floor((this.pointer.clientY - box.top) / box.height * this.canvas.height);
    const unit = y * this.canvas.width + x;
    this.canvas.title = unit >= 0 && unit < this.values.length
      ? `${this.layout.channels[this.overview.indices[unit]]} = ${this.values[unit]}` : '';
  }
}
