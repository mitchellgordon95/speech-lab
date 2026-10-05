// Keep only the newest 160 ms window. The browser supplies mono audio at 16 kHz.
class SoundWindow extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.ring = new Float32Array(2560);
    this.cursor = 0;
    this.seen = 0;
    this.hop = 0;
    this.hopSamples = options.processorOptions?.hopSamples === 640 ? 640 : 1280;
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    for (const sample of input) {
      this.ring[this.cursor] = sample;
      this.cursor = (this.cursor + 1) % this.ring.length;
      this.seen++;
      this.hop++;
      if (this.seen >= 2560 && this.hop >= this.hopSamples) {
        this.hop = 0;
        const frame = new Float32Array(2560);
        frame.set(this.ring.subarray(this.cursor));
        frame.set(this.ring.subarray(0, this.cursor), 2560 - this.cursor);
        this.port.postMessage(frame.buffer, [frame.buffer]);
      }
    }
    return true;
  }
}
registerProcessor("sound-window", SoundWindow);
