export class NStepAccumulator {
  constructor(n = 1, gamma = 0.99) {
    this.setConfig(n, gamma);
  }

  setConfig(n, gamma) {
    this.n = Math.max(1, n | 0);
    this.gamma = gamma;
    this.queue = [];
  }

  push(step) {
    const item = {
      s: Float32Array.from(step.s),
      a: step.a | 0,
      r: +step.r,
      ns: Float32Array.from(step.ns),
      d: !!step.d,
    };
    this.queue.push(item);
    const ready = [];
    while (this.queue.length >= this.n) {
      ready.push(this.build());
      this.queue.shift();
      if (ready[ready.length - 1].d) {
        this.queue.length = 0;
        return ready;
      }
    }
    if (item.d) {
      ready.push(...this.flush());
    }
    return ready;
  }

  build() {
    let reward = 0;
    let discount = 1;
    let done = false;
    let nextState = this.queue[0].ns;
    const limit = Math.min(this.n, this.queue.length);
    for (let i = 0; i < limit; i++) {
      const step = this.queue[i];
      reward += discount * step.r;
      discount *= this.gamma;
      nextState = step.ns;
      if (step.d) {
        done = true;
        break;
      }
    }
    const first = this.queue[0];
    return { s: first.s, a: first.a, r: reward, ns: nextState, d: done };
  }

  flush() {
    const out = [];
    while (this.queue.length) {
      out.push(this.build());
      this.queue.shift();
    }
    return out;
  }
}
