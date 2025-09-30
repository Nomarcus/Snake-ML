export class NStepAccumulator {
  constructor(n = 1, gamma = 0.99, lambdaValue = 1) {
    this.setConfig(n, gamma, lambdaValue);
  }

  setConfig(n, gamma, lambdaValue = this.lambda ?? 1) {
    this.n = Math.max(1, n | 0);
    this.gamma = gamma;
    this.lambda = Math.max(0, Math.min(1, lambdaValue ?? 1));
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
    const first = this.queue[0];
    const limit = Math.min(this.n, this.queue.length);
    const partials = [];
    let reward = 0;
    let discount = 1;
    for (let i = 0; i < limit; i++) {
      const step = this.queue[i];
      reward += discount * step.r;
      discount *= this.gamma;
      partials.push({ reward, nextState: step.ns, done: step.d });
      if (step.d) break;
    }
    const last = partials[partials.length - 1];
    const lambda = this.lambda;
    let lambdaReturn;
    if (!partials.length) {
      lambdaReturn = 0;
    } else if (lambda >= 0.999) {
      lambdaReturn = partials[partials.length - 1].reward;
    } else {
      lambdaReturn = 0;
      for (let i = 0; i < partials.length; i++) {
        const weight = i === partials.length - 1 ? Math.pow(lambda, i) : (1 - lambda) * Math.pow(lambda, i);
        lambdaReturn += weight * partials[i].reward;
      }
    }
    const nextState = last ? last.nextState : first.ns;
    const done = last ? last.done : false;
    return { s: first.s, a: first.a, r: lambdaReturn, ns: nextState, d: done };
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
