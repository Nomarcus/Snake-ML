export class PrioritizedReplayBuffer {
  constructor(capacity = 50000, opts = {}) {
    this.capacity = Math.max(1, capacity | 0);
    this.buffer = [];
    this.position = 0;
    this.alpha = opts.alpha ?? 0.6;
    this.beta = opts.beta ?? 0.4;
    this.betaIncrement = opts.betaIncrement ?? 0.000002;
    this.priorityEps = opts.priorityEps ?? 0.001;
    this.priorities = new Float32Array(this.capacity);
    this.maxPriority = this.priorityEps;
  }

  size() {
    return this.buffer.length;
  }

  setCapacity(cap) {
    const newCap = Math.max(1, cap | 0);
    if (newCap === this.capacity) return;
    this.capacity = newCap;
    if (this.buffer.length > this.capacity) {
      this.buffer = this.buffer.slice(-this.capacity);
    }
    this.position = Math.min(this.position, this.capacity - 1);
    this.priorities = new Float32Array(this.capacity);
    this.maxPriority = this.priorityEps;
  }

  setAlpha(val) {
    this.alpha = Math.max(0.01, +val || 0.01);
  }

  setBeta(val) {
    this.beta = Math.min(1, Math.max(0, +val || 0));
  }

  setBetaIncrement(val) {
    this.betaIncrement = Math.max(0, +val || 0);
  }

  setPriorityEps(val) {
    this.priorityEps = Math.max(1e-6, +val || 1e-6);
  }

  push(sample) {
    const entry = {
      s: Float32Array.from(sample.s),
      a: sample.a | 0,
      r: +sample.r,
      ns: Float32Array.from(sample.ns),
      d: !!sample.d,
      w: sample.w ?? 1,
    };
    if (this.buffer.length < this.capacity) {
      this.buffer.push(entry);
      const idx = this.buffer.length - 1;
      this.priorities[idx] = this.maxPriority;
    } else {
      const idx = this.position % this.capacity;
      this.buffer[idx] = entry;
      this.priorities[idx] = this.maxPriority;
    }
    this.position = (this.position + 1) % this.capacity;
  }

  sample(batchSize) {
    if (!this.buffer.length) return null;
    const size = Math.min(batchSize, this.buffer.length);
    const priorities = this.priorities.slice(0, this.buffer.length);
    const probs = priorities.map((p) => Math.pow(p || this.priorityEps, this.alpha));
    const sum = probs.reduce((a, b) => a + b, 0) || 1;
    const normalized = probs.map((p) => p / sum);
    const batch = [];
    const indices = [];
    const weights = [];
    const beta = this.beta;
    this.beta = Math.min(1, this.beta + this.betaIncrement);
    const maxWeight = Math.pow(this.buffer.length * Math.min(...normalized), -beta);
    for (let i = 0; i < size; i++) {
      const r = Math.random();
      let acc = 0;
      let index = 0;
      for (let j = 0; j < normalized.length; j++) {
        acc += normalized[j];
        if (r <= acc) {
          index = j;
          break;
        }
      }
      batch.push(this.buffer[index]);
      indices.push(index);
      const w = Math.pow(this.buffer.length * normalized[index], -beta);
      weights.push(w / (maxWeight || 1));
    }
    return { batch, indices, weights };
  }

  updatePriorities(indices, priorities) {
    indices.forEach((idx, i) => {
      const priority = Math.max(this.priorityEps, priorities[i]);
      this.priorities[idx] = priority;
      this.maxPriority = Math.max(this.maxPriority, priority);
    });
  }

  dropOldest(fraction = 0.5) {
    const removeCount = Math.floor(this.buffer.length * clampFraction(fraction));
    if (removeCount <= 0) return;
    this.buffer = this.buffer.slice(removeCount);
    this.priorities = new Float32Array(this.capacity);
    this.maxPriority = this.priorityEps;
    this.position = this.buffer.length % this.capacity;
  }

  toJSON() {
    return {
      cap: this.capacity,
      buf: this.buffer.map((item) => ({
        s: Array.from(item.s),
        a: item.a,
        r: item.r,
        ns: Array.from(item.ns),
        d: item.d,
        w: item.w,
      })),
      pos: this.position,
      alpha: this.alpha,
      beta: this.beta,
      betaIncrement: this.betaIncrement,
      priorityEps: this.priorityEps,
      priorities: Array.from(this.priorities),
      maxPriority: this.maxPriority,
    };
  }
}

function clampFraction(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(Math.max(value, 0), 1);
}
