export class RingBuffer {
  constructor(capacity) {
    if (!Number.isFinite(capacity) || capacity <= 0) {
      throw new Error('RingBuffer capacity must be > 0');
    }
    this.capacity = capacity | 0;
    this.values = new Array(this.capacity);
    this.index = 0;
    this.length = 0;
  }

  push(value) {
    this.values[this.index] = value;
    this.index = (this.index + 1) % this.capacity;
    if (this.length < this.capacity) {
      this.length++;
    }
  }

  clear() {
    this.index = 0;
    this.length = 0;
    this.values.fill(undefined);
  }

  toArray() {
    if (!this.length) return [];
    const result = new Array(this.length);
    const start = (this.index - this.length + this.capacity) % this.capacity;
    for (let i = 0; i < this.length; i++) {
      result[i] = this.values[(start + i) % this.capacity];
    }
    return result;
  }

  last() {
    if (!this.length) return undefined;
    const idx = (this.index - 1 + this.capacity) % this.capacity;
    return this.values[idx];
  }
}

export function mean(arr) {
  if (!arr?.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

export function std(arr) {
  if (!arr?.length) return 0;
  const m = mean(arr);
  const variance = arr.reduce((acc, value) => acc + (value - m) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

export function percentile(arr, p) {
  if (!arr?.length) return 0;
  const values = [...arr].sort((a, b) => a - b);
  const rank = (p / 100) * (values.length - 1);
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  if (lower === upper) return values[lower];
  const weight = rank - lower;
  return values[lower] * (1 - weight) + values[upper] * weight;
}

export function linearRegressionSlope(samples) {
  if (!samples?.length || samples.length < 2) return 0;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;
  const n = samples.length;
  for (let i = 0; i < n; i++) {
    const x = i;
    const y = samples[i];
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
  }
  const denominator = n * sumXX - sumX ** 2;
  if (denominator === 0) return 0;
  return (n * sumXY - sumX * sumY) / denominator;
}

export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export function movingAverage(values, window) {
  if (!values?.length) return [];
  const result = [];
  let acc = 0;
  const queue = [];
  for (const value of values) {
    queue.push(value);
    acc += value;
    if (queue.length > window) {
      acc -= queue.shift();
    }
    if (queue.length === window) {
      result.push(acc / window);
    }
  }
  return result;
}
