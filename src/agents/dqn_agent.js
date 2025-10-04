import * as tf from '@tensorflow/tfjs-node';
import { NStepAccumulator } from '../replay/n_step.js';
import { PrioritizedReplayBuffer } from '../replay/prioritized_buffer.js';
import { RingBuffer, mean, std, percentile } from '../utils/ring_buffer.js';

const DEFAULT_LAYERS = [256, 256, 128];

export class DQNAgent {
  constructor({
    stateDim,
    actionDim,
    envCount = 1,
    gamma = 0.98,
    lr = 5e-4,
    batchSize = 128,
    bufferSize = 500000,
    priorityAlpha = 0.6,
    priorityBeta = 0.4,
    priorityBetaIncrement = 0.000002,
    priorityEps = 0.001,
    epsStart = 1.0,
    epsEnd = 0.12,
    epsDecay = 80000,
    targetSync = 2000,
    nStep = 3,
    dueling = true,
    double = false,
    layers = DEFAULT_LAYERS,
    gradientClip = 10,
    warmupSteps = 5000,
  }) {
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.envCount = envCount;
    this.gamma = gamma;
    this.lr = lr;
    this.batchSize = batchSize;
    this.targetSync = targetSync;
    this.gradientClip = gradientClip;
    this.warmupSteps = warmupSteps;

    this.epsStart = epsStart;
    this.epsEnd = epsEnd;
    this.epsDecay = epsDecay;
    this.epsilon = epsStart;

    this.trainStep = 0;
    this.globalStep = 0;

    this.buffer = new PrioritizedReplayBuffer(bufferSize, {
      alpha: priorityAlpha,
      beta: priorityBeta,
      betaIncrement: priorityBetaIncrement,
      priorityEps,
    });

    this.priorityAlpha = priorityAlpha;
    this.priorityBeta = priorityBeta;
    this.priorityBetaIncrement = priorityBetaIncrement;
    this.priorityEps = priorityEps;

    this.dueling = dueling;
    this.double = double ?? false;
    this.kind = this.double ? 'double-dqn' : 'dqn';
    this.layers = layers.slice();

    this.nStep = nStep;
    this.nStepBuffers = Array.from({ length: envCount }, () => new NStepAccumulator(nStep, this.gamma));

    this.optimizer = tf.train.adam(this.lr);
    this.online = this.buildModel();
    this.target = this.buildModel();
    this.syncTarget();

    this.lossHistory = new RingBuffer(2000);
    this.tdHistory = new RingBuffer(4000);
  }

  buildModel() {
    const input = tf.input({ shape: [this.stateDim] });
    let x = input;
    this.layers.forEach((units) => {
      x = tf.layers.dense({ units, activation: 'relu', kernelInitializer: 'heNormal' }).apply(x);
    });
    let output;
    if (this.dueling) {
      const advantage = tf.layers
        .dense({ units: 128, activation: 'relu', kernelInitializer: 'heNormal' })
        .apply(x);
      const advOut = tf.layers.dense({ units: this.actionDim, activation: 'linear' }).apply(advantage);
      const value = tf.layers
        .dense({ units: 128, activation: 'relu', kernelInitializer: 'heNormal' })
        .apply(x);
      const valOut = tf.layers.dense({ units: 1, activation: 'linear' }).apply(value);
      output = tf.layers.add().apply([advOut, valOut]);
    } else {
      output = tf.layers.dense({ units: this.actionDim, activation: 'linear' }).apply(x);
    }
    return tf.model({ inputs: input, outputs: output });
  }

  dispose() {
    this.online?.dispose();
    this.target?.dispose();
    this.optimizer?.dispose?.();
  }

  setEnvCount(count) {
    if (count === this.envCount) return;
    this.envCount = count;
    this.nStepBuffers = Array.from({ length: this.envCount }, () => new NStepAccumulator(this.nStep, this.gamma));
  }

  setGamma(value) {
    this.gamma = value;
    this.nStepBuffers.forEach((buf) => buf.setConfig(this.nStep, this.gamma));
  }

  setLearningRate(value) {
    if (value === this.lr) return;
    this.lr = value;
    this.optimizer.dispose?.();
    this.optimizer = tf.train.adam(this.lr);
  }

  setNStep(value) {
    const n = Math.max(1, value | 0);
    if (n === this.nStep) return;
    this.nStep = n;
    this.nStepBuffers.forEach((buf) => buf.setConfig(this.nStep, this.gamma));
  }

  setBatchSize(size) {
    this.batchSize = Math.max(1, size | 0);
  }

  setBufferSize(size) {
    this.buffer.setCapacity(size);
  }

  setTargetSync(steps) {
    this.targetSync = Math.max(1, steps | 0);
  }

  setPriorityAlpha(alpha) {
    this.priorityAlpha = alpha;
    this.buffer.setAlpha(alpha);
  }

  setPriorityBeta(beta) {
    this.priorityBeta = beta;
    this.buffer.setBeta(beta);
  }

  setPriorityBetaIncrement(inc) {
    this.priorityBetaIncrement = inc;
    this.buffer.setBetaIncrement(inc);
  }

  setPriorityEps(eps) {
    this.priorityEps = eps;
    this.buffer.setPriorityEps(eps);
  }

  setEpsilonSchedule({ start, end, decay }) {
    if (start !== undefined) this.epsStart = start;
    if (end !== undefined) this.epsEnd = end;
    if (decay !== undefined) this.epsDecay = decay;
    this.updateEpsilon(this.globalStep);
  }

  setGlobalStep(step) {
    this.globalStep = step;
    this.updateEpsilon(step);
  }

  updateEpsilon(step = this.globalStep) {
    const t = Math.min(1, step / this.epsDecay);
    this.epsilon = this.epsStart * (1 - t) + this.epsEnd * t;
    return this.epsilon;
  }

  syncTarget() {
    this.target.setWeights(this.online.getWeights());
  }

  act(state) {
    if (Math.random() < this.epsilon) {
      return (Math.random() * this.actionDim) | 0;
    }
    return tf.tidy(() => {
      const tensor = tf.tensor2d([state], [1, this.stateDim]);
      const action = this.online.predict(tensor).argMax(1).dataSync()[0];
      tensor.dispose();
      return action;
    });
  }

  greedyAction(state) {
    return tf.tidy(() => {
      const tensor = tf.tensor2d([state], [1, this.stateDim]);
      const action = this.online.predict(tensor).argMax(1).dataSync()[0];
      tensor.dispose();
      return action;
    });
  }

  recordTransition(envIndex, state, action, reward, nextState, done, weight = 1) {
    const buffer = this.nStepBuffers[envIndex];
    if (!buffer) return;
    const ready = buffer.push({ s: state, a: action, r: reward, ns: nextState, d: done });
    ready.forEach((sample) => this.buffer.push({ ...sample, w: weight }));
    if (done) {
      const tail = buffer.flush();
      tail.forEach((sample) => this.buffer.push({ ...sample, w: weight }));
    }
  }

  flushTransitions() {
    this.nStepBuffers.forEach((buffer) => {
      const tail = buffer.flush();
      tail.forEach((sample) => this.buffer.push(sample));
    });
  }

  async learn(repeats = 1) {
    if (this.buffer.size() < Math.max(this.batchSize, this.warmupSteps)) {
      return null;
    }
    let result = null;
    for (let i = 0; i < repeats; i++) {
      result = await this._learnOnce();
    }
    return result;
  }

  async _learnOnce() {
    const sample = this.buffer.sample(this.batchSize);
    if (!sample) return null;
    const { batch, indices, weights } = sample;

    const states = tf.tensor2d(batch.map((item) => Array.from(item.s)), [batch.length, this.stateDim]);
    const nextStates = tf.tensor2d(batch.map((item) => Array.from(item.ns)), [batch.length, this.stateDim]);
    const actions = tf.tensor1d(batch.map((item) => item.a), 'int32');
    const rewards = tf.tensor1d(batch.map((item) => item.r));
    const dones = tf.tensor1d(batch.map((item) => (item.d ? 1 : 0)));
    const isWeights = tf.tensor1d(weights.length ? weights : new Array(batch.length).fill(1));

    let tdTensor;
    const { value: lossTensor, grads } = tf.variableGrads(() => {
      const qPredAll = this.online.apply(states);
      const actionMask = tf.oneHot(actions, this.actionDim);
      const qPred = qPredAll.mul(actionMask).sum(1);

      const nextQTargetAll = this.target.apply(nextStates);
      let nextQ;
      if (this.double) {
        const nextQOnline = this.online.apply(nextStates);
        const nextActionIdx = nextQOnline.argMax(1);
        const nextActionMask = tf.oneHot(nextActionIdx, this.actionDim);
        const nextQDouble = nextQTargetAll.mul(nextActionMask).sum(1);
        nextQ = nextQDouble;
      } else {
        nextQ = nextQTargetAll.max(1);
      }
      const notDoneMask = tf.scalar(1).sub(dones);
      const targets = rewards.add(nextQ.mul(tf.scalar(this.gamma)).mul(notDoneMask));
      tdTensor = targets.sub(qPred);
      const loss = tdTensor.square().mul(isWeights).mean();
      return loss;
    }, this.online.trainableWeights);

    const gradList = this.online.trainableWeights.map((weight) => grads[weight.name]);
    const [clipped, globalNorm] = tf.clipByGlobalNorm(gradList, this.gradientClip);
    const gradMap = {};
    this.online.trainableWeights.forEach((weight, idx) => {
      gradMap[weight.name] = clipped[idx];
    });
    this.optimizer.applyGradients(gradMap);

    const lossValue = (await lossTensor.data())[0];
    const tdErrors = Array.from(await tdTensor.abs().data());
    this.buffer.updatePriorities(indices, tdErrors);
    tdErrors.forEach((err) => this.tdHistory.push(err));

    this.trainStep += 1;
    if (this.targetSync > 0 && this.trainStep % this.targetSync === 0) {
      this.syncTarget();
    }

    this.lossHistory.push(lossValue);

    // Clean up tensors
    lossTensor.dispose();
    tdTensor.dispose();
    states.dispose();
    nextStates.dispose();
    actions.dispose();
    rewards.dispose();
    dones.dispose();
    isWeights.dispose();
    gradList.forEach((grad) => grad.dispose());
    clipped.forEach((grad) => grad.dispose());
    globalNorm.dispose();

    return { loss: lossValue, tdErrors };
  }

  getLossStats(window = 500) {
    const values = this.lossHistory.toArray().slice(-window);
    return {
      mean: mean(values),
      std: std(values),
    };
  }

  getTdErrorStats(window = 1000) {
    const values = this.tdHistory.toArray().slice(-window).filter(Number.isFinite);
    if (!values.length) return { p50: 0, p95: 0 };
    return {
      p50: percentile(values, 50),
      p95: percentile(values, 95),
    };
  }

  pruneReplay(fraction = 0.5) {
    this.buffer.dropOldest(fraction);
  }

  async exportState() {
    const weights = await Promise.all(
      this.online.getWeights().map(async (tensor) => ({
        shape: tensor.shape,
        dtype: tensor.dtype,
        data: Buffer.from(await tensor.data()).toString('base64'),
      })),
    );
    return {
      version: 5,
      kind: this.kind,
      stateDim: this.stateDim,
      actionDim: this.actionDim,
      config: {
        gamma: this.gamma,
        lr: this.lr,
        batchSize: this.batchSize,
        bufferSize: this.buffer.capacity,
        priorityAlpha: this.priorityAlpha,
        priorityBeta: this.priorityBeta,
        priorityBetaIncrement: this.priorityBetaIncrement,
        priorityEps: this.priorityEps,
        epsStart: this.epsStart,
        epsEnd: this.epsEnd,
        epsDecay: this.epsDecay,
        targetSync: this.targetSync,
        nStep: this.nStep,
        dueling: this.dueling,
        double: this.double,
        layers: this.layers,
      },
      weights,
    };
  }

  async importState(state) {
    if (!state) throw new Error('Invalid state');
    if (state.stateDim && state.stateDim !== this.stateDim) {
      throw new Error('State dimension mismatch');
    }
    if (state.actionDim && state.actionDim !== this.actionDim) {
      throw new Error('Action dimension mismatch');
    }
    const cfg = state.config ?? {};
    this.setGamma(cfg.gamma ?? this.gamma);
    this.setLearningRate(cfg.lr ?? this.lr);
    this.setBatchSize(cfg.batchSize ?? this.batchSize);
    this.setBufferSize(cfg.bufferSize ?? this.buffer.capacity);
    this.setPriorityAlpha(cfg.priorityAlpha ?? this.priorityAlpha);
    this.setPriorityBeta(cfg.priorityBeta ?? this.priorityBeta);
    this.setPriorityBetaIncrement(cfg.priorityBetaIncrement ?? this.priorityBetaIncrement);
    this.setPriorityEps(cfg.priorityEps ?? this.priorityEps);
    this.setEpsilonSchedule({
      start: cfg.epsStart ?? this.epsStart,
      end: cfg.epsEnd ?? this.epsEnd,
      decay: cfg.epsDecay ?? this.epsDecay,
    });
    this.setTargetSync(cfg.targetSync ?? this.targetSync);
    this.setNStep(cfg.nStep ?? this.nStep);
    this.dueling = cfg.dueling ?? this.dueling;
    this.double = cfg.double ?? this.double;
    this.kind = this.double ? 'double-dqn' : 'dqn';
    this.layers = Array.isArray(cfg.layers) ? cfg.layers.slice() : this.layers;
    this.online.dispose();
    this.target.dispose();
    this.online = this.buildModel();
    this.target = this.buildModel();
    if (Array.isArray(state.weights)) {
      const tensors = state.weights.map((w) =>
        tf.tensor(Buffer.from(w.data, 'base64'), w.shape, w.dtype),
      );
      this.online.setWeights(tensors);
      this.target.setWeights(tensors);
      tensors.forEach((t) => t.dispose());
    }
    this.syncTarget();
  }

  getHyperparams() {
    return {
      gamma: this.gamma,
      lr: this.lr,
      batchSize: this.batchSize,
      bufferSize: this.buffer.capacity,
      priorityAlpha: this.priorityAlpha,
      priorityBeta: this.priorityBeta,
      priorityBetaIncrement: this.priorityBetaIncrement,
      priorityEps: this.priorityEps,
      epsStart: this.epsStart,
      epsEnd: this.epsEnd,
      epsDecay: this.epsDecay,
      targetSync: this.targetSync,
      nStep: this.nStep,
      dueling: this.dueling,
      double: this.double,
      layers: this.layers.slice(),
      epsilon: this.epsilon,
    };
  }
}
