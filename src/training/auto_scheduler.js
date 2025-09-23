import { clamp } from '../utils/ring_buffer.js';
import { MetricsCollector } from './metrics_collector.js';

const EPS_END_MIN = 0.01;
const EPS_END_MAX = 0.3;
const EPS_DECAY_MIN = 5000;
const EPS_DECAY_MAX = 200000;
const GAMMA_MIN = 0.9;
const GAMMA_MAX = 0.999;
const LR_MIN = 2e-4;
const LR_MAX = 5e-4;
const NSTEP_MIN = 1;
const NSTEP_MAX = 5;
const ADJUST_COOLDOWN_EPISODES = 500;
const EVAL_INTERVAL_EPISODES = 2000;
const ROLLBACK_WINDOW_EPISODES = 5000;

export class AutoScheduler {
  constructor({
    initialConfig,
    rewardConfig,
    minSaveCooldownMinutes = 10,
    cosineCycleEpisodes = 200000,
  }) {
    this.metrics = new MetricsCollector();
    this.baseEpsEnd = initialConfig.epsEnd;
    this.baseEpsDecay = initialConfig.epsDecay;
    this.baseGamma = initialConfig.gamma ?? 0.98;
    this.currentEpsEnd = initialConfig.epsEnd;
    this.currentEpsDecay = initialConfig.epsDecay;
    this.epsilonBoost = null;

    this.lrScale = 1;
    this.cosineCycle = cosineCycleEpisodes;
    this.lastLRAdjustEpisode = 0;

    this.lastAdjustEpisode = new Map();
    this.lastEvalEpisode = 0;
    this.lastEvalTime = 0;
    this.lastCheckpointTime = 0;
    this.minSaveCooldownMs = minSaveCooldownMinutes * 60 * 1000;

    this.rewardConfig = { ...rewardConfig };
    this.boardStages = [
      { size: 10, threshold: 0 },
      { size: 14, threshold: 60 },
      { size: 18, threshold: 120 },
      { size: 20, threshold: 200 },
    ];
    this.boardIndex = 0;
    this.lastBoardChangeEpisode = 0;

    this.tdMeanHistory = [];
    this.bestEval = null;
    this.bestEvalEpisode = 0;
  }

  observe(summary, tdStats, episode) {
    this.metrics.recordLoss(summary.tdLossMean ?? 0);
    this.tdMeanHistory.push(summary.tdLossMean ?? 0);
    if (this.tdMeanHistory.length > 10) this.tdMeanHistory.shift();
  }

  getMetrics() {
    return this.metrics.getSummary();
  }

  recordEpisode(data) {
    const record = this.metrics.toEpisodeRecord(data);
    this.metrics.recordEpisode(record);
  }

  maybeAdvanceBoard(episode) {
    const summary = this.metrics.getSummary();
    const nextStage = this.boardStages[this.boardIndex + 1];
    if (!nextStage) return null;
    if (summary.maFruit500 > nextStage.threshold) {
      this.boardIndex += 1;
      this.lastBoardChangeEpisode = episode;
      return this.boardStages[this.boardIndex].size;
    }
    return null;
  }

  shouldCheckpoint(now) {
    if (now - this.lastCheckpointTime < this.minSaveCooldownMs) return false;
    this.lastCheckpointTime = now;
    return true;
  }

  maybeEval(episode) {
    return episode - this.lastEvalEpisode >= EVAL_INTERVAL_EPISODES;
  }

  registerEval(episode) {
    this.lastEvalEpisode = episode;
    this.lastEvalTime = Date.now();
  }

  updateBestEval(episode, fruits, meta) {
    if (!this.bestEval || fruits >= this.bestEval.fruits + 5 || (fruits >= this.bestEval.fruits * 1.02)) {
      this.bestEval = { episode, fruits, meta };
      this.bestEvalEpisode = episode;
      return true;
    }
    return false;
  }

  needsRollback(metrics, episode) {
    if (!this.bestEval) return false;
    if (episode - this.bestEvalEpisode < ROLLBACK_WINDOW_EPISODES) return false;
    const threshold = this.bestEval.fruits * 0.75;
    return metrics.maFruit100 < threshold;
  }

  _canAdjust(key, episode) {
    const last = this.lastAdjustEpisode.get(key) ?? -Infinity;
    if (episode - last < ADJUST_COOLDOWN_EPISODES) return false;
    this.lastAdjustEpisode.set(key, episode);
    return true;
  }

  tuneHyperparams(agent, rewardConfig, metrics, tdStats, episode) {
    const adjustments = [];
    this._applyCosineLR(agent, episode, adjustments);
    this._adjustEpsilon(agent, metrics, episode, adjustments);
    this._adjustLR(agent, metrics, adjustments, episode);
    this._adjustNStep(agent, metrics, episode, adjustments);
    this._adjustGamma(agent, metrics, episode, adjustments);
    this._annealPER(agent, tdStats, episode, adjustments);
    this._adjustRewards(agent, rewardConfig, metrics, episode, adjustments);
    return adjustments;
  }

  _applyCosineLR(agent, episode, adjustments) {
    if (!this.cosineCycle) return;
    const cyclePos = (episode % this.cosineCycle) / this.cosineCycle;
    const base = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + Math.cos(Math.PI * cyclePos));
    const lr = clamp(base * this.lrScale, LR_MIN, LR_MAX);
    if (Math.abs(lr - agent.lr) / agent.lr > 0.05) {
      agent.setLearningRate(lr);
      adjustments.push({ type: 'lr', value: lr, reason: 'cosine' });
    }
  }

  _adjustLR(agent, metrics, adjustments, episode) {
    if (this.tdMeanHistory.length < 4) return;
    const last = this.tdMeanHistory.slice(-3);
    const increasing = last[0] < last[1] && last[1] < last[2];
    const ratio = metrics.tdLossMean ? metrics.tdLossStd / metrics.tdLossMean : 0;
    if ((increasing || ratio > 0.8) && this._canAdjust('lr-decay', episode)) {
      this.lrScale = Math.max(0.2, this.lrScale * 0.8);
      const lr = clamp(agent.lr * 0.8, LR_MIN, LR_MAX);
      agent.setLearningRate(lr);
      adjustments.push({ type: 'lr', value: lr, reason: 'reduce_plateau' });
      this.lastLRAdjustEpisode = episode;
    } else if (metrics.fruitSlope > 0 && this.lrScale < 1 && this._canAdjust('lr-recover', episode)) {
      this.lrScale = Math.min(1, this.lrScale * 1.05);
      const lr = clamp(agent.lr * 1.05, LR_MIN, LR_MAX);
      agent.setLearningRate(lr);
      adjustments.push({ type: 'lr', value: lr, reason: 'recover' });
    }
  }

  _adjustEpsilon(agent, metrics, episode, adjustments) {
    const improvement = this.metrics.getFruitImprovement(2000);
    const slope = metrics.fruitSlope ?? 0;
    if (slope <= 0 && improvement < 2 && this._canAdjust('epsilon-boost', episode)) {
      const newEnd = clamp(agent.epsEnd + 0.03, EPS_END_MIN, EPS_END_MAX);
      const newDecay = clamp(agent.epsDecay * 1.2, EPS_DECAY_MIN, EPS_DECAY_MAX);
      agent.setEpsilonSchedule({ end: newEnd, decay: newDecay });
      this.currentEpsEnd = newEnd;
      this.currentEpsDecay = newDecay;
      adjustments.push({ type: 'epsilon', end: newEnd, decay: newDecay, reason: 'stagnation' });
    }
    if (improvement < -5 && this._canAdjust('epsilon-regress', episode)) {
      const newEnd = clamp(agent.epsEnd + 0.02, EPS_END_MIN, EPS_END_MAX);
      agent.setEpsilonSchedule({ end: newEnd });
      this.currentEpsEnd = newEnd;
      this.lrScale = Math.max(0.2, this.lrScale * 0.9);
      adjustments.push({ type: 'epsilon', end: newEnd, reason: 'regression' });
    }
    if (this.currentEpsEnd > this.baseEpsEnd && improvement > 0 && slope > 0 && this._canAdjust('epsilon-revert', episode)) {
      const stepBack = Math.max(0.01, (this.currentEpsEnd - this.baseEpsEnd) * 0.5);
      const newEnd = clamp(this.currentEpsEnd - stepBack, EPS_END_MIN, this.baseEpsEnd);
      const newDecay = clamp(this.currentEpsDecay * 0.9, EPS_DECAY_MIN, this.baseEpsDecay);
      agent.setEpsilonSchedule({ end: newEnd, decay: newDecay });
      this.currentEpsEnd = newEnd;
      this.currentEpsDecay = newDecay;
      adjustments.push({ type: 'epsilon', end: newEnd, decay: newDecay, reason: 'revert' });
    }
  }

  _adjustNStep(agent, metrics, episode, adjustments) {
    if (metrics.timeToFruitAvg > 250 && agent.nStep < NSTEP_MAX && this._canAdjust('nstep-inc', episode)) {
      const newN = Math.min(NSTEP_MAX, agent.nStep + 1);
      agent.setNStep(newN);
      adjustments.push({ type: 'nStep', value: newN, reason: 'slow_fruit' });
    } else if (metrics.timeToFruitAvg < 120 && agent.nStep > NSTEP_MIN && this._canAdjust('nstep-dec', episode)) {
      const newN = Math.max(NSTEP_MIN, agent.nStep - 1);
      agent.setNStep(newN);
      adjustments.push({ type: 'nStep', value: newN, reason: 'fast_fruit' });
    }
  }

  _adjustGamma(agent, metrics, episode, adjustments) {
    if (!this.prevAvgLen) this.prevAvgLen = metrics.avgEpisodeLen100;
    const lenIncrease = metrics.avgEpisodeLen100 > this.prevAvgLen * 1.1;
    if (lenIncrease && agent.gamma < GAMMA_MAX && this._canAdjust('gamma-inc', episode)) {
      const newGamma = clamp(agent.gamma + 0.005, GAMMA_MIN, GAMMA_MAX);
      agent.setGamma(newGamma);
      adjustments.push({ type: 'gamma', value: newGamma, reason: 'longer_episodes' });
    }
    if (metrics.crashRateSelf > 0.5 && agent.gamma > this.baseGamma && this._canAdjust('gamma-dec', episode)) {
      const newGamma = clamp(agent.gamma - 0.005, GAMMA_MIN, GAMMA_MAX);
      agent.setGamma(newGamma);
      adjustments.push({ type: 'gamma', value: newGamma, reason: 'self_crash' });
    }
    this.prevAvgLen = metrics.avgEpisodeLen100;
  }

  _annealPER(agent, tdStats, episode, adjustments) {
    const beta = clamp(agent.priorityBeta + 0.0005, agent.priorityBeta, 1);
    if (beta !== agent.priorityBeta) {
      agent.setPriorityBeta(beta);
      adjustments.push({ type: 'per', beta, reason: 'anneal_beta' });
    }
    const ratio = tdStats?.p50 ? tdStats.p95 / (tdStats.p50 || 1) : 1;
    if (ratio > 1.6 && this._canAdjust('per-alpha-inc', episode)) {
      const alpha = clamp(agent.priorityAlpha + 0.05, 0.1, 1);
      agent.setPriorityAlpha(alpha);
      adjustments.push({ type: 'per', alpha, reason: 'heavy_tail' });
    } else if (ratio < 1.2 && this._canAdjust('per-alpha-dec', episode)) {
      const alpha = clamp(agent.priorityAlpha - 0.05, 0.1, 1);
      agent.setPriorityAlpha(alpha);
      adjustments.push({ type: 'per', alpha, reason: 'light_tail' });
    }
  }

  _adjustRewards(agent, rewardConfig, metrics, episode, adjustments) {
    if (metrics.loopHitRate > 0.01 && metrics.fruitSlope <= 0 && this._canAdjust('reward-loop', episode)) {
      rewardConfig.loopPenalty = clamp((rewardConfig.loopPenalty ?? 0.5) + 0.05, 0, 1);
      rewardConfig.compactWeight = 0;
      adjustments.push({ type: 'reward', key: 'loopPenalty', value: rewardConfig.loopPenalty });
    }
    if (metrics.revisitRate > 0.01 && this._canAdjust('reward-revisit', episode)) {
      rewardConfig.revisitPenalty = clamp((rewardConfig.revisitPenalty ?? 0.05) + 0.005, 0, 0.1);
      const newEnd = clamp(agent.epsEnd + 0.02, EPS_END_MIN, EPS_END_MAX);
      agent.setEpsilonSchedule({ end: newEnd });
      adjustments.push({ type: 'reward', key: 'revisitPenalty', value: rewardConfig.revisitPenalty });
    }
    if (metrics.crashRateSelf > 0.4 && this._canAdjust('reward-self', episode)) {
      rewardConfig.selfPenalty = clamp((rewardConfig.selfPenalty ?? 25.5) + 1, 0, 30);
      rewardConfig.turnPenalty = clamp((rewardConfig.turnPenalty ?? 0.001) - 0.0002, 0, 0.02);
      adjustments.push({ type: 'reward', key: 'selfPenalty', value: rewardConfig.selfPenalty });
    }
    if (metrics.timeToFruitAvg > 200 && metrics.loopHitRate < 0.005 && metrics.revisitRate < 0.005 && this._canAdjust('reward-fruit', episode)) {
      rewardConfig.approachBonus = clamp((rewardConfig.approachBonus ?? 0.03) + 0.005, 0, 0.1);
      rewardConfig.retreatPenalty = clamp((rewardConfig.retreatPenalty ?? 0.03) + 0.005, 0, 0.1);
      adjustments.push({ type: 'reward', key: 'approachBonus', value: rewardConfig.approachBonus });
    }
  }
}
