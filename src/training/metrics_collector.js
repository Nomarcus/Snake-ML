import { RingBuffer, mean, std, linearRegressionSlope } from '../utils/ring_buffer.js';

export class MetricsCollector {
  constructor() {
    this.episodes = 0;
    this.steps = 0;
    this.window100 = [];
    this.window500 = [];
    this.lossBuffer = new RingBuffer(2000);
    this.maFruitHistory = new RingBuffer(200);
    this.lastEpisodeAt = Date.now();
    this.bestEval = null;
    this.bestEvalEpisode = 0;
  }

  recordEpisode(data) {
    this.episodes += 1;
    this.steps += data.length || 0;
    this.window500.push(data);
    if (this.window500.length > 500) this.window500.shift();
    this.window100.push(data);
    if (this.window100.length > 100) this.window100.shift();

    const ma100 = this.average(this.window100, 'fruits');
    this.maFruitHistory.push(ma100);
    this.lastEpisodeAt = Date.now();
  }

  recordLoss(loss) {
    if (Number.isFinite(loss)) {
      this.lossBuffer.push(loss);
    }
  }

  average(list, key) {
    if (!list.length) return 0;
    return list.reduce((sum, item) => sum + (+item[key] || 0), 0) / list.length;
  }

  sum(list, key) {
    if (!list.length) return 0;
    return list.reduce((sum, item) => sum + (+item[key] || 0), 0);
  }

  crashRate(list, type) {
    if (!list.length) return 0;
    const crashes = list.filter((item) => item.crash === type).length;
    return crashes / list.length;
  }

  getSummary() {
    const maFruit100 = this.average(this.window100, 'fruits');
    const maFruit500 = this.average(this.window500, 'fruits');
    const maReward100 = this.average(this.window100, 'reward');
    const maReward500 = this.average(this.window500, 'reward');
    const avgEpisodeLen100 = this.average(this.window100, 'length');
    const crashWall = this.crashRate(this.window500, 'wall');
    const crashSelf = this.crashRate(this.window500, 'self');
    const loopHits = this.sum(this.window500, 'loopHits');
    const totalSteps = this.sum(this.window500, 'length') || 1;
    const revisitPenalty = this.sum(this.window500, 'revisitPenalty');
    const timeToFruitSum = this.sum(this.window500, 'timeToFruitTotal');
    const timeToFruitCount = this.sum(this.window500, 'timeToFruitCount') || 0;

    const lossValues = this.lossBuffer.toArray();
    const tdLossMean = mean(lossValues);
    const tdLossStd = std(lossValues);

    const slope = linearRegressionSlope(this.maFruitHistory.toArray().slice(-30));

    return {
      episodes: this.episodes,
      steps: this.steps,
      maFruit100,
      maFruit500,
      maReward100,
      maReward500,
      avgEpisodeLen100,
      crashRateWall: crashWall,
      crashRateSelf: crashSelf,
      loopHitRate: loopHits / totalSteps,
      revisitRate: revisitPenalty / totalSteps,
      timeToFruitAvg: timeToFruitCount ? timeToFruitSum / timeToFruitCount : 0,
      tdLossMean,
      tdLossStd,
      fruitSlope: slope,
    };
  }

  getFruitSlope(windowEpisodes = 2000) {
    const history = this.maFruitHistory.toArray();
    if (!history.length) return 0;
    const slice = history.slice(-Math.min(history.length, Math.max(10, Math.floor(windowEpisodes / 50))));
    return linearRegressionSlope(slice);
  }

  getFruitImprovement(windowEpisodes = 2000) {
    const history = this.maFruitHistory.toArray();
    if (history.length < 2) return 0;
    const windowSize = Math.max(1, Math.min(history.length, Math.floor(windowEpisodes / 50)));
    const recent = history.slice(-windowSize);
    const past = history.slice(-2 * windowSize, -windowSize);
    if (!past.length) return 0;
    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const pastAvg = past.reduce((a, b) => a + b, 0) / past.length;
    return recentAvg - pastAvg;
  }

  updateBestEval(episode, fruits, meta) {
    if (!this.bestEval || fruits > this.bestEval.fruits) {
      this.bestEval = { episode, fruits, meta };
      this.bestEvalEpisode = episode;
      return true;
    }
    return false;
  }

  toEpisodeRecord({ reward, fruits, length, crash, loopHits, revisitPenalty, timeToFruitSamples }) {
    const count = timeToFruitSamples?.length || 0;
    const total = count ? timeToFruitSamples.reduce((a, b) => a + b, 0) : 0;
    return {
      reward: reward ?? 0,
      fruits: fruits ?? 0,
      length: length ?? 0,
      crash: crash ?? null,
      loopHits: loopHits ?? 0,
      revisitPenalty: revisitPenalty ?? 0,
      timeToFruitTotal: total,
      timeToFruitCount: count,
    };
  }
}
