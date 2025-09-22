import fs from 'fs/promises';
import path from 'path';
import { ensureDir } from '../utils/fs_utils.js';

export class TrainingLogger {
  constructor({ logDir, consoleInterval = 200, logInterval = 500, diskGuard }) {
    this.logDir = logDir;
    this.consoleInterval = consoleInterval;
    this.logInterval = logInterval;
    this.diskGuard = diskGuard;
    this.lastConsoleEpisode = 0;
    this.lastLogEpisode = 0;
    this.logFile = path.join(this.logDir, 'training_log.jsonl');
  }

  async init() {
    await ensureDir(this.logDir);
  }

  maybeLogConsole({ episode, metrics, agent }) {
    if (episode - this.lastConsoleEpisode < this.consoleInterval) return;
    this.lastConsoleEpisode = episode;
    const msg = [
      `ep=${episode}`,
      `maFruit100=${metrics.maFruit100.toFixed(2)}`,
      `maFruit500=${metrics.maFruit500.toFixed(2)}`,
      `maReward100=${metrics.maReward100.toFixed(2)}`,
      `eps=${agent.epsilon?.toFixed(3)}`,
      `gamma=${agent.gamma?.toFixed(3)}`,
      `lr=${agent.lr?.toExponential(3)}`,
    ].join(' | ');
    console.log(`[TRAIN] ${msg}`);
  }

  async maybeLogJson({ episode, metrics, agent, adjustments = [], extra = {} }) {
    if (episode - this.lastLogEpisode < this.logInterval) return;
    this.lastLogEpisode = episode;
    const entry = {
      type: 'metrics',
      ts: new Date().toISOString(),
      episode,
      metrics,
      agent: {
        epsilon: agent.epsilon,
        gamma: agent.gamma,
        lr: agent.lr,
        batchSize: agent.batchSize,
        bufferSize: agent.buffer?.capacity,
        nStep: agent.nStep,
      },
      adjustments,
      ...extra,
    };
    await fs.appendFile(this.logFile, `${JSON.stringify(entry)}\n`);
    await this.diskGuard.rotateLogs('training_log.jsonl');
  }

  async logEval({ episode, fruitsAvg, fruitsPerEpisode }) {
    const entry = {
      type: 'eval',
      ts: new Date().toISOString(),
      episode,
      fruitsAvg,
      fruitsPerEpisode,
    };
    await fs.appendFile(this.logFile, `${JSON.stringify(entry)}\n`);
    await this.diskGuard.rotateLogs('training_log.jsonl');
  }
}
