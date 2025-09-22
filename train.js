#!/usr/bin/env node
import path from 'path';
import { runTraining } from './src/training/run_training.js';

function parseArgs(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith('--')) {
      result[key] = true;
    } else {
      result[key] = next;
      i += 1;
    }
  }
  return result;
}

function toNumber(value, fallback) {
  if (value === undefined || value === null) return fallback;
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const mode = args.mode === 'manual' ? 'manual' : 'auto';
  const envCount = toNumber(args.envCount, mode === 'auto' ? 12 : 1);
  const saveDir = path.resolve(args.saveDir ?? path.join('models', mode));

  const config = {
    mode,
    envCount,
    saveDir,
    checkpointIntervalEpisodes: toNumber(args['checkpoint-interval-episodes'], 5000),
    checkpointIntervalMinutes: toNumber(args['checkpoint-interval-minutes'], 60),
    retainCheckpoints: toNumber(args['retain-checkpoints'], 5),
    saveCapMB: toNumber(args['save-cap-mb'], 1024),
    logCapMB: toNumber(args['log-cap-mb'], 200),
    retainLogs: toNumber(args['retain-logs'], 3),
    logIntervalEpisodes: toNumber(args['log-interval-episodes'], 500),
    consoleIntervalEpisodes: toNumber(args['console-interval-episodes'], 200),
    minSaveCooldownMinutes: toNumber(args['min-save-cooldown-minutes'], 10),
    batchSize: toNumber(args.batchSize, undefined),
    bufferSize: toNumber(args.bufferSize, undefined),
    epsilon: {
      start: toNumber(args['eps-start'], undefined),
      end: toNumber(args['eps-end'], undefined),
      decay: toNumber(args['eps-decay'], undefined),
    },
    gamma: toNumber(args.gamma, undefined),
    learningRate: toNumber(args.lr, undefined),
    nStep: toNumber(args['n-step'], undefined),
    targetSync: toNumber(args['target-sync'], undefined),
    priorityAlpha: toNumber(args['priority-alpha'], undefined),
    priorityBeta: toNumber(args['priority-beta'], undefined),
    priorityBetaIncrement: toNumber(args['priority-beta-increment'], undefined),
    manualBoardSize: toNumber(args['board-size'], 20),
  };

  try {
    await runTraining(config);
  } catch (err) {
    console.error('[TRAIN] Fatal error', err);
    process.exitCode = 1;
  }
}

main();
