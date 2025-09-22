import fs from 'fs/promises';
import path from 'path';
import { VecSnakeEnv } from '../env/vec_env.js';
import { SnakeEnv, REWARD_DEFAULTS, clampRewardConfig } from '../env/snake.js';
import { DQNAgent } from '../agents/dqn_agent.js';
import { DiskGuard } from '../disk_guard.js';
import { AutoScheduler } from './auto_scheduler.js';
import { CheckpointManager } from './checkpoint_manager.js';
import { TrainingLogger } from './logger.js';
import { MetricsCollector } from './metrics_collector.js';
import { ensureDir } from '../utils/fs_utils.js';

function createEpisodeContext(state) {
  return {
    state,
    totalReward: 0,
    fruits: 0,
    steps: 0,
    crash: null,
    loopHits: 0,
    revisitPenalty: 0,
    timeToFruit: [],
  };
}

async function runEvaluation(agent, boardSize, rewardConfig, episodes = 5) {
  const env = new SnakeEnv(boardSize, boardSize, rewardConfig);
  const perEpisode = [];
  for (let i = 0; i < episodes; i += 1) {
    let state = env.reset();
    let done = false;
    let steps = 0;
    let fruits = 0;
    const maxSteps = boardSize * boardSize * 6;
    while (!done && steps < maxSteps) {
      let action = agent.greedyAction(state);
      const result = env.step(action);
      state = result.state;
      done = result.done;
      if (result.info?.ateFruit) fruits += 1;
      steps += 1;
    }
    perEpisode.push(fruits);
  }
  const fruitsAvg = perEpisode.reduce((a, b) => a + b, 0) / perEpisode.length;
  return { fruitsAvg, perEpisode };
}

async function loadCheckpoint(agent, rewardConfig, env, dir) {
  const file = path.join(dir, 'checkpoint.json');
  const text = await fs.readFile(file, 'utf-8');
  const data = JSON.parse(text);
  await agent.importState(data.agent);
  Object.assign(rewardConfig, data.rewardConfig ?? {});
  env.setRewardConfig(rewardConfig);
  return data.meta ?? {};
}

export async function runTraining(options) {
  const {
    mode,
    envCount,
    saveDir,
    checkpointIntervalEpisodes,
    checkpointIntervalMinutes,
    retainCheckpoints,
    saveCapMB,
    logCapMB,
    retainLogs,
    logIntervalEpisodes,
    consoleIntervalEpisodes,
    minSaveCooldownMinutes,
    batchSize,
    bufferSize,
    epsilon,
    gamma,
    learningRate,
    nStep,
    targetSync,
    priorityAlpha,
    priorityBeta,
    priorityBetaIncrement,
    rewardOverrides = {},
    manualBoardSize = 20,
  } = options;

  await ensureDir(saveDir);
  const diskGuard = new DiskGuard({
    saveDir,
    retainCheckpoints,
    saveCapMB,
    logCapMB,
    retainLogs,
  });
  await diskGuard.init();

  const logDir = path.join(saveDir, 'logs');
  const logger = new TrainingLogger({
    logDir,
    consoleInterval: consoleIntervalEpisodes,
    logInterval: logIntervalEpisodes,
    diskGuard,
  });
  await logger.init();

  const checkpointManager = new CheckpointManager({ saveDir, diskGuard, retain: retainCheckpoints });
  await checkpointManager.init();

  const rewardConfig = clampRewardConfig({ ...REWARD_DEFAULTS, ...rewardOverrides });
  const initialBoard = mode === 'auto' ? 10 : manualBoardSize;

  const env = new VecSnakeEnv(envCount, { cols: initialBoard, rows: initialBoard, rewardConfig });
  let boardSize = initialBoard;

  const agent = new DQNAgent({
    stateDim: env.getStateDim(),
    actionDim: env.getActionDim(),
    envCount,
    gamma: gamma ?? 0.98,
    lr: learningRate ?? (mode === 'auto' ? 5e-4 : 3e-4),
    batchSize: batchSize ?? (mode === 'auto' ? 256 : 128),
    bufferSize: bufferSize ?? (mode === 'auto' ? 500000 : 50000),
    priorityAlpha: priorityAlpha ?? 0.6,
    priorityBeta: priorityBeta ?? 0.4,
    priorityBetaIncrement: priorityBetaIncrement ?? 0.000002,
    epsStart: epsilon?.start ?? 1,
    epsEnd: epsilon?.end ?? 0.12,
    epsDecay: epsilon?.decay ?? 80000,
    targetSync: targetSync ?? 2000,
    nStep: nStep ?? 3,
    warmupSteps: options.warmupSteps ?? 5000,
  });

  const metricsCollector = mode === 'auto' ? null : new MetricsCollector();
  const scheduler = mode === 'auto'
    ? new AutoScheduler({
        initialConfig: agent.getHyperparams(),
        rewardConfig,
        minSaveCooldownMinutes,
      })
    : null;

  const contexts = env.reset().map((state) => createEpisodeContext(state));
  let totalEpisodes = 0;
  let totalSteps = 0;
  let stopRequested = false;
  let lastPeriodicCheckpointEpisode = 0;
  let lastPeriodicCheckpointTime = Date.now();
  let lastAdjustments = [];

  const shutdown = async (signal, err) => {
    if (stopRequested) return;
    if (err) console.error('[TRAIN] Fatal error', err);
    console.log(`[TRAIN] Received ${signal}, stopping...`);
    stopRequested = true;
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
  process.on('uncaughtException', (err) => shutdown('uncaughtException', err));

  const metrics = () => (mode === 'auto' ? scheduler.getMetrics() : metricsCollector.getSummary());

  const flushFinalCheckpoint = async () => {
    const now = Date.now();
    const episodesSinceSave = totalEpisodes - lastPeriodicCheckpointEpisode;
    const minutesSinceSave = (now - lastPeriodicCheckpointTime) / 60000;
    if (
      episodesSinceSave < checkpointIntervalEpisodes ||
      minutesSinceSave < Math.max(checkpointIntervalMinutes, minSaveCooldownMinutes)
    ) {
      return;
    }
    const summary = metrics();
    const agentState = await agent.exportState();
    const meta = {
      mode,
      episode: totalEpisodes,
      steps: totalSteps,
      totalSteps,
      envCount,
      boardSize,
      metrics: summary,
      hyperparams: agent.getHyperparams(),
    };
    await checkpointManager.save({
      agentState,
      rewardConfig,
      meta,
      reason: 'final',
      isBest: false,
    });
  };

  while (!stopRequested) {
    const actions = contexts.map((ctx) => agent.act(ctx.state));
    const stepResult = env.step(actions);
    stepResult.nextStates.forEach((nextState, idx) => {
      const ctx = contexts[idx];
      const reward = stepResult.rewards[idx];
      const done = stepResult.dones[idx];
      const info = stepResult.infos[idx] ?? {};
      agent.recordTransition(idx, ctx.state, actions[idx], reward, nextState, done);
      ctx.state = nextState;
      ctx.totalReward += reward;
      ctx.steps += 1;
      ctx.loopHits += info.loopPenaltyApplied ? 1 : 0;
      ctx.revisitPenalty += info.revisitPenalty ?? 0;
      if (info.timeToFruit) ctx.timeToFruit.push(info.timeToFruit);
      if (info.ateFruit) ctx.fruits += 1;
      if (info.crash) ctx.crash = info.crash;
      totalSteps += 1;
    });

    agent.setGlobalStep(totalSteps);
    const learnResult = await agent.learn();
    if (learnResult) {
      if (mode === 'auto') scheduler.metrics.recordLoss(learnResult.loss);
      else metricsCollector.recordLoss(learnResult.loss);
    }

    for (let i = 0; i < contexts.length; i += 1) {
      if (!stepResult.dones[i]) continue;
      const ctx = contexts[i];
      const record = {
        reward: ctx.totalReward,
        fruits: ctx.fruits,
        length: ctx.steps,
        crash: ctx.crash,
        loopHits: ctx.loopHits,
        revisitPenalty: ctx.revisitPenalty,
        timeToFruitSamples: ctx.timeToFruit,
      };
      if (mode === 'auto') scheduler.recordEpisode(record);
      else metricsCollector.recordEpisode(metricsCollector.toEpisodeRecord(record));
      const newState = env.resetEnv(i);
      contexts[i] = createEpisodeContext(newState);
      totalEpisodes += 1;

      const summary = metrics();
      if (mode === 'auto') {
        const tdStats = agent.getTdErrorStats();
        scheduler.observe(summary, tdStats, totalEpisodes);
        lastAdjustments = scheduler.tuneHyperparams(agent, rewardConfig, summary, tdStats, totalEpisodes) || [];
        if (lastAdjustments.some((adj) => adj.type === 'reward')) {
          env.setRewardConfig(rewardConfig);
        }
        const newBoard = scheduler.maybeAdvanceBoard(totalEpisodes);
        if (newBoard) {
          boardSize = newBoard;
          env.setBoardSize(boardSize, boardSize);
          const resetStates = env.reset();
          resetStates.forEach((state, idx) => {
            contexts[idx] = createEpisodeContext(state);
          });
          agent.pruneReplay(0.5);
          const now = Date.now();
          if (scheduler.shouldCheckpoint(now)) {
            const agentState = await agent.exportState();
            const meta = {
              mode,
              episode: totalEpisodes,
              steps: totalSteps,
              totalSteps,
              envCount,
              boardSize,
              metrics: summary,
              hyperparams: agent.getHyperparams(),
              adjustments: lastAdjustments,
            };
            await checkpointManager.save({
              agentState,
              rewardConfig,
              meta,
              reason: 'board_change',
              isBest: false,
            });
            lastPeriodicCheckpointEpisode = totalEpisodes;
            lastPeriodicCheckpointTime = now;
          }
        }
        if (scheduler.maybeEval(totalEpisodes)) {
          scheduler.registerEval(totalEpisodes);
          const evalResult = await runEvaluation(agent, boardSize, rewardConfig, options.evalEpisodes ?? 5);
          await logger.logEval({ episode: totalEpisodes, fruitsAvg: evalResult.fruitsAvg, fruitsPerEpisode: evalResult.perEpisode });
          const improved = scheduler.updateBestEval(totalEpisodes, evalResult.fruitsAvg, { boardSize });
          const nowEval = Date.now();
          if (improved && scheduler.shouldCheckpoint(nowEval)) {
            const agentState = await agent.exportState();
            const meta = {
              mode,
              episode: totalEpisodes,
              steps: totalSteps,
              totalSteps,
              envCount,
              boardSize,
              metrics: summary,
              hyperparams: agent.getHyperparams(),
              eval: evalResult,
            };
            await checkpointManager.save({
              agentState,
              rewardConfig,
              meta,
              reason: 'best_eval',
              isBest: true,
            });
            lastPeriodicCheckpointEpisode = totalEpisodes;
            lastPeriodicCheckpointTime = nowEval;
          }
          const metricsAfterEval = metrics();
          if (scheduler.needsRollback(metricsAfterEval, totalEpisodes)) {
            const bestDir = path.join(saveDir, 'best');
            try {
              const meta = await loadCheckpoint(agent, rewardConfig, env, bestDir);
              if (meta.boardSize) {
                boardSize = meta.boardSize;
                env.setBoardSize(boardSize, boardSize);
              }
              const resetStates = env.reset();
              resetStates.forEach((state, idx) => {
                contexts[idx] = createEpisodeContext(state);
              });
              agent.flushTransitions?.();
              console.warn('[TRAIN] Rollback to best checkpoint due to regression');
            } catch (err) {
              console.error('[TRAIN] Failed to rollback', err);
            }
          }
        }
      }

      logger.maybeLogConsole({ episode: totalEpisodes, metrics: summary, agent });
      await logger.maybeLogJson({ episode: totalEpisodes, metrics: summary, agent, adjustments: lastAdjustments });

      const now = Date.now();
      const episodesSinceSave = totalEpisodes - lastPeriodicCheckpointEpisode;
      const minutesSinceSave = (now - lastPeriodicCheckpointTime) / 60000;
      if (
        totalEpisodes > 0 &&
        episodesSinceSave >= checkpointIntervalEpisodes &&
        minutesSinceSave >= checkpointIntervalMinutes
      ) {
        const agentState = await agent.exportState();
        const meta = {
          mode,
          episode: totalEpisodes,
          steps: totalSteps,
          totalSteps,
          envCount,
          boardSize,
          metrics: summary,
          hyperparams: agent.getHyperparams(),
          adjustments: lastAdjustments,
        };
        await checkpointManager.save({ agentState, rewardConfig, meta, reason: 'periodic' });
        lastPeriodicCheckpointEpisode = totalEpisodes;
        lastPeriodicCheckpointTime = now;
      }
    }
  }

  try {
    agent.flushTransitions?.();
    logger.lastLogEpisode = totalEpisodes - logIntervalEpisodes;
    await logger.maybeLogJson({
      episode: totalEpisodes,
      metrics: metrics(),
      agent,
      adjustments: lastAdjustments,
      extra: { final: true },
    });
  } catch (err) {
    console.error('[TRAIN] Failed to log final metrics', err);
  }

  await flushFinalCheckpoint();

  process.removeAllListeners('SIGINT');
  process.removeAllListeners('SIGTERM');
  process.removeAllListeners('uncaughtException');
}
