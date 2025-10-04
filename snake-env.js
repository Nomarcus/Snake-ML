import { bfsPath } from './src/path_helpers.js';

const LOOP_FACTOR = 6;

function toFloatState(state) {
  if (state instanceof Float32Array) return state;
  if (ArrayBuffer.isView(state)) return Float32Array.from(state);
  if (Array.isArray(state)) return Float32Array.from(state);
  if (state && typeof state.length === 'number') return Float32Array.from(state);
  return new Float32Array();
}

function normalizeDirection(dir) {
  const dx = Number.isFinite(dir?.x) ? Math.sign(dir.x) : 1;
  const dy = Number.isFinite(dir?.y) ? Math.sign(dir.y) : 0;
  if (dx === 0 && dy === 0) return { x: 1, y: 0 };
  if (![ -1, 0, 1 ].includes(dx) || ![ -1, 0, 1 ].includes(dy)) {
    return { x: 1, y: 0 };
  }
  return { x: dx, y: dy };
}

function boardSize(env) {
  const cols = Number.isFinite(env?.cols) ? env.cols | 0 : 0;
  const rows = Number.isFinite(env?.rows) ? env.rows | 0 : 0;
  if (cols && rows) return Math.max(cols, rows);
  return cols || rows || 0;
}

function nextStepAction(dir, head, path) {
  if (!Array.isArray(path) || path.length < 2) return null;
  if (!head || !Number.isFinite(head.x) || !Number.isFinite(head.y)) return null;
  const next = path[1];
  if (!next || !Number.isFinite(next.x) || !Number.isFinite(next.y)) return null;
  const dx = next.x - head.x;
  const dy = next.y - head.y;
  const forward = dir;
  const left = { x: -forward.y, y: forward.x };
  const right = { x: forward.y, y: -forward.x };
  if (dx === forward.x && dy === forward.y) return 0;
  if (dx === left.x && dy === left.y) return 1;
  if (dx === right.x && dy === right.y) return 2;
  return null;
}

function planTailFollowAction(env, { debug = false } = {}) {
  const snake = Array.isArray(env?.snake) ? env.snake : null;
  if (!snake || snake.length === 0) return null;
  const size = boardSize(env);
  if (!size) return null;
  const head = snake[0];
  if (!head) return null;
  const dir = normalizeDirection(env?.dir);

  const fruit = env?.fruit;
  const hasFruit = fruit && Number.isFinite(fruit.x) && Number.isFinite(fruit.y);
  if (hasFruit) {
    const pathToFruit = bfsPath(size, snake, fruit);
    const fruitAction = nextStepAction(dir, head, pathToFruit);
    if (fruitAction !== null) {
      return { type: 'fruit', action: fruitAction, path: pathToFruit };
    }
  }

  const tail = snake[snake.length - 1];
  if (!tail || !Number.isFinite(tail.x) || !Number.isFinite(tail.y)) return null;
  const pathToTail = bfsPath(size, snake, tail);
  const tailAction = nextStepAction(dir, head, pathToTail);
  if (tailAction !== null) {
    if (debug && typeof console !== 'undefined' && typeof console.log === 'function') {
      console.log('🐍 Following tail for safety');
    }
    return { type: 'tail', action: tailAction, path: pathToTail };
  }
  return null;
}

function selectAction(agent, state, train) {
  if (!agent) return 0;
  try {
    if (!train && typeof agent.greedyAction === 'function') {
      return agent.greedyAction(state);
    }
    if (typeof agent.act === 'function') {
      return agent.act(state);
    }
    if (typeof agent.greedyAction === 'function') {
      return agent.greedyAction(state);
    }
  } catch (err) {
    console.warn('[snake-env] Failed to select action', err);
  }
  return 0;
}

async function renderStep(env, before, options) {
  if (!options.render) return;
  if (typeof window === 'undefined') return;
  const enqueue = window.enqueueRenderFrame;
  const snapshot = window.snapshotEnv;
  if (typeof enqueue !== 'function' || typeof snapshot !== 'function') return;
  const frameMs = options.frameMs ?? window.playbackModes?.watch?.frameMs ?? 100;
  const queueTarget = options.queueTarget ?? window.playbackModes?.watch?.queueTarget ?? 60;
  try {
    const after = snapshot(env);
    enqueue(before ?? after, after, frameMs);
    const waitCapacity = window.waitForRenderCapacity;
    if (typeof waitCapacity === 'function') {
      await waitCapacity(queueTarget);
    }
  } catch (err) {
    console.warn('[snake-env] Failed to enqueue render frame', err);
  }
}

async function finishRender(options) {
  if (!options.render) return;
  if (typeof window === 'undefined') return;
  const waitIdle = window.waitForRenderIdle;
  if (typeof waitIdle === 'function') {
    try {
      await waitIdle();
    } catch (err) {
      console.warn('[snake-env] Failed waiting for render idle', err);
    }
  }
}

export async function runEpisode(env, agent, options = {}) {
  if (!env) throw new Error('[snake-env] Environment is required');
  if (!agent) throw new Error('[snake-env] Agent is required');
  const { train = true, render = false } = options;
  const tailFollowFallback = options.tailFollowFallback ?? !train;
  const debugTailFallback = options.debugTailFallback ?? false;
  const cols = env.cols ?? 20;
  const rows = env.rows ?? 20;
  const maxSteps = Number.isFinite(options.maxSteps)
    ? options.maxSteps
    : Math.max(50, cols * rows * LOOP_FACTOR);

  let state = toFloatState(env.reset());
  let totalReward = 0;
  let fruitEaten = 0;
  let steps = 0;
  let crashType = 'none';

  if (render && typeof window !== 'undefined' && typeof window.setImmediateState === 'function') {
    try {
      window.setImmediateState(env);
    } catch (err) {
      console.warn('[snake-env] Failed to set immediate render state', err);
    }
  }

  while (steps < maxSteps) {
    let before = null;
    if (render && typeof window !== 'undefined' && typeof window.snapshotEnv === 'function') {
      try {
        before = window.snapshotEnv(env);
      } catch (err) {
        console.warn('[snake-env] Failed to snapshot environment', err);
      }
    }

    let action = selectAction(agent, state, train);
    if (tailFollowFallback) {
      const fallback = planTailFollowAction(env, { debug: debugTailFallback });
      if (fallback?.type === 'tail' && fallback.action !== null) {
        action = fallback.action;
      }
    }
    const result = env.step(action) ?? {};
    const nextState = toFloatState(result.state);
    const reward = Number(result.reward ?? 0);
    const done = Boolean(result.done);
    const info = result.info ?? {};

    totalReward += reward;
    if (info.ateFruit) fruitEaten += 1;
    steps += 1;

    await renderStep(env, before, { ...options, render });

    state = nextState;
    if (done) {
      crashType = info.crash ?? 'done';
      break;
    }

    if (!train && typeof tf !== 'undefined' && typeof tf.nextFrame === 'function') {
      await tf.nextFrame();
    }
  }

  if (steps >= maxSteps && crashType === 'none') {
    crashType = 'loop';
  }

  await finishRender({ render });

  return {
    totalReward,
    fruitEaten,
    steps,
    crashType,
  };
}
