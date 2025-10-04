const LOOP_FACTOR = 6;

function toFloatState(state) {
  if (state instanceof Float32Array) return state;
  if (ArrayBuffer.isView(state)) return Float32Array.from(state);
  if (Array.isArray(state)) return Float32Array.from(state);
  if (state && typeof state.length === 'number') return Float32Array.from(state);
  return new Float32Array();
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

    const action = selectAction(agent, state, train);
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
