import { runEpisode, SnakeEnv } from './snake-env.js';

/**
 * Run a zero-epsilon evaluation phase that freezes training and
 * collects metrics for fruit, reward, steps, and crash types.
 */
export async function runFinalWatch(agent, envManager, episodes = 100) {
  console.log('🟢 Starting Final Watch Mode...');

  // Save state to restore after test
  const savedState = {
    epsilon: agent.epsilon,
    epsilonStart: agent.epsilonStart,
    epsilonEnd: agent.epsilonEnd,
    training: agent.training,
    autoActive: window.autoActive ?? false,
    aiAutoTuneEnabled: window.aiAutoTuneEnabled ?? false,
  };

  // Disable exploration and training
  agent.training = false;
  agent.epsilon = 0;
  window.autoActive = false;
  window.aiAutoTuneEnabled = false;

  // --- Get grid and reward config from current environment ---
  const cols = envManager?.cols ?? 15;
  const rows = envManager?.rows ?? 15;
  const rewardConfig = envManager?.rewardConfig ?? {};

  // --- Create isolated eval environment ---
  const evalEnv = new SnakeEnv(cols, rows, rewardConfig);

  const results = {
    fruit: [],
    reward: [],
    steps: [],
    wallCrashes: 0,
    selfCrashes: 0,
    loopCrashes: 0,
  };

  for (let i = 0; i < episodes; i++) {
    const { totalReward, fruitEaten, steps, crashType } =
      await runEpisode(evalEnv, agent, { train: false, render: true });

    results.fruit.push(fruitEaten);
    results.reward.push(totalReward);
    results.steps.push(steps);

    if (crashType === 'wall') results.wallCrashes++;
    else if (crashType === 'self') results.selfCrashes++;
    else if (crashType === 'loop') results.loopCrashes++;

    document.dispatchEvent(
      new CustomEvent('watchUpdate', { detail: { i, fruitEaten, totalReward } })
    );
  }

  const summary = {
    episodes,
    avgFruit: avg(results.fruit),
    avgReward: avg(results.reward),
    avgSteps: avg(results.steps),
    wallRate: results.wallCrashes / episodes,
    selfRate: results.selfCrashes / episodes,
    loopRate: results.loopCrashes / episodes,
    timestamp: new Date().toLocaleString(),
  };

  console.log('🏁 Final Watch Summary:', summary);

  renderFinalStats(summary, savedState);
  exportResults(summary);
  renderMiniSummary(summary);

  return summary;
}

// --- Helpers ---
function avg(a) {
  return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
}

// --- Floating overlay after test ---
function renderFinalStats(s, savedState) {
  const old = document.getElementById('finalWatchStats');
  if (old) old.remove();

  const div = document.createElement('div');
  div.id = 'finalWatchStats';
  div.className =
    'final-watch-stats absolute top-4 right-4 bg-black/70 text-white p-3 rounded-xl shadow-lg text-sm font-mono z-50';
  div.innerHTML = `
    <h3 class="font-bold text-base mb-2">🏁 Final Watch Stats</h3>
    <p>Episodes: ${s.episodes}</p>
    <p>Avg Fruit: ${s.avgFruit.toFixed(2)}</p>
    <p>Avg Reward: ${s.avgReward.toFixed(1)}</p>
    <p>Avg Steps: ${s.avgSteps.toFixed(0)}</p>
    <p>Wall crash rate: ${(s.wallRate * 100).toFixed(1)}%</p>
    <p>Self crash rate: ${(s.selfRate * 100).toFixed(1)}%</p>
    <p>Loop crash rate: ${(s.loopRate * 100).toFixed(1)}%</p>
    <div class="flex gap-2 mt-2">
      <button id="exportFinalWatch" class="btn accent">💾 Export JSON</button>
      <button id="resumeTraining" class="btn success">▶️ Resume Training</button>
    </div>
  `;
  document.body.appendChild(div);

  document
    .getElementById('exportFinalWatch')
    .addEventListener('click', () => exportResults(s));

  document
    .getElementById('resumeTraining')
    .addEventListener('click', () => restoreTraining(savedState));
}

// --- Add small summary box inside existing stats panel ---
function renderMiniSummary(s) {
  let panel = document.getElementById('miniFinalSummary');
  if (!panel) {
    const statsPanel = document.querySelector('.stats-panel, .ai-stats, .training-stats');
    if (!statsPanel) {
      console.warn('⚠️ No stats panel found for mini summary');
      return;
    }
    panel = document.createElement('div');
    panel.id = 'miniFinalSummary';
    panel.className = 'mini-final-summary border-t mt-2 pt-1 text-xs font-mono';
    statsPanel.appendChild(panel);
  }

  panel.innerHTML = `
    <div class="flex flex-col gap-0.5">
      <strong class="text-accent">🏁 Last Final Test (${s.timestamp})</strong>
      <span>Fruit: ${s.avgFruit.toFixed(2)}</span>
      <span>Reward: ${s.avgReward.toFixed(1)}</span>
      <span>Steps: ${s.avgSteps.toFixed(0)}</span>
      <span>Wall: ${(s.wallRate * 100).toFixed(1)}% | Self: ${(s.selfRate * 100).toFixed(1)}% | Loop: ${(s.loopRate * 100).toFixed(1)}%</span>
    </div>
  `;
}

// --- JSON export helper ---
function exportResults(data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'final-watch-results.json';
  a.click();
  URL.revokeObjectURL(url);
  console.log('💾 Exported final-watch-results.json');
}

// --- Restore training and continue ---
function restoreTraining(state) {
  const agent = window.agent;
  if (!agent) return;
  agent.epsilon = state.epsilon;
  agent.epsilonStart = state.epsilonStart;
  agent.epsilonEnd = state.epsilonEnd;
  agent.training = state.training;
  window.autoActive = state.autoActive;
  window.aiAutoTuneEnabled = state.aiAutoTuneEnabled;

  const uiDiv = document.getElementById('finalWatchStats');
  if (uiDiv) uiDiv.remove();

  console.log('🔁 Restored previous training state, resuming...');
  if (typeof window.startTraining === 'function') {
    window.startTraining();
  }
}
