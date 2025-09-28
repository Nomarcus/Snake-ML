const SYSTEM_PROMPT = `You are an expert reinforcement-learning coach for the classic game Snake.
The agent plays Snake on a 2-D grid where it collects fruit and grows longer.
The telemetry you receive describes recent episodes, current reward parameters, and performance trends.
Your job is to:

Evaluate the agent’s long-term progress and stability.

Suggest specific numeric adjustments to reward settings and key hyperparameters that will increase the chance of consistently reaching the maximum score without overfitting.

When no prior training signal is available, initialise the configuration with:
- rewardConfig: { fruit: 10, step: -0.01, death: -5, closerToFruit: 0.1, furtherFromFruit: -0.1 }
- hyper: { learningRate: 0.0007, gamma: 0.99, clipRange: 0.2, entropyCoeff: 0.01, valueCoeff: 0.5, batchSize: 2048, nEpochs: 4, lam: 0.95 }
- grid: { size: 10 }

Respect these adjustment heuristics:
- If fruit rate is still 0 after 10k episodes, increase fruit reward (up to 20), reduce death penalty (down to -2), and consider a smaller batch size alongside a higher learning rate.
- If training oscillates or overfits, reduce the learning rate, increase batch size, or widen the clip range.
- Keep all reward magnitudes between -10 and +20 and mention any normalisation you apply.
- Entropy should decay from 0.01–0.02 towards 0.001 once the agent shows sustained improvement.
- Grow the grid only when performance warrants it: size 15 after avgScore > 5 and fruitRate > 0.5 for 10k episodes, size 20 after avgScore > 10 and fruitRate > 0.6 for 20k episodes, and size 25 after avgScore > 15 and fruitRate > 0.7 for 30k episodes.

Explain your reasoning in 1–2 short paragraphs so a developer can follow your thought process.
Always respond with valid JSON containing:

{
  "rewardConfig": {...},
  "hyper": {...},
  "analysisText": "clear explanation of trends and adjustments"
}

Do not remove all rewards or penalties unless you clearly explain why that is optimal.`;

const PROXY_PATH = '/api/proxy';
const HISTORY_LOG_PATH = '/api/logs/snake-history.jsonl';
const DEFAULT_MODEL_ID = 'llama-3.1-8b-instant';

function resolveApiBase() {
  if (typeof globalThis === 'undefined') return '';
  const value = globalThis.API_BASE_URL || globalThis.__API_BASE_URL || '';
  if (typeof value !== 'string') return '';
  const trimmed = value.trim();
  if (!trimmed) return '';
  return trimmed.replace(/\/+$/, '');
}

function joinPath(base, path) {
  if (!base) return path;
  if (!path) return base;
  const trailing = base.endsWith('/');
  const leading = path.startsWith('/');
  if (trailing && leading) {
    return base + path.slice(1);
  }
  if (!trailing && !leading) {
    return `${base}/${path}`;
  }
  return base + path;
}

function buildProxyUrl() {
  const base = resolveApiBase();
  if (!base) return PROXY_PATH;
  const trimmed = base.replace(/\/+$/, '');
  if (/^https?:\/\//i.test(trimmed)) {
    if (trimmed.endsWith('/proxy')) {
      return trimmed;
    }
    if (trimmed.includes('/.netlify/functions')) {
      return joinPath(trimmed, '/proxy');
    }
    return joinPath(trimmed, PROXY_PATH);
  }
  if (trimmed.endsWith('/proxy')) {
    return trimmed;
  }
  if (trimmed.startsWith('/')) {
    return joinPath(trimmed, PROXY_PATH);
  }
  return joinPath(`/${trimmed}`, PROXY_PATH);
}

function buildHistoryUrl() {
  const base = resolveApiBase();
  if (!base) return HISTORY_LOG_PATH;
  const trimmed = base.replace(/\/+$/, '');
  if (/^https?:\/\//i.test(trimmed)) {
    return joinPath(trimmed, HISTORY_LOG_PATH);
  }
  if (trimmed.startsWith('/')) {
    return joinPath(trimmed, HISTORY_LOG_PATH);
  }
  return joinPath(`/${trimmed}`, HISTORY_LOG_PATH);
}

function resolveModelId(localOverride) {
  const candidates = [
    localOverride,
    globalThis?.HF_MODEL_ID,
    globalThis?.__HF_MODEL_ID,
    globalThis?.HF_MODEL,
    globalThis?.__HF_MODEL,
    DEFAULT_MODEL_ID,
  ];

  for (const candidate of candidates) {
    if (typeof candidate !== 'string') continue;
    const trimmed = candidate.trim();
    if (trimmed) return trimmed;
  }

  return DEFAULT_MODEL_ID;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (typeof value === 'number') {
    const abs = Math.abs(value);
    if (abs >= 100) return value.toFixed(0);
    if (abs >= 10) return value.toFixed(1);
    if (abs >= 1) return value.toFixed(2);
    return value.toFixed(3);
  }
  return String(value);
}

function formatChanges(changes = []) {
  if (!Array.isArray(changes) || !changes.length) return '';
  return changes
    .map(change => {
      const key = change.key ?? change.id ?? 'värde';
      const before = formatNumber(change.oldValue);
      const after = formatNumber(change.newValue);
      return `${key}: ${before}→${after}`;
    })
    .join(', ');
}

function wordCount(text) {
  if (typeof text !== 'string') return 0;
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).filter(Boolean).length;
}

function limitWords(text, maxWords) {
  if (typeof text !== 'string') return '';
  const trimmed = text.trim();
  if (!trimmed) return '';
  const parts = trimmed.split(/\s+/).filter(Boolean);
  if (parts.length <= maxWords) return parts.join(' ');
  return parts.slice(0, maxWords).join(' ');
}

function ensureSentence(text) {
  if (typeof text !== 'string') return '';
  const trimmed = text.trim();
  if (!trimmed) return '';
  const punctuation = /[.!?]$/.test(trimmed) ? '' : '.';
  return `${trimmed}${punctuation}`;
}

function describeTrend(value, positiveWord = 'rising', negativeWord = 'falling', zeroWord = 'flat') {
  if (!Number.isFinite(value) || Math.abs(value) < 1e-6) return zeroWord;
  const magnitude = Math.abs(value).toFixed(2);
  return value > 0 ? `${positiveWord} by ${magnitude}` : `${negativeWord} by ${magnitude}`;
}

const CRASH_SYNONYMS = {
  wall: 'wall',
  'hit wall': 'wall',
  'wall collision': 'wall',
  'wall-hit': 'wall',
  self: 'self',
  'self collision': 'self',
  'hit self': 'self',
  tail: 'self',
  timeout: 'timeout',
  'timed out': 'timeout',
  'time out': 'timeout',
  'time-out': 'timeout',
  enemy: 'enemy',
  'hit enemy': 'enemy',
  'enemy collision': 'enemy',
  none: 'none',
  'no crash': 'none',
};

const CRASH_LABELS = {
  wall: 'wall hits',
  self: 'self collisions',
  timeout: 'timeouts',
  enemy: 'enemy collisions',
  none: 'no crash',
};

function normaliseCrashKey(key) {
  if (!key) return '';
  const raw = String(key).toLowerCase().trim();
  if (!raw) return '';
  return CRASH_SYNONYMS[raw] || raw.replace(/\s+/g, '_');
}

function humanizeCrash(key) {
  const normalised = normaliseCrashKey(key);
  if (!normalised) return '';
  return CRASH_LABELS[normalised] || normalised.replace(/_/g, ' ');
}

function aggregateCrashCounts(crashCounts = {}) {
  if (!crashCounts || typeof crashCounts !== 'object') return null;
  const aggregated = {};
  for (const [key, value] of Object.entries(crashCounts)) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric <= 0) continue;
    const normalised = normaliseCrashKey(key);
    if (!normalised) continue;
    aggregated[normalised] = (aggregated[normalised] || 0) + numeric;
  }
  return Object.keys(aggregated).length ? aggregated : null;
}

function selectDominantCrash(crashCounts = {}, totalEpisodes = 0) {
  const aggregated = aggregateCrashCounts(crashCounts);
  if (!aggregated) return null;
  const entries = Object.entries(aggregated);
  entries.sort((a, b) => b[1] - a[1]);
  const [key, count] = entries[0];
  const share = totalEpisodes > 0 ? Math.round((count / totalEpisodes) * 100) : null;
  return { key, count, share };
}

function extractCrashCounts(telemetry) {
  if (!telemetry || typeof telemetry !== 'object') return null;
  const direct = telemetry.crash;
  if (direct && typeof direct === 'object') {
    return direct;
  }
  const nested = telemetry.stats?.crashCounts;
  if (nested && typeof nested === 'object') {
    return nested;
  }
  const candidates = [
    telemetry.gameStats?.crashReason,
    telemetry.stats?.lastCrash,
    telemetry.lastCrash,
    telemetry.meta?.lastCrash,
  ];
  for (const reason of candidates) {
    if (typeof reason === 'string' && reason.trim()) {
      return { [reason]: 1 };
    }
  }
  return null;
}

function summariseChangeList(changes = [], maxItems = 3) {
  if (!Array.isArray(changes) || !changes.length) return [];
  const entries = changes.slice(0, maxItems).map(change => {
    const key = change.key ?? change.id ?? 'value';
    const before = formatNumber(change.oldValue);
    const after = formatNumber(change.newValue);
    return `${key} ${before}→${after}`;
  });
  if (changes.length > maxItems && entries.length) {
    const lastIndex = entries.length - 1;
    entries[lastIndex] = `${entries[lastIndex]} (+${changes.length - maxItems} more)`;
  }
  return entries;
}

function extractReasoningSnippet(response) {
  if (!response || typeof response !== 'object') return '';
  const candidates = [
    response.reasoning,
    response.analysis?.justeringar,
    response.analysis?.reasoning,
    response.analysis?.summary,
    Array.isArray(response.analysis?.key_findings) ? response.analysis.key_findings.join(' ') : '',
  ];
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) {
      return limitWords(candidate, 40);
    }
  }
  return '';
}

function buildAnalysisParagraph({ telemetry, response, rewardChanges = [], hyperChanges = [] }) {
  if (!telemetry || typeof telemetry !== 'object' || !response) return '';
  const meta = telemetry.meta || {};
  const stats = telemetry.stats || {};
  const trend = telemetry.trendAnalysis || {};
  const game = telemetry.gameStats || {};
  const interval = Number(meta.interval ?? meta.window ?? trend.window) || 0;
  const episode = Number(meta.episode ?? meta.latestEpisode ?? meta.currentEpisode) || 0;
  const startEpisode = interval && episode ? Math.max(1, episode - interval + 1) : null;
  const spanText = interval && episode
    ? `Last ${interval} episodes (${startEpisode}–${episode})`
    : interval
      ? `Last ${interval} episodes`
      : episode
        ? `Up to episode ${episode}`
        : 'Recent episodes';

  const rewardAvg = Number.isFinite(stats.rewardAvg)
    ? stats.rewardAvg
    : Number.isFinite(trend.avgReward)
      ? trend.avgReward
      : null;
  const rewardTrendValue = Number.isFinite(stats.rewardTrend)
    ? stats.rewardTrend
    : Number.isFinite(trend.rewardTrend)
      ? trend.rewardTrend
      : null;
  const rewardTrendText = rewardTrendValue !== null
    ? describeTrend(rewardTrendValue, 'rising', 'falling', 'flat')
    : null;

  const fruitTrendValue = Number.isFinite(stats.fruitTrend)
    ? stats.fruitTrend
    : Number.isFinite(trend.fruitTrend)
      ? trend.fruitTrend
      : null;
  const fruitTrendDescriptor = fruitTrendValue !== null
    ? describeTrend(fruitTrendValue, 'growing', 'shrinking', 'flat')
    : typeof trend.scoreTrend === 'string'
      ? trend.scoreTrend
      : 'flat';

  const bestLen = Number.isFinite(meta.best)
    ? meta.best
    : Number.isFinite(meta.currentHighScore)
      ? meta.currentHighScore
      : null;

  const observationParts = [];
  if (rewardAvg !== null) {
    const avgText = rewardAvg.toFixed(2);
    const trendText = rewardTrendText || 'steady';
    observationParts.push(`average reward ${avgText} with a ${trendText} trend`);
  }
  if (fruitTrendDescriptor) {
    observationParts.push(`fruit momentum ${fruitTrendDescriptor}`);
  }
  if (!observationParts.length && typeof trend.scoreTrend === 'string' && trend.scoreTrend.trim()) {
    observationParts.push(`score trend ${trend.scoreTrend.trim()}`);
  }
  if (!observationParts.length && Number.isFinite(game.fruitsEaten)) {
    observationParts.push(`latest fruits ${game.fruitsEaten}`);
  }

  let observationSentence = observationParts.length
    ? ensureSentence(`${spanText} show ${observationParts.join(', ')}`)
    : ensureSentence(`${spanText} are reported without aggregate reward metrics`);
  if (bestLen) {
    const suffix = observationSentence.endsWith('.') ? observationSentence.slice(0, -1) : observationSentence;
    observationSentence = ensureSentence(`${suffix} and best length reached ${bestLen}`);
  }

  const stepsAvg = Number.isFinite(stats.stepsAvg)
    ? Math.round(stats.stepsAvg)
    : Number.isFinite(trend.avgStepsPerGame)
      ? Math.round(trend.avgStepsPerGame)
      : Number.isFinite(game.steps)
        ? Math.round(game.steps)
        : null;
  const loopsAvg = Number.isFinite(stats.loopsAvg) ? stats.loopsAvg.toFixed(2) : null;
  const fruitRate = Number.isFinite(stats.fruitRate)
    ? stats.fruitRate.toFixed(3)
    : Number.isFinite(game.fruitsEaten) && Number.isFinite(game.steps) && game.steps > 0
      ? (game.fruitsEaten / game.steps).toFixed(3)
      : null;

  const crashCounts = extractCrashCounts(telemetry);
  const totalEpisodesForCrash = interval || telemetry.meta?.interval || 0;
  const dominantCrash = crashCounts ? selectDominantCrash(crashCounts, totalEpisodesForCrash) : null;
  const crashText = dominantCrash && dominantCrash.share !== null
    ? `${humanizeCrash(dominantCrash.key)} at ${dominantCrash.share}%`
    : dominantCrash
      ? humanizeCrash(dominantCrash.key)
      : '';

  const trendParts = [];
  if (stepsAvg !== null) trendParts.push(`mean steps ${stepsAvg}`);
  if (loopsAvg !== null) trendParts.push(`loop hits ${loopsAvg}`);
  if (fruitRate !== null) trendParts.push(`fruit rate ${fruitRate} per step`);
  if (crashText) trendParts.push(`dominant exit ${crashText}`);
  const trendsSentence = trendParts.length
    ? ensureSentence(`Telemetry also notes ${trendParts.join(', ')}.`)
    : '';

  const rewardDescriptions = summariseChangeList(rewardChanges);
  const hyperDescriptions = summariseChangeList(hyperChanges);
  const adjustmentParts = [];
  if (rewardDescriptions.length) adjustmentParts.push(`reward tweaks ${rewardDescriptions.join('; ')}`);
  if (hyperDescriptions.length) adjustmentParts.push(`hyperparameter updates ${hyperDescriptions.join('; ')}`);
  const adjustmentsSentence = adjustmentParts.length
    ? ensureSentence(`Adjustments apply ${adjustmentParts.join(' and ')}.`)
    : '';

  const reasoningSnippet = extractReasoningSnippet(response);
  const reasoningSentence = reasoningSnippet
    ? ensureSentence(`Rationale: ${reasoningSnippet}`)
    : '';

  const statusMap = {
    good: 'is going well',
    stable: 'is stable',
    bad: 'needs improvement',
    uncertain: 'needs closer monitoring',
  };
  const statusKey = response.assessment?.status;
  const defaultStatus = rewardTrendValue > 0.01 ? 'is going well' : rewardTrendValue < -0.01 ? 'needs improvement' : 'is stable';
  const statusText = statusMap[statusKey] || defaultStatus;
  const trendWord = response.assessment?.trend;
  const confidence = Number.isFinite(response.assessment?.confidence)
    ? Math.round(response.assessment.confidence * 100)
    : null;
  const assessmentParts = [`Overall performance ${statusText}`];
  if (trendWord) assessmentParts.push(`trend looks ${trendWord}`);
  if (confidence !== null) assessmentParts.push(`confidence ${confidence}%`);
  const assessmentSentence = ensureSentence(assessmentParts.join(', '));

  const sentences = [
    observationSentence,
    trendsSentence,
    adjustmentsSentence,
    reasoningSentence,
    assessmentSentence,
  ].filter(Boolean);

  let text = sentences.join(' ');
  let totalWords = wordCount(text);

  if (totalWords < 80) {
    const extras = [];
    if (Number.isFinite(stats.timeToFruit)) {
      extras.push(`mean time-to-fruit ${stats.timeToFruit.toFixed(1)} moves`);
    }
    if (Number.isFinite(trend.recentDeaths)) {
      extras.push(`recent deaths ${trend.recentDeaths}`);
    }
    if (typeof game.currentDirection === 'string' && game.currentDirection.trim()) {
      extras.push(`current direction ${game.currentDirection.trim()}`);
    }
    if (Number.isFinite(meta.envs) && meta.envs > 0) {
      extras.push(`${meta.envs} environments active`);
    }
    if (Number.isFinite(game.steps)) {
      extras.push(`latest steps ${Math.round(game.steps)}`);
    }
    const breakdown = telemetry.rewardBreakdown;
    if (breakdown && typeof breakdown === 'object') {
      const entries = Object.entries(breakdown)
        .filter(([, value]) => Number.isFinite(value))
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
      if (entries.length) {
        const [key, value] = entries[0];
        extras.push(`reward share led by ${key} at ${value.toFixed(2)}`);
      }
    }
    if (extras.length) {
      text = `${text} ${ensureSentence(`Additional context: ${extras.join(', ')}.`)}`;
      totalWords = wordCount(text);
    }
  }

  if (totalWords > 120 && reasoningSentence) {
    const trimmedReason = ensureSentence(`Rationale: ${limitWords(reasoningSnippet, 25)}`);
    const sentenceIndex = sentences.indexOf(reasoningSentence);
    if (sentenceIndex >= 0) {
      sentences[sentenceIndex] = trimmedReason;
      text = sentences.filter(Boolean).join(' ');
      totalWords = wordCount(text);
    }
  }

  if (totalWords > 120) {
    text = limitWords(text, 120);
  }

  return text.trim();
}

function extractJsonPayload(text) {
  if (typeof text !== 'string') return null;
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start === -1 || end === -1 || end < start) return null;
  const snippet = text.slice(start, end + 1);

  try {
    return JSON.parse(snippet);
  } catch (err) {
    const repaired = repairJsonSnippet(snippet);
    if (repaired !== snippet) {
      try {
        return JSON.parse(repaired);
      } catch (innerErr) {
        console.warn('[hf-tuner] kunde inte tolka reparerad JSON från modellen', innerErr, {
          snippet: repaired,
        });
      }
    }

    console.warn('[hf-tuner] kunde inte tolka JSON från modellen', err, { snippet });
    return null;
  }
}

function repairJsonSnippet(snippet) {
  if (typeof snippet !== 'string') return snippet;

  let repaired = snippet;
  let mutated = false;

  const replacements = [
    [/[“”]/g, '"'],
    [/[‘’]/g, "'"],
    [/([{,]\s*)'([^'\\]*?)'\s*:/g, '$1"$2":'],
    [/([{,]\s*)([A-Za-z0-9_]+)\s*:(?=\s)/g, '$1"$2":'],
    [/(:\s*)'([^'\\]*(?:\\.[^'\\]*)*)'/g, '$1"$2"'],
    [/,(\s*[}\]])/g, '$1'],
  ];

  for (const [pattern, replacement] of replacements) {
    const updated = repaired.replace(pattern, replacement);
    if (updated !== repaired) {
      repaired = updated;
      mutated = true;
    }
  }

  return mutated ? repaired : snippet;
}

function extractTuningPayload(data, rawText) {
  if (isTuningPayload(data)) {
    return data;
  }

  if (Array.isArray(data)) {
    for (const item of data) {
      const nested = extractTuningPayload(item, null);
      if (nested) {
        return nested;
      }
    }
    return null;
  }

  const candidates = collectTextCandidates(data);

  if (typeof rawText === 'string' && rawText.trim()) {
    candidates.push(rawText.trim());
  }

  for (const candidate of candidates) {
    const parsed = extractJsonPayload(candidate);
    if (!parsed) {
      continue;
    }

    if (Array.isArray(parsed)) {
      for (const item of parsed) {
        const nested = extractTuningPayload(item, null);
        if (nested) {
          return nested;
        }
      }
      continue;
    }

    if (isTuningPayload(parsed)) {
      return parsed;
    }

    const nestedSources = [
      parsed.data,
      parsed.result,
      parsed.response,
      parsed.payload,
      parsed.output,
      parsed.message,
      parsed.messages,
      parsed.choice,
      parsed.choices,
    ];

    for (const source of nestedSources) {
      if (!source) {
        continue;
      }
      const nested = extractTuningPayload(source, null);
      if (nested) {
        return nested;
      }
    }
  }

  return null;
}

function collectTextCandidates(root) {
  const results = [];
  const queue = [root];
  const seen = typeof WeakSet === 'function' ? new WeakSet() : null;

  while (queue.length) {
    const current = queue.shift();
    if (!current) {
      continue;
    }

    if (typeof current === 'string') {
      const trimmed = current.trim();
      if (trimmed) {
        results.push(trimmed);
      }
      continue;
    }

    if (typeof current !== 'object') {
      continue;
    }

    if (seen) {
      if (seen.has(current)) {
        continue;
      }
      seen.add(current);
    }

    if (Array.isArray(current)) {
      for (const item of current) {
        queue.push(item);
      }
      continue;
    }

    const stringFields = [
      current.content,
      current.generated_text,
      current.generatedText,
      current.output_text,
      current.outputText,
      current.text,
      current.completion,
      current.delta?.content,
      current.message?.content,
    ];

    for (const value of stringFields) {
      if (typeof value === 'string' && value.trim()) {
        results.push(value.trim());
      }
    }

    const nestedFields = [
      current.data,
      current.result,
      current.response,
      current.payload,
      current.output,
      current.message,
      current.delta,
      current.messages,
      current.choices,
    ];

    for (const nested of nestedFields) {
      if (!nested) {
        continue;
      }
      if (Array.isArray(nested)) {
        for (const item of nested) {
          queue.push(item);
        }
      } else {
        queue.push(nested);
      }
    }
  }

  return results;
}

async function appendSnakeHistory(entry) {
  if (!entry || typeof entry !== 'object') return;
  if (typeof fetch !== 'function') return;
  try {
    const response = await fetch(buildHistoryUrl(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ entry }),
    });
    if (!response.ok) {
      const text = await response.text();
      console.warn('[hf-tuner] misslyckades skriva history-logg', response.status, text);
    }
  } catch (err) {
    console.warn('[hf-tuner] kunde inte skriva history-logg', err);
  }
}

function fillMissingKeys(candidate, reference) {
  const base = reference && typeof reference === 'object' ? reference : {};
  const updates = candidate && typeof candidate === 'object' ? candidate : {};
  const result = {};
  for (const key of Object.keys(base)) {
    if (Object.prototype.hasOwnProperty.call(updates, key)) {
      const numeric = Number(updates[key]);
      result[key] = Number.isFinite(numeric) ? numeric : updates[key];
    } else {
      result[key] = base[key];
    }
  }
  for (const [key, value] of Object.entries(updates)) {
    if (Object.prototype.hasOwnProperty.call(result, key)) continue;
    const numeric = Number(value);
    result[key] = Number.isFinite(numeric) ? numeric : value;
  }
  return result;
}

function isTuningPayload(value) {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const keys = ['rewardConfig', 'reward', 'hyper', 'hyperparameters'];
  return keys.some(key => key in value);
}

export function createAITuner(options = {}) {
  const {
    getVecEnv = () => null,
    fetchTelemetry,
    applyRewardConfig,
    applyHyperparameters,
    log,
    modelId,
    instruction,
    getInstruction,
    isCheckpointEnabled,
  } = options;

  if (typeof fetchTelemetry !== 'function') {
    throw new Error('createAITuner requires a fetchTelemetry() function');
  }

  const logger = typeof log === 'function' ? log : () => {};
  const resolveEnv = typeof getVecEnv === 'function' ? getVecEnv : () => getVecEnv ?? null;
  let enabled = false;
  let interval = 500;
  let busy = false;
  let warnedNoFetch = false;
  let lastAnalysisText = '';

  async function resolveInstructionValue(context) {
    try {
      if (typeof getInstruction === 'function') {
        const dynamicValue = await Promise.resolve(getInstruction(context));
        if (typeof dynamicValue === 'string' && dynamicValue.trim()) {
          return dynamicValue.trim();
        }
      }
    } catch (err) {
      console.warn('[hf-tuner] kunde inte hämta dynamisk instruktion', err);
    }

    if (typeof instruction === 'string' && instruction.trim()) {
      return instruction.trim();
    }

    return '';
  }

  function logEvent(payload) {
    try {
      logger(payload);
    } catch (err) {
      console.warn('[hf-tuner] failed to log event', err);
    }
  }

  async function runTuningCycle(telemetry, episode) {
    if (typeof fetch !== 'function') {
      if (!warnedNoFetch) {
        logEvent({
          title: 'AI Auto-Tune',
          detail: 'fetch saknas i miljön – kan inte kontakta Hugging Face.',
          tone: 'error',
          episodeNumber: episode,
        });
        warnedNoFetch = true;
      }
      return;
    }

    const activeModel = resolveModelId(modelId);
    const meta = telemetry && typeof telemetry === 'object' ? telemetry.meta || {} : {};
    const intervalEpisodes = telemetry?.intervalEpisodes ?? interval;
    const rewardReference = telemetry?.currentConfig?.reward || telemetry?.rewardConfig || {};
    const hyperReference = telemetry?.currentConfig?.hyper || telemetry?.hyper || {};
    const currentConfig = {
      reward: { ...rewardReference },
      hyper: { ...hyperReference },
    };

    const messagePayload = {
      version: 2,
      algoId: meta.algo ?? null,
      intervalEpisodes,
      meta: {
        episode: meta.episode ?? null,
        board: meta.board ?? null,
        envs: meta.envs ?? null,
        best: meta.best ?? null,
        agent: meta.agent ?? null,
      },
      stats: telemetry?.stats ?? {},
      currentConfig,
    };

    if (telemetry?.rewardBreakdown) {
      messagePayload.rewardBreakdown = telemetry.rewardBreakdown;
    }
    if (telemetry?.crash) {
      messagePayload.crash = telemetry.crash;
    }
    if (telemetry?.rollup) {
      messagePayload.rollup = telemetry.rollup;
    }
    if (Array.isArray(telemetry?.recentEpisodes) && telemetry.recentEpisodes.length) {
      messagePayload.recent = {
        window: telemetry.recentEpisodes.length,
        episodes: telemetry.recentEpisodes,
      };
    }
    if (telemetry?.latestEpisode) {
      messagePayload.latestEpisode = telemetry.latestEpisode;
      const checkpointEntry = {
        type: 'checkpoint',
        ts: new Date().toISOString(),
        intervalEpisodes,
        episode: telemetry.latestEpisode.episode ?? meta.episode ?? null,
        algoId: meta.algo ?? null,
        reward: telemetry.latestEpisode.reward ?? null,
        fruits: telemetry.latestEpisode.fruits ?? null,
        steps: telemetry.latestEpisode.steps ?? null,
        crash: telemetry.latestEpisode.crash ?? null,
      };
      if (telemetry.latestEpisode.loopHits !== undefined) {
        checkpointEntry.loopHits = telemetry.latestEpisode.loopHits;
      }
      if (telemetry.latestEpisode.revisitPenalty !== undefined) {
        checkpointEntry.revisitPenalty = telemetry.latestEpisode.revisitPenalty;
      }
      if (telemetry.latestEpisode.timeToFruitAvg !== undefined) {
        checkpointEntry.timeToFruitAvg = telemetry.latestEpisode.timeToFruitAvg;
      }
      await appendSnakeHistory(checkpointEntry);
    }

    const instructionText = await resolveInstructionValue({
      telemetry: messagePayload,
      rawTelemetry: telemetry,
      episode,
      interval,
      model: activeModel,
    });
    const requestBody = {
      telemetry: messagePayload,
      model: activeModel,
    };

    if (instructionText) {
      requestBody.instruction = instructionText;
    }

    const response = await fetch(buildProxyUrl(), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const text = await response.text();

      let parsedError = null;
      try {
        parsedError = JSON.parse(text);
      } catch (parseErr) {
        // ignore – handled generically below
      }

      const baseMessage = parsedError?.error || text || 'Okänt fel från proxyn';
      const rawPayload = typeof parsedError?.raw === 'string' ? parsedError.raw : '';
      const upstreamModel = parsedError?.model || activeModel;
      const upstreamUrl = parsedError?.url;
      const statusText = parsedError?.statusText;
      if (rawPayload) {
        console.error('[hf-tuner] Proxy råsvar (fel):', rawPayload);
      }
      const enriched = [
        `Proxy ${response.status}${statusText ? ` (${statusText})` : ''}: ${baseMessage}`,
        upstreamModel ? `modell: ${upstreamModel}` : null,
        upstreamUrl ? `endpoint: ${upstreamUrl}` : null,
        rawPayload ? `HF: ${rawPayload}` : null,
      ]
        .filter(Boolean)
        .join(' | ');
      throw new Error(enriched);
    }

    const payload = await response.json();
    if (payload?.error) {
      const baseMessage = payload.error;
      const rawDetails = typeof payload.raw === 'string' && payload.raw ? payload.raw : '';
      if (rawDetails) {
        console.error('[hf-tuner] Proxy råsvar (fel):', rawDetails);
      }
      const enriched = [
        `Proxy error: ${baseMessage}`,
        payload?.model ? `modell: ${payload.model}` : null,
        payload?.url ? `endpoint: ${payload.url}` : null,
        rawDetails ? `HF: ${rawDetails}` : null,
      ]
        .filter(Boolean)
        .join(' | ');
      throw new Error(enriched);
    }

    const rawText = typeof payload?.raw === 'string' ? payload.raw : '';
    if (rawText) {
      console.log('[hf-tuner] Hugging Face råsvar:', rawText);
    }

    if (payload?.contentType) {
      console.log('[hf-tuner] Hugging Face content-type:', payload.contentType);
    }

    if (payload?.model) {
      console.log('[hf-tuner] Modell som användes:', payload.model);
    }

    if (payload?.url) {
      console.log('[hf-tuner] Hugging Face-endpoint:', payload.url);
    }

    const data = payload?.data ?? payload;
    const parsed = extractTuningPayload(data, rawText);

    if (!parsed || typeof parsed !== 'object') {
      const snippet = rawText ? ` Rådata: ${rawText.slice(0, 200)}` : '';
      throw new Error(`Saknar giltigt JSON-svar från modellen.${snippet}`);
    }

    const rewardTarget = fillMissingKeys(parsed.rewardConfig || parsed.reward || {}, currentConfig.reward);
    const hyperTarget = fillMissingKeys(parsed.hyper || parsed.hyperparameters || {}, currentConfig.hyper);
    const shouldLogPrompt = typeof isCheckpointEnabled === 'function' ? !!isCheckpointEnabled() : false;

    const rewardResult = typeof applyRewardConfig === 'function'
      ? applyRewardConfig(rewardTarget)
      : { changes: [], config: null };

    const hyperResult = typeof applyHyperparameters === 'function'
      ? applyHyperparameters(hyperTarget)
      : { changes: [], hyper: null };

    const envInstance = resolveEnv?.();
    if (envInstance?.setRewardConfig && rewardResult?.config) {
      try {
        envInstance.setRewardConfig(rewardResult.config);
      } catch (err) {
        console.warn('[hf-tuner] setRewardConfig failed', err);
      }
    }

    const rewardSummary = formatChanges(rewardResult?.changes);
    const hyperSummary = formatChanges(hyperResult?.changes);

    const analysisParts = [];
    const primaryAnalysis =
      typeof parsed.analysisText === 'string'
        ? parsed.analysisText.trim()
        : typeof parsed.analysis?.analysisText === 'string'
          ? parsed.analysis.analysisText.trim()
          : '';
    if (primaryAnalysis) {
      analysisParts.push(primaryAnalysis);
    }
    const summaryField = parsed.summary ?? parsed.analysis?.summary ?? null;
    if (Array.isArray(summaryField)) {
      const joined = summaryField
        .map(item => (typeof item === 'string' ? item.trim() : ''))
        .filter(Boolean)
        .join(' ');
      if (joined) {
        analysisParts.push(joined);
      }
    } else if (typeof summaryField === 'string') {
      const trimmed = summaryField.trim();
      if (trimmed) {
        analysisParts.push(trimmed);
      }
    }
    let combinedAnalysis = analysisParts.join(' ').replace(/\s+/g, ' ').trim();
    if (combinedAnalysis) {
      combinedAnalysis = limitWords(combinedAnalysis, 80).trim();
    }

    if (combinedAnalysis) {
      lastAnalysisText = combinedAnalysis;
      console.log('[AI Analysis]', combinedAnalysis);
      logEvent({ title: 'AI analys', detail: combinedAnalysis, tone: 'summary', episodeNumber: episode });
    }

    if (shouldLogPrompt) {
      await appendSnakeHistory({
        type: 'analysis',
        ts: new Date().toISOString(),
        episode: meta.episode ?? telemetry?.latestEpisode?.episode ?? null,
        model: payload?.model ?? activeModel,
        prompt: messagePayload,
        response: {
          rewardConfig: rewardTarget,
          hyper: hyperTarget,
          analysisText: primaryAnalysis || null,
          summary: summaryField ?? null,
        },
      });
    }

    if (rewardSummary) {
      logEvent({ title: 'AI belöningar', detail: rewardSummary, tone: 'reward', episodeNumber: episode });
    }
    if (hyperSummary) {
      logEvent({ title: 'AI hyperparametrar', detail: hyperSummary, tone: 'lr', episodeNumber: episode });
    }
    if (!rewardSummary && !hyperSummary) {
      logEvent({ title: 'AI Auto-Tune', detail: 'Ingen justering rekommenderades.', tone: 'ai', episodeNumber: episode });
    }
  }

  function maybeTune({ episode } = {}) {
    if (!enabled) return;
    if (!episode || episode % interval !== 0) return;
    if (busy) return;
    busy = true;
    (async () => {
      try {
        const telemetry = await Promise.resolve(fetchTelemetry({ interval, episode }));
        if (!telemetry) {
          return;
        }
        await runTuningCycle(telemetry, episode);
      } catch (err) {
        console.error('[hf-tuner]', err);
        logEvent({ title: 'AI Auto-Tune', detail: err.message || String(err), tone: 'error', episodeNumber: episode });
      } finally {
        busy = false;
      }
    })();
  }

  return {
    setEnabled(value) { enabled = !!value; },
    setInterval(value) {
      const num = Number(value);
      if (Number.isFinite(num) && num > 0) {
        interval = Math.max(1, Math.floor(num));
      }
    },
    maybeTune,
    isEnabled() { return enabled; },
    getInterval() { return interval; },
    getLastAnalysis() { return lastAnalysisText; },
  };
}
