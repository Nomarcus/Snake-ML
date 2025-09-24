const PROXY_PATH = '/api/proxy';

const SYSTEM_PROMPT = `Du är en expert på reinforcement learning.
Ditt mål är att justera Snake-MLs belöningsparametrar och centrala
hyperparametrar så att ormen klarar spelet konsekvent.
Returnera ENDAST minifierad JSON med nya värden för alla parametrar
du vill uppdatera, t.ex.
{
  "rewardConfig": {stepPenalty:0.008, fruitReward:12, ...},
  "hyper": {gamma:0.985, lr:0.0004, epsDecay:90000, ...}
}`;

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

function extractJsonPayload(text) {
  if (typeof text !== 'string') return null;
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start === -1 || end === -1 || end < start) return null;
  const snippet = text.slice(start, end + 1);
  try {
    return JSON.parse(snippet);
  } catch (err) {
    console.warn('[hf-tuner] kunde inte tolka JSON från modellen', err);
    return null;
  }
}

export function createAITuner(options = {}) {
  const {
    getVecEnv = () => null,
    fetchTelemetry,
    applyRewardConfig,
    applyHyperparameters,
    log,
  } = options;

  if (typeof fetchTelemetry !== 'function') {
    throw new Error('createAITuner requires a fetchTelemetry() function');
  }

  const logger = typeof log === 'function' ? log : () => {};
  const resolveEnv = typeof getVecEnv === 'function' ? getVecEnv : () => getVecEnv ?? null;
  let enabled = false;
  let interval = 1000;
  let busy = false;
  let warnedNoFetch = false;

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

    const response = await fetch(buildProxyUrl(), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        telemetry,
        instruction: SYSTEM_PROMPT,
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Proxy ${response.status}: ${text.slice(0, 200)}`);
    }

    const data = await response.json();
    if (data?.error) {
      throw new Error(`Proxy error: ${data.error}`);
    }

    const primary = Array.isArray(data) ? data[0] : data;
    const text = primary?.generated_text ?? primary?.output_text ?? primary?.content ?? '';

    const parsed = extractJsonPayload(text);
    if (!parsed) {
      throw new Error('Saknar giltigt JSON-svar från modellen.');
    }

    const rewardResult = typeof applyRewardConfig === 'function'
      ? applyRewardConfig(parsed.rewardConfig || parsed.reward || {})
      : { changes: [], config: null };

    const hyperResult = typeof applyHyperparameters === 'function'
      ? applyHyperparameters(parsed.hyper || parsed.hyperparameters || {})
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
  };
}
