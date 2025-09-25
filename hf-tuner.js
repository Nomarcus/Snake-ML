const PROXY_PATH = '/api/proxy';
const DEFAULT_MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.3';

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
    modelId,
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
    // 👇 Ny rad för felsökning
    console.log('[hf-tuner] Telemetry som skickas till proxy:', telemetry);

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
    const response = await fetch(buildProxyUrl(), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        telemetry,
        instruction: SYSTEM_PROMPT,
        model: activeModel,
      }),
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

    const primary = Array.isArray(data) ? data[0] : data;
    const text = primary?.generated_text ?? primary?.output_text ?? primary?.content ?? '';

    let parsed = extractJsonPayload(text);
    if (Array.isArray(parsed)) {
      parsed = parsed[0];
    }
    if (!parsed || typeof parsed !== 'object') {
      const fallback = extractJsonPayload(rawText);
      const normalizedFallback = Array.isArray(fallback) ? fallback[0] : fallback;
      if (normalizedFallback && typeof normalizedFallback === 'object') {
        parsed = normalizedFallback;
        console.warn('[hf-tuner] JSON extraherat från råsvar efter fallback.');
      } else {
        const snippet = rawText ? ` Rådata: ${rawText.slice(0, 200)}` : '';
        throw new Error(`Saknar giltigt JSON-svar från modellen.${snippet}`);
      }
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
