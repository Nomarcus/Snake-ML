const PROXY_PATH = '/api/proxy';
const DEFAULT_MODEL_ID = 'Qwen/Qwen1.5-7B-Chat';

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
    const instructionText = await resolveInstructionValue({ telemetry, episode, interval, model: activeModel });
    const requestBody = {
      telemetry,
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

    if (!parsed) {
      const snippet = rawText ? ` Rådata: ${rawText.slice(0, 200)}` : '';
      throw new Error(`Saknar giltigt JSON-svar från modellen.${snippet}`);
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
