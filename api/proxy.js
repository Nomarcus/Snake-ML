
import express from 'express';
import fetch from 'node-fetch';

const HF_API_URL =
  'https://api-inference.huggingface.co/models/Qwen/Qwen2-7B-Instruct';


const HF_BASE_URL =
  (process.env.HF_BASE_URL && process.env.HF_BASE_URL.trim().replace(/\/+$/, '')) ||
  'https://api-inference.huggingface.co/models';
const DEFAULT_MODEL_ID =
  (process.env.HF_MODEL_ID && process.env.HF_MODEL_ID.trim()) ||
  'Qwen/Qwen2-7B-Instruct';


const SYSTEM_PROMPT = `Du är en expert på reinforcement learning.
Ditt mål är att justera Snake-MLs belöningsparametrar och centrala
hyperparametrar så att ormen klarar spelet konsekvent.
Returnera ENDAST minifierad JSON med nya värden för alla parametrar
du vill uppdatera, t.ex.
{
  "rewardConfig": {stepPenalty:0.008, fruitReward:12, ...},
  "hyper": {gamma:0.985, lr:0.0004, epsDecay:90000, ...}
}
Svara **endast** med en enda rad giltig JSON utan någon extra text eller förklaring.`;

const DEFAULT_ALLOWED_ORIGINS = [
  'https://nomarcus.github.io',
  'http://localhost:3000',
  'http://localhost:4173',
  'http://localhost:5173',
  'http://localhost:8080',
  'http://127.0.0.1:3000',
  'http://127.0.0.1:4173',
  'http://127.0.0.1:5173',
  'http://127.0.0.1:8080',
];

const allowedOriginSet = buildAllowedOriginSet(
  process.env.CORS_ALLOW_ORIGINS,
  DEFAULT_ALLOWED_ORIGINS,
);
const allowAllOrigins = allowedOriginSet.has('*');
if (allowAllOrigins) {
  allowedOriginSet.delete('*');
}

const app = express();

app.use((req, res, next) => {
  applyCorsHeaders({ req, res, allowAllOrigins, allowedOriginSet });
  next();
});

app.use(express.json({ limit: '1mb' }));

app.options('/api/proxy', (req, res) => {
  const { allowed } = applyCorsHeaders({ req, res, allowAllOrigins, allowedOriginSet });
  if (req.headers.origin && !allowed) {
    return res.status(403).json({ error: 'Otillåten origin.' });
  }
  res.setHeader('Access-Control-Allow-Methods', 'POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  return res.sendStatus(204);
});


app.post('/api/proxy', async (req, res) => {
  const { allowed } = applyCorsHeaders({ req, res, allowAllOrigins, allowedOriginSet });
  if (req.headers.origin && !allowed) {
    return res.status(403).json({ error: 'Otillåten origin.' });
  }

  let targetModelId = DEFAULT_MODEL_ID;
  let targetUrl = buildModelUrl(targetModelId);
  try {
    const { telemetry, instruction, model, modelId } = req.body ?? {};

    if (!telemetry) {
      return res.status(400).json({ error: 'Fältet "telemetry" saknas.' });
    }

    const token = process.env.HF_TOKEN;
    if (!token) {
      return res.status(500).json({ error: 'HF_TOKEN saknas i miljön.' });
    }

    targetModelId = resolveModelId(model ?? modelId);
    targetUrl = buildModelUrl(targetModelId);


    const payload = {
      inputs: `${
        typeof instruction === 'string' && instruction.trim()
          ? instruction
          : SYSTEM_PROMPT
      }\n\n${JSON.stringify(telemetry)}`,
      parameters: {
        temperature: 0.2,
        max_new_tokens: 300,
      },
    };


    const response = await fetch(targetUrl, {

      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const text = await response.text();
    console.log('HF raw response:', text);
    let content = null;
    let parseError = null;

    const contentType = response.headers.get('content-type') || '';


    if (text) {
      try {
        content = safeJsonParse(text);
      } catch (err) {
        parseError = err;


        const recovered = tryRecoverInlineJson(text);
        if (recovered !== null) {
          content = recovered;
          parseError = null;
        }
      }
    }

    const rawText = typeof text === 'string' ? text : '';

    if (!response.ok) {

      const upstreamError =
        (content && typeof content.error === 'string' && content.error.trim())
          ? content.error.trim()
          : rawText.trim();


      let message = upstreamError || `Hugging Face svarade ${response.status}`;

      if (response.status === 401) {
        message = upstreamError || 'Ogiltig eller saknad Hugging Face-token.';
      } else if (response.status === 403) {
        message =
          upstreamError || 'Behörighet saknas för den valda Hugging Face-modellen.';
      } else if (response.status === 404) {
        const default404 =
          'Hugging Face rapporterade 404 (Not Found). Kontrollera modellnamnet.';
        message = upstreamError || default404;

      }

      return res.status(response.status).json({
        error: typeof message === 'string' ? message.slice(0, 500) : 'Fel från Hugging Face.',

        raw: rawText ? rawText.slice(0, 2000) : undefined,
        model: targetModelId,
        url: targetUrl,
        statusText: response.statusText,

      });
    }

    if (parseError) {

      if (rawText.trim().toLowerCase() === 'not found') {
        return res.status(404).json({
          error: 'Hugging Face svarade "Not Found" – kontrollera modellnamn eller åtkomst.',
          raw: rawText.slice(0, 2000),
          model: targetModelId,
          url: targetUrl,

        });
      }
      throw parseError;
    }

    return res.status(200).json({
      data: content,
      raw: rawText,
      contentType,

      model: targetModelId,
      url: targetUrl,

    });
  } catch (error) {
    console.error('Proxyfel:', error);
    const statusCode = error instanceof JsonParseError ? 502 : 500;
    const message = error instanceof JsonParseError ? error.message : 'Oväntat fel i proxyn.';

    const responsePayload = {
      error: message,
      raw: error instanceof JsonParseError && error.raw ? error.raw.slice(0, 2000) : undefined,
    };

    if (targetModelId) {
      responsePayload.model = targetModelId;
    }
    if (targetUrl) {
      responsePayload.url = targetUrl;
    }

    return res.status(statusCode).json(responsePayload);

  }
});

function applyCorsHeaders({ req, res, allowAllOrigins, allowedOriginSet }) {
  appendVaryHeader(res, 'Origin');

  const origin = req.headers.origin;
  if (!origin) {
    return { allowed: true, applied: false };
  }

  const normalizedOrigin = normalizeOrigin(origin);
  if (!normalizedOrigin) {
    return { allowed: false, applied: false };
  }

  if (allowAllOrigins) {
    res.setHeader('Access-Control-Allow-Origin', '*');
  } else if (allowedOriginSet.has(normalizedOrigin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
  } else {
    return { allowed: false, applied: false };
  }

  res.setHeader('Access-Control-Allow-Methods', 'POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  return { allowed: true, applied: true };
}

function buildAllowedOriginSet(envValue, defaults = []) {
  const baseList = Array.isArray(defaults) ? defaults : [];
  const extraList =
    typeof envValue === 'string' && envValue.trim()
      ? envValue
          .split(',')
          .map(value => value.trim())
          .filter(Boolean)
      : [];

  const result = new Set();

  for (const value of [...baseList, ...extraList]) {
    if (value === '*') {
      result.add('*');
      continue;
    }

    const normalized = normalizeOrigin(value);
    if (normalized) {
      result.add(normalized);
    }
  }

  return result;
}

function normalizeOrigin(value) {
  if (typeof value !== 'string') {
    return '';
  }

  return value.trim().replace(/\/+$/, '').toLowerCase();
}

function appendVaryHeader(res, value) {
  const current = res.getHeader('Vary');
  if (!current) {
    res.setHeader('Vary', value);
    return;
  }

  const existing = String(current)
    .split(',')
    .map(part => part.trim())
    .filter(Boolean);

  if (!existing.includes(value)) {
    existing.push(value);
    res.setHeader('Vary', existing.join(', '));
  }
}

class JsonParseError extends Error {
  constructor(message, raw) {
    super(message);
    this.name = 'JsonParseError';
    this.raw = raw;
  }
}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch (err) {
    const streamed = tryParseEventStream(text);
    if (streamed !== null) {
      return streamed;
    }
    throw new JsonParseError('Kunde inte tolka svaret från Hugging Face.', text);

  }
}

function tryRecoverInlineJson(text) {
  if (typeof text !== 'string') {
    return null;

  }

  const candidates = [];

  const objectStart = text.indexOf('{');
  const objectEnd = text.lastIndexOf('}');
  if (objectStart !== -1 && objectEnd !== -1 && objectEnd > objectStart) {
    candidates.push(text.slice(objectStart, objectEnd + 1));
  }


  const arrayStart = text.indexOf('[');
  const arrayEnd = text.lastIndexOf(']');
  if (arrayStart !== -1 && arrayEnd !== -1 && arrayEnd > arrayStart) {
    candidates.push(text.slice(arrayStart, arrayEnd + 1));
  }

  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate);
    } catch (err) {
      // ignore candidate and keep trying
    }
  }

  return null;
}

function tryParseEventStream(text) {
  if (typeof text !== 'string' || !text.trim()) {
    return null;
  }

  const lines = text.split(/\r?\n/);
  const jsonCandidates = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed === 'data: [DONE]') {
      continue;
    }
    if (trimmed.startsWith('data:')) {
      const payload = trimmed.slice(5).trim();
      if (payload) {
        jsonCandidates.push(payload);
      }
    }
  }

  for (let i = jsonCandidates.length - 1; i >= 0; i -= 1) {
    const candidate = jsonCandidates[i];
    try {
      return JSON.parse(candidate);
    } catch (err) {
      // Ignorera och prova nästa kandidat
    }
  }

  return null;

}

function resolveModelId(value) {
  const candidates = [value, process.env.HF_MODEL_ID, DEFAULT_MODEL_ID];

  for (const candidate of candidates) {
    if (typeof candidate !== 'string') {
      continue;
    }
    const trimmed = candidate.trim();
    if (trimmed) {
      return trimmed;
    }
  }

  return DEFAULT_MODEL_ID;
}

function buildModelUrl(modelId) {
  if (typeof modelId === 'string' && /^https?:\/\//i.test(modelId.trim())) {
    return modelId.trim();
  }

  const trimmed = typeof modelId === 'string' ? modelId.trim().replace(/^\/+/, '') : '';
  if (!trimmed) {
    return `${HF_BASE_URL}/${DEFAULT_MODEL_ID}`;
  }

  return `${HF_BASE_URL}/${trimmed}`;

}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
