import express from 'express';
import fetch from 'node-fetch';

const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const DEFAULT_MODEL_ID =
  process.env.GROQ_MODEL?.trim() || 'llama-3.1-8b-instant';

const SYSTEM_PROMPT = `You are an expert reinforcement-learning coach for the classic game Snake.
The agent plays Snake on a 2-D grid where it collects fruit and grows longer.
The telemetry you receive describes recent episodes, current reward parameters, and performance trends.
Your job is to:

Evaluate the agent’s long-term progress and stability.

Suggest specific numeric adjustments to reward settings and key hyperparameters that will increase the chance of consistently reaching the maximum score without overfitting.

Explain your reasoning in 1–2 short paragraphs so a developer can follow your thought process.
Always respond with valid JSON containing:

{
  "rewardConfig": {...},
  "hyper": {...},
  "analysisText": "clear explanation of trends and adjustments"
}

Do not remove all rewards or penalties unless you clearly explain why that is optimal.`;

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

  const token = process.env.GROQ_API_KEY;
  if (!token) {
    return res.status(500).json({ error: 'GROQ_API_KEY saknas i miljön.' });
  }

  try {
    const { telemetry, instruction, model } = req.body ?? {};

    if (typeof telemetry === 'undefined') {
      return res.status(400).json({ error: 'Fältet "telemetry" saknas.' });
    }

    const targetModel = resolveModelId(model);
    const userContent = buildUserContent({ telemetry, instruction });

    const payload = {
      model: targetModel,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: userContent },
      ],
    };

    const response = await fetch(GROQ_API_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const rawText = await response.text();
    console.log('Groq raw response:', rawText);

    if (!response.ok) {
      const parsedError = safeJsonParse(rawText);
      const message =
        extractErrorMessage(parsedError) || `Groq svarade ${response.status}`;
      return res.status(response.status).json({
        error: message,
        raw: rawText.slice(0, 2000),
        status: response.status,
      });
    }

    const data = safeJsonParse(rawText);
    if (data === null) {
      return res.status(502).json({
        error: 'Kunde inte tolka svaret från Groq.',
        raw: rawText.slice(0, 2000),
        status: 502,
      });
    }

    return res.status(200).json({
      data,
      raw: rawText,
      status: 200,
      model: targetModel,
    });
  } catch (error) {
    console.error('Proxyfel:', error);
    return res.status(500).json({
      error: 'Oväntat fel i proxyn.',
      raw: error instanceof Error && error.message ? error.message : undefined,
      status: 500,
    });
  }
});

const port = resolvePort(process.env.PORT);
app.listen(port, '0.0.0.0', () => {
  console.log(`Proxyserver lyssnar på port ${port}`);
});

function buildUserContent({ telemetry, instruction }) {
  const parts = [];
  if (typeof instruction === 'string' && instruction.trim()) {
    parts.push(instruction.trim());
  }

  const telemetryPart =
    typeof telemetry === 'string'
      ? telemetry
      : JSON.stringify(telemetry);
  parts.push(telemetryPart);

  return parts.join('\n\n');
}

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

function resolveModelId(candidate) {
  const candidates = [candidate, DEFAULT_MODEL_ID];

  for (const item of candidates) {
    if (typeof item !== 'string') {
      continue;
    }
    const trimmed = item.trim();
    if (trimmed) {
      return trimmed;
    }
  }

  return DEFAULT_MODEL_ID;
}

function safeJsonParse(text) {
  if (typeof text !== 'string' || !text.trim()) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch (error) {
    return null;
  }
}

function extractErrorMessage(parsed) {
  if (!parsed || typeof parsed !== 'object') {
    return '';
  }

  if (typeof parsed.error === 'string') {
    return parsed.error;
  }

  if (parsed.error && typeof parsed.error === 'object') {
    if (typeof parsed.error.message === 'string') {
      return parsed.error.message;
    }
    if (typeof parsed.error.error === 'string') {
      return parsed.error.error;
    }
  }

  if (typeof parsed.message === 'string') {
    return parsed.message;
  }

  return '';
}

function resolvePort(candidate) {
  const parsed = Number.parseInt(candidate ?? '', 10);
  if (Number.isFinite(parsed) && parsed > 0) {
    return parsed;
  }

  return 3001;
}

export default app;
