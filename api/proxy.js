import express from 'express';
import fetch from 'node-fetch';
import fs from 'fs/promises';
import path from 'path';

const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const DEFAULT_MODEL_ID =
  process.env.GROQ_MODEL?.trim() || 'llama-3.1-8b-instant';
const HISTORY_LOG_MAX_BYTES = 8 * 1024 * 1024;
const HISTORY_LOG_PATH = path.join(process.cwd(), 'api', 'logs', 'snake-history.jsonl');

const SYSTEM_PROMPT = `You are an advanced reinforcement learning tuner for Snake-ML.  
The agent plays Snake on a 2-D grid.  

You will always receive JSON telemetry containing:  
- General: episode, avgFruit, avgReward  
- Exploration: epsilon, epsilonDecay  
- Learning: gamma, learningRate, batchSize, targetUpdate  
- Environment: gridSize, maxSteps, snakeLength, fruitEaten  
- Reward config: fruitReward, deathPenalty, stepPenalty, loopPenalty, tailPenalty, wallPenalty, idlePenalty (if present)  

Your task:
- Analyze all telemetry and determine if the agent is improving, stagnating, or regressing.  
- Propose numeric adjustments to both rewardConfig and hyperparameters if they are likely to improve learning.  
- If performance is stable and improving, return the same values.  
- If you detect instability, runaway behavior, or catastrophic degradation, set "pause": true to pause training until the human resumes. Otherwise set "pause": false.  

Output must always be strict JSON in this format:  
{
  "rewardConfig": {
    "fruitReward": <number>,
    "deathPenalty": <number>,
    "stepPenalty": <number>,
    "loopPenalty": <number>,
    "tailPenalty": <number>,
    "wallPenalty": <number>,
    "idlePenalty": <number>
  },
  "hyper": {
    "gamma": <number>,
    "learningRate": <number>,
    "epsilonDecay": <number>,
    "batchSize": <number>,
    "targetUpdate": <number>,
    "gridSize": <number>,
    "maxSteps": <number>,
    "snakeLength": <number>,
    "fruitEaten": <number>
  },
  "pause": <true|false>,
  "analysisText": "<short explanation (1–3 sentences)>"
}  

Rules:
- Always include ALL fields listed above, even if unchanged.  
- Never output extra commentary or markdown. JSON only.  
- If "pause": true, analysisText must explain why.  `;


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

app.options('/api/logs/snake-history.jsonl', (req, res) => {
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

app.post('/api/logs/snake-history.jsonl', async (req, res) => {
  const { allowed } = applyCorsHeaders({ req, res, allowAllOrigins, allowedOriginSet });
  if (req.headers.origin && !allowed) {
    return res.status(403).json({ error: 'Otillåten origin.' });
  }

  const body = req.body ?? {};
  let line = '';
  if (typeof body.line === 'string' && body.line.trim()) {
    line = body.line.trim();
  } else if (body.entry && typeof body.entry === 'object') {
    try {
      line = JSON.stringify(body.entry);
    } catch (err) {
      return res.status(400).json({ error: 'Kunde inte serialisera loggpost.' });
    }
  }

  if (!line) {
    return res.status(400).json({ error: 'Saknar loggrad.' });
  }

  if (line.length > 200000) {
    return res.status(413).json({ error: 'Loggrad för stor.' });
  }

  try {
    await appendHistoryLine(line);
    return res.status(204).end();
  } catch (error) {
    console.error('Misslyckades spara history-logg', error);
    return res.status(500).json({ error: 'Kunde inte skriva history-logg.' });
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
    const repaired = repairJsonSnippet(text);
    if (repaired !== text) {
      try {
        return JSON.parse(repaired);
      } catch (innerError) {
        console.warn('Kunde inte tolka reparerad JSON-sträng', innerError, {
          snippet: repaired.slice(0, 2000),
        });
      }
    }
    return null;
  }
}

function repairJsonSnippet(snippet) {
  if (typeof snippet !== 'string') {
    return snippet;
  }

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

async function appendHistoryLine(line) {
  await ensureHistoryDir();
  await fs.appendFile(HISTORY_LOG_PATH, `${line}\n`);
  await rotateHistoryLog();
}

async function ensureHistoryDir() {
  const dir = path.dirname(HISTORY_LOG_PATH);
  await fs.mkdir(dir, { recursive: true });
}

async function rotateHistoryLog() {
  let stat;
  try {
    stat = await fs.stat(HISTORY_LOG_PATH);
  } catch (err) {
    if (err.code === 'ENOENT') return;
    throw err;
  }

  if (stat.size <= HISTORY_LOG_MAX_BYTES) {
    return;
  }

  const backupPath = `${HISTORY_LOG_PATH}.1`;
  try {
    await fs.unlink(backupPath);
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
  }

  await fs.rename(HISTORY_LOG_PATH, backupPath);
  await fs.writeFile(HISTORY_LOG_PATH, '');
}

export default app;
