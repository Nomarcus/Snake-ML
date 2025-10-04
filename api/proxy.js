import express from 'express';
import fetch from 'node-fetch';
import fs from 'fs/promises';
import path from 'path';

const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const DEFAULT_MODEL_ID =
  process.env.GROQ_MODEL?.trim() || 'llama-3.1-8b-instant';
const HISTORY_LOG_MAX_BYTES = 8 * 1024 * 1024;
const HISTORY_LOG_PATH = path.join(process.cwd(), 'api', 'logs', 'snake-history.jsonl');

/* ------------------  SYSTEM PROMPT  ------------------ */
const SYSTEM_PROMPT = `
You are an expert reinforcement-learning optimizer specialized in Proximal Policy Optimization (PPO) for Snake-ML.
The Snake agent plays on a 2-D grid. Your goal is to analyze its training telemetry and automatically propose stable
parameter and reward adjustments to maximize fruit collection and survival.

Incoming telemetry JSON always includes:
{
  "episode": number,
  "avgReward100": number,
  "avgRewardTrend": number,
  "fruitPerEp": number,
  "entropy": number,
  "entropyTrend": number,
  "wallShare": number,
  "loopShare": number,
  "timeoutShare": number,
  "config": {
    "gamma": number,
    "lr": number,
    "lam": number,
    "clipRatio": number,
    "entropyCoef": number,
    "valueCoef": number,
    "rewardConfig": {
      "fruitReward": number,
      "stepPenalty": number,
      "wallPenalty": number,
      "loopPenalty": number,
      "towardFruitBonus": number,
      "growthBonus": number,
      "openSpaceBonus": number
    }
  }
}

Your tasks:
1. Detect improvement, stagnation, or regression.
2. Adjust PPO hyperparameters and reward coefficients smoothly.
3. Always return a full JSON object — even if nothing changes.
4. Include a concise analysisText (1–3 sentences).

Output JSON format:
{
  "adjustments": {
    "gamma": <number>,
    "lr": <number>,
    "lam": <number>,
    "clipRatio": <number>,
    "entropyCoef": <number>,
    "valueCoef": <number>,
    "rewardConfig": {
      "fruitReward": <number>,
      "stepPenalty": <number>,
      "wallPenalty": <number>,
      "loopPenalty": <number>,
      "towardFruitBonus": <number>,
      "growthBonus": <number>,
      "openSpaceBonus": <number>
    }
  },
  "comment": "<short reasoning>"
}

Guidelines:
- If avgReward100 < −9 and fruitPerEp < 0.2 → increase exploration (entropyCoef +10-20%, clipRatio +0.05).
- If reward variance high → decrease lr 10-20%.
- If fruitPerEp > 0.5 and reward variance < 1.0 → decrease entropyCoef slightly.
- If wallShare > 0.7 → reduce wallPenalty by 1–2, raise towardFruitBonus by 0.01–0.02.
- Clamp to:
  0.0001≤lr≤0.0005,  0.004≤entropyCoef≤0.02,  0.15≤clipRatio≤0.35,
  0.9≤lam≤0.96,  0.94≤gamma≤0.99
- Rewards within:
  fruitReward 10–22, wallPenalty 6–14, stepPenalty 0.005–0.02
- Output only valid JSON, no markdown or commentary.
`;

/* ---------------------------------------------------- */

const DEFAULT_ALLOWED_ORIGINS = [
  'https://nomarcus.github.io',
  'http://localhost:3000',
  'http://localhost:4173',
  'http://localhost:5173',
  'http://localhost:8080',
  'http://127.0.0.1:3000',
  'http://127.0.0.1:4173',
  'http://127.0.0.1:5173',
  'http://127.0.0.1:8080'
];

const allowedOriginSet = buildAllowedOriginSet(
  process.env.CORS_ALLOW_ORIGINS,
  DEFAULT_ALLOWED_ORIGINS
);
const allowAllOrigins = allowedOriginSet.has('*');
if (allowAllOrigins) allowedOriginSet.delete('*');

const app = express();

/* ----------------  CORS + BODY PARSER ---------------- */
app.use((req, res, next) => {
  applyCorsHeaders({ req, res, allowAllOrigins, allowedOriginSet });
  next();
});
app.use(express.json({ limit: '1mb' }));

/* ------------------  CORS OPTIONS  ------------------ */
for (const route of ['/api/proxy', '/api/logs/snake-history.jsonl']) {
  app.options(route, (req, res) => {
    const { allowed } = applyCorsHeaders({ req, res, allowAllOrigins, allowedOriginSet });
    if (req.headers.origin && !allowed)
      return res.status(403).json({ error: 'Otillåten origin.' });
    res.setHeader('Access-Control-Allow-Methods', 'POST,OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    return res.sendStatus(204);
  });
}

/* ------------------  PPO PROXY ENDPOINT  ------------------ */
app.post('/api/proxy', async (req, res) => {
  const { allowed } = applyCorsHeaders({ req, res, allowAllOrigins, allowedOriginSet });
  if (req.headers.origin && !allowed)
    return res.status(403).json({ error: 'Otillåten origin.' });

  const token = process.env.GROQ_API_KEY;
  if (!token)
    return res.status(500).json({ error: 'GROQ_API_KEY saknas i miljön.' });

  try {
    const { telemetry, instruction, model } = req.body ?? {};
    if (!telemetry)
      return res.status(400).json({ error: 'Fältet "telemetry" saknas.' });

    const targetModel = resolveModelId(model);
    const payload = {
      model: targetModel,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: buildUserContent({ telemetry, instruction }) }
      ]
    };

    const response = await fetch(GROQ_API_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
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
        status: response.status
      });
    }

    const data = safeJsonParse(rawText);
    if (data === null)
      return res.status(502).json({
        error: 'Kunde inte tolka svaret från Groq.',
        raw: rawText.slice(0, 2000),
        status: 502
      });

    return res.status(200).json({ data, raw: rawText, model: targetModel });
  } catch (err) {
    console.error('Proxyfel:', err);
    return res.status(500).json({
      error: 'Oväntat fel i proxyn.',
      raw: err instanceof Error ? err.message : undefined
    });
  }
});

/* ------------------  LOGGING ENDPOINT  ------------------ */
app.post('/api/logs/snake-history.jsonl', async (req, res) => {
  const { allowed } = applyCorsHeaders({ req, res, allowAllOrigins, allowedOriginSet });
  if (req.headers.origin && !allowed)
    return res.status(403).json({ error: 'Otillåten origin.' });

  const body = req.body ?? {};
  let line = '';

  if (typeof body.line === 'string' && body.line.trim()) line = body.line.trim();
  else if (body.entry && typeof body.entry === 'object')
    try {
      line = JSON.stringify(body.entry);
    } catch {
      return res.status(400).json({ error: 'Kunde inte serialisera loggpost.' });
    }

  if (!line) return res.status(400).json({ error: 'Saknar loggrad.' });
  if (line.length > 200000)
    return res.status(413).json({ error: 'Loggrad för stor.' });

  try {
    await appendHistoryLine(line);
    return res.status(204).end();
  } catch (error) {
    console.error('Misslyckades spara history-logg', error);
    return res.status(500).json({ error: 'Kunde inte skriva history-logg.' });
  }
});

/* ------------------  START SERVER  ------------------ */
const port = resolvePort(process.env.PORT);
app.listen(port, '0.0.0.0', () => {
  console.log(`PPO Proxyserver lyssnar på port ${port}`);
});

/* ------------------  HELPERS  ------------------ */
function buildUserContent({ telemetry, instruction }) {
  const parts = [];
  if (typeof instruction === 'string' && instruction.trim())
    parts.push(instruction.trim());
  parts.push(typeof telemetry === 'string' ? telemetry : JSON.stringify(telemetry));
  return parts.join('\n\n');
}

function applyCorsHeaders({ req, res, allowAllOrigins, allowedOriginSet }) {
  appendVaryHeader(res, 'Origin');
  const origin = req.headers.origin;
  if (!origin) return { allowed: true, applied: false };
  const norm = normalizeOrigin(origin);
  if (!norm) return { allowed: false, applied: false };

  if (allowAllOrigins) res.setHeader('Access-Control-Allow-Origin', '*');
  else if (allowedOriginSet.has(norm)) res.setHeader('Access-Control-Allow-Origin', origin);
  else return { allowed: false, applied: false };

  res.setHeader('Access-Control-Allow-Methods', 'POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  return { allowed: true, applied: true };
}

function buildAllowedOriginSet(env, defaults = []) {
  const base = Array.isArray(defaults) ? defaults : [];
  const extra =
    typeof env === 'string' && env.trim()
      ? env.split(',').map(x => x.trim()).filter(Boolean)
      : [];
  const result = new Set();
  for (const v of [...base, ...extra]) {
    if (v === '*') result.add('*');
    else {
      const n = normalizeOrigin(v);
      if (n) result.add(n);
    }
  }
  return result;
}

function normalizeOrigin(v) {
  return typeof v === 'string' ? v.trim().replace(/\/+$/, '').toLowerCase() : '';
}

function appendVaryHeader(res, val) {
  const cur = res.getHeader('Vary');
  if (!cur) return res.setHeader('Vary', val);
  const arr = String(cur)
    .split(',')
    .map(p => p.trim())
    .filter(Boolean);
  if (!arr.includes(val)) {
    arr.push(val);
    res.setHeader('Vary', arr.join(', '));
  }
}

function resolveModelId(cand) {
  const list = [cand, DEFAULT_MODEL_ID];
  for (const c of list) {
    if (typeof c === 'string' && c.trim()) return c.trim();
  }
  return DEFAULT_MODEL_ID;
}

function safeJsonParse(txt) {
  if (typeof txt !== 'string' || !txt.trim()) return null;
  try {
    return JSON.parse(txt);
  } catch {
    return null;
  }
}

function extractErrorMessage(parsed) {
  if (!parsed || typeof parsed !== 'object') return '';
  if (typeof parsed.error === 'string') return parsed.error;
  if (parsed.error?.message) return parsed.error.message;
  if (parsed.message) return parsed.message;
  return '';
}

function resolvePort(val) {
  const n = Number.parseInt(val ?? '', 10);
  return Number.isFinite(n) && n > 0 ? n : 3001;
}

/* ------------------  LOG ROTATION ------------------ */
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
  if (stat.size <= HISTORY_LOG_MAX_BYTES) return;
  const backup = `${HISTORY_LOG_PATH}.1`;
  try {
    await fs.unlink(backup);
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
  }
  await fs.rename(HISTORY_LOG_PATH, backup);
  await fs.writeFile(HISTORY_LOG_PATH, '');
}

export default app;
