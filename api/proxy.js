const HF_API_URL = 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3';

const SYSTEM_PROMPT = `Du är en expert på reinforcement learning.
Ditt mål är att justera Snake-MLs belöningsparametrar och centrala
hyperparametrar så att ormen klarar spelet konsekvent.
Returnera ENDAST minifierad JSON med nya värden för alla parametrar
du vill uppdatera, t.ex.
{
  "rewardConfig": {stepPenalty:0.008, fruitReward:12, ...},
  "hyper": {gamma:0.985, lr:0.0004, epsDecay:90000, ...}
}`;

class ProxyError extends Error {
  constructor(statusCode, message) {
    super(message);
    this.statusCode = statusCode;
  }
}

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'content-type',
  'Access-Control-Allow-Methods': 'POST,OPTIONS',
};

function toJsonResponse(statusCode, payload = null) {
  return {
    statusCode,
    headers: corsHeaders,
    body: payload ? JSON.stringify(payload) : '',
  };
}

function extractTelemetry(body) {
  if (body && typeof body === 'object') {
    if ('telemetry' in body) {
      return body.telemetry;
    }
    return body;
  }
  return null;
}

function ensureJson(data) {
  if (!data) return {};
  if (typeof data === 'string') {
    try {
      return JSON.parse(data);
    } catch (err) {
      throw new ProxyError(400, 'Ogiltig JSON i begäran.');
    }
  }
  if (data instanceof Buffer) {
    return ensureJson(data.toString('utf8'));
  }
  return data;
}

async function invokeModel(telemetry, instruction = SYSTEM_PROMPT) {
  if (!telemetry) {
    throw new ProxyError(400, 'Fältet "telemetry" saknas.');
  }

  const token = process.env.HF_TOKEN;
  if (!token) {
    throw new ProxyError(500, 'HF_TOKEN saknas i miljön.');
  }

  const payload = {
    inputs: JSON.stringify({
      instruktion: instruction || SYSTEM_PROMPT,
      telemetri: telemetry,
    }),
    parameters: {
      temperature: 0.2,
      max_new_tokens: 300,
    },
  };

  const response = await fetch(HF_API_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (err) {
      throw new ProxyError(502, 'Kunde inte tolka svaret från Hugging Face.');
    }
  }

  if (!response.ok) {
    const message = data?.error || text || `Hugging Face svarade ${response.status}`;
    throw new ProxyError(response.status, typeof message === 'string' ? message.slice(0, 500) : 'Fel från Hugging Face.');
  }

  return data;
}

function send(res, statusCode, payload) {
  if (!res) {
    return toJsonResponse(statusCode, payload);
  }

  if (typeof res.setHeader === 'function') {
    Object.entries(corsHeaders).forEach(([key, value]) => {
      res.setHeader(key, value);
    });
  }

  if (typeof res.status === 'function') {
    res.status(statusCode);
    if (typeof res.json === 'function') {
      res.json(payload);
    } else {
      res.end(payload ? JSON.stringify(payload) : '');
    }
  } else if (typeof res.writeHead === 'function') {
    res.writeHead(statusCode, corsHeaders);
    res.end(payload ? JSON.stringify(payload) : '');
  } else {
    res.end(payload ? JSON.stringify(payload) : '');
  }

  return undefined;
}

async function handleRequestBody(body) {
  const parsed = ensureJson(body || {});
  const telemetry = extractTelemetry(parsed);
  const instruction = typeof parsed.instruction === 'string' ? parsed.instruction : SYSTEM_PROMPT;
  return invokeModel(telemetry, instruction);
}

export default async function handler(req, res) {
  const method = req?.method?.toUpperCase?.();

  if (method === 'OPTIONS') {
    return send(res, 204, null);
  }

  if (method && method !== 'POST') {
    return send(res, 405, { error: 'Endast POST stöds.' });
  }

  try {
    const data = await handleRequestBody(req?.body);
    return send(res, 200, data);
  } catch (err) {
    const statusCode = err instanceof ProxyError ? err.statusCode : 500;
    const message = err instanceof ProxyError ? err.message : 'Oväntat fel i proxyn.';
    return send(res, statusCode, { error: message });
  }
}

export const handler = async (event) => {
  const method = event?.httpMethod?.toUpperCase?.();

  if (method === 'OPTIONS') {
    return toJsonResponse(204);
  }

  if (method && method !== 'POST') {
    return toJsonResponse(405, { error: 'Endast POST stöds.' });
  }

  try {
    const data = await handleRequestBody(event?.body);
    return toJsonResponse(200, data);
  } catch (err) {
    const statusCode = err instanceof ProxyError ? err.statusCode : 500;
    const message = err instanceof ProxyError ? err.message : 'Oväntat fel i proxyn.';
    return toJsonResponse(statusCode, { error: message });
  }
};
