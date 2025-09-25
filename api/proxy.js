
import express from 'express';
import cors from 'cors';
import fetch from 'node-fetch';


const HF_API_URL = 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3';

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

const app = express();


app.use(cors({ origin: 'https://nomarcus.github.io' }));
app.use(express.json({ limit: '1mb' }));


app.post('/api/proxy', async (req, res) => {
  try {
    const { telemetry, instruction } = req.body ?? {};

    if (!telemetry) {
      return res.status(400).json({ error: 'Fältet "telemetry" saknas.' });
    }

    const token = process.env.HF_TOKEN;
    if (!token) {
      return res.status(500).json({ error: 'HF_TOKEN saknas i miljön.' });
    }

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

    const response = await fetch(HF_API_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const text = await response.text();
    console.log('HF raw response:', text);
    const content = text ? safeJsonParse(text) : null;

    if (!response.ok) {
      const message = (content && content.error) || text || `Hugging Face svarade ${response.status}`;
      return res.status(response.status).json({
        error: typeof message === 'string' ? message.slice(0, 500) : 'Fel från Hugging Face.',
        raw: typeof text === 'string' ? text.slice(0, 2000) : undefined,
      });
    }

    return res.status(200).json({
      data: content,
      raw: typeof text === 'string' ? text : '',
    });
  } catch (error) {
    console.error('Proxyfel:', error);
    const statusCode = error instanceof JsonParseError ? 502 : 500;
    const message = error instanceof JsonParseError ? error.message : 'Oväntat fel i proxyn.';
    return res.status(statusCode).json({
      error: message,
      raw: error instanceof JsonParseError && error.raw ? error.raw.slice(0, 2000) : undefined,
    });
  }
});

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

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
