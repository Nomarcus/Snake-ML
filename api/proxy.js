
import express from 'express';
import cors from 'cors';
import fetch from 'node-fetch';


const HF_API_URL = 'https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta';

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

      let message;
      if (response.status === 401) {
        message = 'Ogiltig eller saknad Hugging Face-token.';
      } else if (response.status === 403) {
        message = 'Behörighet saknas för den valda Hugging Face-modellen.';
      } else if (response.status === 404) {
        message = 'Hugging Face rapporterade 404 (Not Found). Kontrollera modellnamnet.';
      } else if (upstreamError) {
        message = upstreamError;
      } else {
        message = `Hugging Face svarade ${response.status}`;
      }

      return res.status(response.status).json({
        error: typeof message === 'string' ? message.slice(0, 500) : 'Fel från Hugging Face.',

        raw: rawText ? rawText.slice(0, 2000) : undefined,
      });
    }

    if (parseError) {

      if (rawText.trim().toLowerCase() === 'not found') {
        return res.status(404).json({
          error: 'Hugging Face svarade "Not Found" – kontrollera modellnamn eller åtkomst.',
          raw: rawText.slice(0, 2000),
        });
      }
      throw parseError;
    }

    return res.status(200).json({
      data: content,
      raw: rawText,
      contentType,

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

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
