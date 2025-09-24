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
}`;

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
      inputs: JSON.stringify({
        instruktion: typeof instruction === 'string' && instruction.trim() ? instruction : SYSTEM_PROMPT,
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
    const content = text ? safeJsonParse(text) : null;

    if (!response.ok) {
      const message = (content && content.error) || text || `Hugging Face svarade ${response.status}`;
      return res.status(response.status).json({
        error: typeof message === 'string' ? message.slice(0, 500) : 'Fel från Hugging Face.',
      });
    }

    return res.status(200).json(content);
  } catch (error) {
    console.error('Proxyfel:', error);
    const statusCode = error instanceof JsonParseError ? 502 : 500;
    const message = error instanceof JsonParseError ? error.message : 'Oväntat fel i proxyn.';
    return res.status(statusCode).json({ error: message });
  }
});

class JsonParseError extends Error {}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch (err) {
    throw new JsonParseError('Kunde inte tolka svaret från Hugging Face.');
  }
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
