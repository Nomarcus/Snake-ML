# Snake-ML

Browser-first reinforcement learning playground for the classic Snake environment. The repository now focuses on the in-browser simulator, Auto-PPO helper, and the Hugging Face proxy used by the AI auto-tuner.

## Highlights

- **Fully client-side training loops** – TensorFlow.js runs in the browser via `index.html`, so no Node.js bindings are required.
- **Auto-PPO controller** – UI exposes an inline Auto-PPO brain that stages PPO hyperparameters and reward weights automatically while persisting state and logging transitions.
- **AI auto-tuner** – `hf-tuner.js` can delegate reward/hyperparameter suggestions to Hugging Face models through the lightweight proxy (`api/proxy.js`).
- **Render-friendly backend** – the only server component is the minimal Express proxy that keeps your HF token off the client.


## Hugging Face-proxy för AI Auto-Tune


AI Auto-Tune går via en liten Express-server (`api/proxy.js`) som körs som en Render Web Service. Den exponerar `POST /api/proxy`, accepterar ett JSON-objekt `{ "telemetry": ..., "instruction": ... }`, läser din Hugging Face-token från miljövariabeln `HF_TOKEN` och returnerar svar från Inference API:t utan att tokenen någonsin lämnar backend.

### Deploya på Render

1. Skapa ett lästoken i [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens) om du inte redan har ett.
2. På [Render](https://dashboard.render.com/) väljer du **New → Web Service** och kopplar ditt Snake-ML-repo.
3. Ställ in:
   - **Environment**: Node
   - **Region**: valfri
   - **Build Command**: `npm install`
   - **Start Command**: `node api/proxy.js`
4. Under **Environment → Add Environment Variable** lägger du till `HF_TOKEN` med ditt Hugging Face-lästoken.
5. Deploya tjänsten. Render tilldelar en publik URL, exempelvis `https://snake-ml-backend.onrender.com`.

### Koppla frontend till proxyn

- Om backend och frontend ligger på samma host (t.ex. du proxar via ett eget domännamn) kan frontend fortsätta anropa `fetch('/api/proxy', ...)` utan ändringar.
- För fristående frontend (GitHub Pages, Netlify, etc.) sätter du `window.API_BASE_URL` till basadressen för Render-tjänsten innan `hf-tuner.js` laddas:

  ```html
  <script>
    window.API_BASE_URL = 'https://snake-ml-backend.onrender.com';
  </script>
  <script type="module" src="hf-tuner.js"></script>
  ```

  `hf-tuner.js` fogar automatiskt på `/api/proxy` så att alla anrop går via Render-backenden.


## Frontend usage

- Open `index.html` in a modern browser to launch the dashboard, render the game board, and access Auto-PPO helpers.
- `ai-tuner.js` can talk to OpenAI directly when you provide an API key via `window.__OPENAI_KEY` or the `OPENAI_API_KEY` environment variable while running a local dev server.
- Telemetry charts live in `ui/charts.js`; feel free to customise the rendering without worrying about Node-side training code.

## Development

- Source is written in modern ES modules (`package.json` sets `"type": "module"`).
- TensorFlow.js Node bindings (`@tensorflow/tfjs-node`) are used for training.
- All heavy file operations (checkpoint pruning, log rotation) run through the shared `DiskGuard` helper which rate-limits work to once every 30 seconds.
- Legacy `presets.js` (which exposed the global `SNAKE_PRESETS` registry and `applySnakePreset` helper) has been removed; UI customization now flows through the current module entry points.

## Example workflows

```bash
# Auto mode with 16 parallel environments and larger replay buffer
node train.js --mode auto --envCount 16 --bufferSize 750000 --saveDir models/auto-exp

# Manual mode, 4 environments, custom gamma and epsilon schedule
node train.js --mode manual --envCount 4 --gamma 0.985 --eps-end 0.1 --saveDir models/manual-finetune
```

Logs and checkpoints are available inside the configured `saveDir`. `latest/checkpoint.json` always reflects the most recent successful save, while `best/checkpoint.json` tracks the best evaluation result.
