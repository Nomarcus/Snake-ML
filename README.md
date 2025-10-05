# Snake-ML

Refactored reinforcement learning playground for the Snake environment with support for automated multi-day training and disk-aware checkpointing.

## Highlights

- **Vectorised environments** – `VecSnakeEnv` runs multiple `SnakeEnv` instances in parallel and feeds a shared replay buffer.
- **AutoTrain mode** – adaptive scheduler tunes ε-greedy exploration, learning rate, PER hyperparameters, and reward weights within UI slider bounds. Curriculum automatically scales the board from 10×10 up to 20×20 as performance improves.
- **Auto-PPO controller** – browser UI ships with an inline Auto-PPO brain that stages PPO hyperparameters and reward weights automatically while persisting state and logging transitions.
- **Manual mode** – keeps fixed hyperparameters but allows selecting the number of parallel environments.
- **Disk guardrails** – checkpoints and logs honour interval, cooldown, retention and disk size caps with atomic writes and automatic pruning.
- **Evaluations & rollback** – lightweight greedy evaluations every 2000 episodes, best-model retention, and rollback on catastrophic regression.


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


## CLI usage

Training is handled through `train.js`. Install dependencies with `npm install` and start training, for example:

```bash
node train.js --mode auto   --envCount 12 --saveDir models/auto
node train.js --mode manual --envCount 1  --saveDir models/manual
```

### Common flags

| Flag | Description | Default |
| ---- | ----------- | ------- |
| `--mode auto\|manual` | Selects AutoTrain or Manual mode. | `auto` |
| `--envCount` | Number of parallel environments. | auto: `12`, manual: `1` |
| `--saveDir` | Directory for checkpoints/logs. | `models/<mode>` |
| `--checkpoint-interval-episodes` | Minimum episodes between periodic checkpoints. | `5000` |
| `--checkpoint-interval-minutes` | Minimum minutes between periodic checkpoints. | `60` |
| `--retain-checkpoints` | Number of periodic checkpoints to keep (latest & best are always kept). | `5` |
| `--save-cap-mb` | Maximum total size for `saveDir`. | `1024` |
| `--log-cap-mb` | Maximum total size for log directory. | `200` |
| `--retain-logs` | Number of rotated log archives to keep. | `3` |
| `--log-interval-episodes` | Episode interval for JSONL summaries. | `500` |
| `--console-interval-episodes` | Episode interval for console summaries. | `200` |
| `--min-save-cooldown-minutes` | Cooldown for extra checkpoints (board changes, best eval). | `10` |

Additional knobs map directly to hyperparameters: `--batchSize`, `--bufferSize`, `--gamma`, `--lr`, `--n-step`, `--target-sync`, `--priority-alpha`, `--priority-beta`, `--priority-beta-increment`, `--eps-start`, `--eps-end`, `--eps-decay`, `--board-size` (manual mode).

### AutoTrain specifics

- Curriculum upgrades board sizes at `maFruit500` > 60 (→14×14), > 120 (→18×18), and > 200 (→20×20). A checkpoint is triggered on each upgrade subject to cooldown.
- Adaptive scheduler reacts to plateaus/regressions by adjusting ε schedule, cosine-annealed learning rate, PER α/β, n-step horizon, γ, and reward weights while respecting UI slider bounds.
- Replay buffer is truncated by 50% of the oldest transitions when the board size increases to promote rapid adaptation.
- Evaluations run every 2000 episodes; new best models require at least +5 fruits and +2 % improvement with a 10 minute cooldown.
- If `maFruit100` drops 25 % beneath the best evaluation for ≥5000 episodes the trainer reloads the best checkpoint.

## Auto-PPO controller

- The browser UI exposes `window.autoPpo`, an instance of `AutoPPOController`, as soon as a PPO agent is instantiated.
- Six inline stages (S1, S2, S2B, S3, S3B, S4) tweak learning rate, γ, λ, clip ratio, entropy, value coefficient, and reward weights without relying on external preset files.
- Telemetry updates are aggregated defensively so partial episode data still advances the state machine and stagnation detection.
- State (current stage + enabled flag) persists in `localStorage`; a dedicated toggle and readout in the dashboard lets you disable or monitor Auto-PPO in real time.
- Stage transitions are logged to the console and, when the proxy is reachable, appended to `/api/logs/snake-history.jsonl`.

### Manual mode

Manual mode keeps all hyperparameters fixed but can still run multiple environments in parallel (`--envCount`). No auto-adjustments are performed; checkpoints/logging follow the same disk constraints as Auto mode.

## Disk and log policy

- **Atomic saves** – checkpoints are written to a temporary directory and renamed into place. Pointers `latest/` and `best/` are updated with atomic renames.
- **Intervals & cooldowns** – periodic checkpoints require both the episode and time thresholds; extra checkpoints (board change/best eval) respect the 10 minute cooldown. Shutdown checkpoints honour the same constraints.
- **Retention** – keep `latest/`, `best/`, and the most recent `--retain-checkpoints` periodic checkpoints while pruning the oldest ones or any exceeding `--save-cap-mb`.
- **Log rotation** – `training_log.jsonl` is appended every `--log-interval-episodes`. Files rotate at 10 MB, compress with gzip, and obey `--retain-logs` plus the `--log-cap-mb` budget.

## Watching the latest model

In browsers that support the File System Access API, the “Titta” button can load the latest checkpoint (`saveDir/latest/checkpoint.json`). When prompted, choose the directory containing your checkpoints (e.g. `models/auto`). The UI imports the model weights and renders a smooth evaluation run.

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
