# Snake-ML

Refactored reinforcement learning playground for the Snake environment with support for automated multi-day training and disk-aware checkpointing.

## Highlights

- **Vectorised environments** – `VecSnakeEnv` runs multiple `SnakeEnv` instances in parallel and feeds a shared replay buffer.
- **AutoTrain mode** – adaptive scheduler tunes ε-greedy exploration, learning rate, PER hyperparameters, and reward weights within UI slider bounds. Curriculum automatically scales the board from 10×10 up to 20×20 as performance improves.
- **Manual mode** – keeps fixed hyperparameters but allows selecting the number of parallel environments.
- **Disk guardrails** – checkpoints and logs honour interval, cooldown, retention and disk size caps with atomic writes and automatic pruning.
- **Evaluations & rollback** – lightweight greedy evaluations every 2000 episodes, best-model retention, and rollback on catastrophic regression.

## AI Auto-Tune API key

The browser-side AI Auto-Tune module calls OpenAI's GPT-4o-mini endpoint. It requires `OPENAI_API_KEY` to be present at runtime.

### Local development

1. Create a `.env` file in the project root containing your key:
   ```env
   OPENAI_API_KEY=sk-your-key
   ```
2. Export the variables before starting a dev server or static file host, for example:
   ```bash
   set -a
   source .env
   set +a
   npx http-server .
   ```
   Any tooling (`vite`, `webpack-dev-server`, etc.) works as long as it is launched from the same shell session so that `process.env.OPENAI_API_KEY` is populated.

### GitHub Pages / Actions

1. In your repository settings, add a secret named `OPENAI_API_KEY` containing the key.
2. The `deploy.yml` workflow writes `public/__key.js` before the build runs, injecting the key via `window.__OPENAI_KEY = '${{ secrets.OPENAI_API_KEY }}';` so the browser can read it.
3. Because the bootstrap script ships with the published HTML, the secret is exposed to clients — treat it as a public token scoped only for this project.

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

## Example workflows

```bash
# Auto mode with 16 parallel environments and larger replay buffer
node train.js --mode auto --envCount 16 --bufferSize 750000 --saveDir models/auto-exp

# Manual mode, 4 environments, custom gamma and epsilon schedule
node train.js --mode manual --envCount 4 --gamma 0.985 --eps-end 0.1 --saveDir models/manual-finetune
```

Logs and checkpoints are available inside the configured `saveDir`. `latest/checkpoint.json` always reflects the most recent successful save, while `best/checkpoint.json` tracks the best evaluation result.
