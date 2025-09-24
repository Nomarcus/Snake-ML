# Snake-ML

Refactored reinforcement learning playground for the Snake environment with support for automated multi-day training and disk-aware checkpointing.

## Highlights

- **Vectorised environments** – `VecSnakeEnv` runs multiple `SnakeEnv` instances in parallel and feeds a shared replay buffer.
- **AutoTrain mode** – adaptive scheduler tunes ε-greedy exploration, learning rate, PER hyperparameters, and reward weights within UI slider bounds. Curriculum automatically scales the board from 10×10 up to 20×20 as performance improves.
- **Manual mode** – keeps fixed hyperparameters but allows selecting the number of parallel environments.
- **Disk guardrails** – checkpoints and logs honour interval, cooldown, retention and disk size caps with atomic writes and automatic pruning.
- **Evaluations & rollback** – lightweight greedy evaluations every 2000 episodes, best-model retention, and rollback on catastrophic regression.

## Hugging Face token for AI Auto-Tune

The browser AI Auto-Tune integration now calls the Hugging Face Inference API with the model `mistralai/Mistral-7B-Instruct-v0.3`. The module reads a browser global named `window.__HF_KEY` from the checked-in `__hf_key.js` bootstrap script.

### Skaffa ett lästoken

1. Logga in på [huggingface.co](https://huggingface.co) och öppna **Settings → Access Tokens**.
2. Skapa ett nytt token med typ **Read** (räcker för Inference API).

### Konfigurera projektet

1. Öppna `__hf_key.js` och ersätt `hf_YOUR_TOKEN_HERE` med ditt Hugging Face-token.
2. Filen laddas före `hf-tuner.js`, vilket gör att token exponeras som `window.__HF_KEY` i webbläsaren.
3. Token distribueras tillsammans med statiska filer (GitHub Pages, lokal hosting). Den räknas därför som publik – använd ett separat lästoken för demo/test.

Workflowen `deploy.yml` kopierar `__hf_key.js` in i `dist/` utan att läsa hemligheter från GitHub Actions. Kontrollera in filen efter att du uppdaterat den så att rätt token följer med byggsteget.

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
