import { clamp } from '../utils/ring_buffer.js';

export const REWARD_DEFAULTS = {
  stepPenalty: 0.01,
  turnPenalty: 0.001,
  approachBonus: 0.03,
  retreatPenalty: 0.03,
  loopPenalty: 0.5,
  revisitPenalty: 0.05,
  wallPenalty: 10,
  selfPenalty: 25.5,
  timeoutPenalty: 5,
  fruitReward: 10,
  growthBonus: 1,
  compactWeight: 0,
  trapPenalty: 0.5,
  spaceGainBonus: 0.05,
};

const LOOP_PATTERNS = new Set(['1,2,1,2', '2,1,2,1']);

export class SnakeEnv {
  constructor(cols = 20, rows = 20, rewardOverrides = {}) {
    this.cols = cols;
    this.rows = rows;
    this.setRewardConfig(rewardOverrides);
    this.reset();
  }

  setRewardConfig(overrides = {}) {
    this.reward = { ...REWARD_DEFAULTS, ...overrides };
  }

  idx(x, y) {
    return y * this.cols + x;
  }

  neighbors(x, y) {
    return [
      { x: x + 1, y },
      { x: x - 1, y },
      { x, y: y + 1 },
      { x, y: y - 1 },
    ].filter((p) => p.x >= 0 && p.y >= 0 && p.x < this.cols && p.y < this.rows);
  }

  freeSpaceFrom(sx, sy, tailWillMove) {
    const seen = new Set();
    const queue = [{ x: sx, y: sy }];
    const blocked = new Set(this.snakeSet);
    blocked.delete(`${sx},${sy}`);
    if (tailWillMove && this.snake.length) {
      const t = this.snake[this.snake.length - 1];
      blocked.delete(`${t.x},${t.y}`);
    }
    while (queue.length) {
      const p = queue.pop();
      const key = `${p.x},${p.y}`;
      if (seen.has(key)) continue;
      if (blocked.has(key)) continue;
      seen.add(key);
      for (const n of this.neighbors(p.x, p.y)) queue.push(n);
      if (seen.size > this.cols * this.rows) break;
    }
    return seen.size;
  }

  computeSlack() {
    if (!this.snake?.length) return 0;
    let minX = this.snake[0].x;
    let maxX = this.snake[0].x;
    let minY = this.snake[0].y;
    let maxY = this.snake[0].y;
    for (const seg of this.snake) {
      if (seg.x < minX) minX = seg.x;
      if (seg.x > maxX) maxX = seg.x;
      if (seg.y < minY) minY = seg.y;
      if (seg.y > maxY) maxY = seg.y;
    }
    const width = maxX - minX + 1;
    const height = maxY - minY + 1;
    const area = width * height;
    return Math.max(0, area - this.snake.length);
  }

  spawnFruit() {
    const free = [];
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        if (!this.snakeSet.has(`${x},${y}`)) free.push({ x, y });
      }
    }
    this.fruit = free.length ? free[(Math.random() * free.length) | 0] : { x: -1, y: -1 };
  }

  reset() {
    this.dir = { x: 1, y: 0 };
    const cx = (this.cols / 2) | 0;
    const cy = (this.rows / 2) | 0;
    this.snake = [{ x: cx - 1, y: cy }, { x: cx, y: cy }];
    this.snakeSet = new Set(this.snake.map((p) => `${p.x},${p.y}`));
    this.visit = new Float32Array(this.cols * this.rows).fill(0);
    this.actionHist = [];
    this.spawnFruit();
    this.steps = 0;
    this.stepsSinceFruit = 0;
    this.alive = true;
    this.prevSlack = this.computeSlack();
    this.maxLength = this.snake.length;
    this.episodeFruit = 0;
    this.loopHits = 0;
    this.revisitAccum = 0;
    this.timeToFruitAccum = 0;
    this.timeToFruitCount = 0;
    this.lastMoveApproach = 0;
    return this.getState();
  }

  getVisit(x, y) {
    if (x < 0 || y < 0 || x >= this.cols || y >= this.rows) return 1;
    return this.visit[this.idx(x, y)] || 0;
  }

  turn(action) {
    const d = this.dir;
    if (action === 1) this.dir = { x: -d.y, y: d.x };
    else if (action === 2) this.dir = { x: d.y, y: -d.x };
  }

  step(action) {
    if (!this.alive) {
      return { state: this.getState(), reward: 0, done: true, info: { crash: 'dead' } };
    }
    const R = this.reward;
    this.turn(action);
    const head = this.snake[0];
    const nx = head.x + this.dir.x;
    const ny = head.y + this.dir.y;
    this.steps++;
    this.stepsSinceFruit++;
    const key = `${nx},${ny}`;
    const tail = this.snake[this.snake.length - 1];
    const willGrow = nx === this.fruit.x && ny === this.fruit.y;
    const survivalBonus = 0.001 * this.snake.length;
    const hitsWall = nx < 0 || ny < 0 || nx >= this.cols || ny >= this.rows;
    const hitsBody = this.snakeSet.has(key) && !(tail && tail.x === nx && tail.y === ny && !willGrow);
    if (hitsWall || hitsBody) {
      this.alive = false;
      const crashReward = (hitsWall ? -R.wallPenalty : -R.selfPenalty) + survivalBonus;
      return {
        state: this.getState(),
        reward: crashReward,
        done: true,
        info: {
          crash: hitsWall ? 'wall' : 'self',
          loopPenaltyApplied: false,
          revisitPenalty: 0,
          ateFruit: false,
          timeToFruit: null,
        },
      };
    }

    const futureSpace = this.freeSpaceFrom(nx, ny, !willGrow);
    let spaceReward = 0;
    if (R.spaceGainBonus) {
      const curSpace = this.freeSpaceFrom(this.snake[0].x, this.snake[0].y, true);
      const growthNeed = this.snake.length + 2;
      const denom = Math.max(1, growthNeed);
      if (futureSpace > curSpace) {
        spaceReward += R.spaceGainBonus * Math.min(1, (futureSpace - curSpace) / denom);
      }
    }
    if (R.trapPenalty) {
      const safeNeed = this.snake.length + 5;
      if (futureSpace < safeNeed) {
        const deficit = safeNeed - futureSpace;
        spaceReward -= R.trapPenalty * (deficit / Math.max(1, safeNeed));
      }
    }

    for (let i = 0; i < this.visit.length; i++) this.visit[i] *= 0.995;

    this.snake.unshift({ x: nx, y: ny });
    let reward = -R.stepPenalty;
    reward += spaceReward;
    let loopPenaltyApplied = false;
    if (action !== 0) reward -= R.turnPenalty;
    this.actionHist.push(action);
    if (this.actionHist.length > 6) this.actionHist.shift();
    if (this.actionHist.length >= 4) {
      const last4 = this.actionHist.slice(-4).join(',');
      if (LOOP_PATTERNS.has(last4)) {
        reward -= R.loopPenalty;
        loopPenaltyApplied = true;
        this.loopHits++;
      }
    }

    const vidx = this.idx(nx, ny);
    const revisitPenalty = this.visit[vidx] * R.revisitPenalty;
    reward -= revisitPenalty;
    this.revisitAccum += revisitPenalty;

    let ateFruit = false;
    let timeToFruit = null;
    let approachDelta = 0;
    if (nx === this.fruit.x && ny === this.fruit.y) {
      ateFruit = true;
      const fruitScale = 1 + this.snake.length * 0.05;
      reward += R.fruitReward * fruitScale;
      this.snakeSet.add(`${nx},${ny}`);
      this.spawnFruit();
      timeToFruit = this.stepsSinceFruit;
      this.timeToFruitAccum += this.stepsSinceFruit;
      this.timeToFruitCount += 1;
      this.stepsSinceFruit = 0;
      this.episodeFruit += 1;
    } else {
      const tailSegment = this.snake.pop();
      this.snakeSet.delete(`${tailSegment.x},${tailSegment.y}`);
      this.snakeSet.add(`${nx},${ny}`);
      this.visit[vidx] = Math.min(1, this.visit[vidx] + 0.3);
      const prevDist = Math.abs(head.x - this.fruit.x) + Math.abs(head.y - this.fruit.y);
      const newDist = Math.abs(nx - this.fruit.x) + Math.abs(ny - this.fruit.y);
      if (newDist < prevDist) {
        reward += R.approachBonus;
        approachDelta = 1;
      } else if (newDist > prevDist) {
        reward -= R.retreatPenalty;
        approachDelta = -1;
      }
    }

    if (this.snake.length > this.maxLength) {
      const gain = this.snake.length - this.maxLength;
      this.maxLength = this.snake.length;
      if (R.growthBonus) {
        reward += R.growthBonus * gain;
      }
    }

    const slack = this.computeSlack();
    const slackDelta = this.prevSlack - slack;
    if (R.compactWeight !== 0) {
      reward += slackDelta * R.compactWeight;
    }
    this.prevSlack = slack;

    if (this.stepsSinceFruit > this.cols * this.rows * 2) {
      this.alive = false;
      reward -= R.timeoutPenalty;
      reward += survivalBonus;
      return {
        state: this.getState(),
        reward,
        done: true,
        info: {
          crash: 'timeout',
          loopPenaltyApplied,
          revisitPenalty,
          ateFruit,
          timeToFruit,
          approachDelta,
        },
      };
    }

    reward += survivalBonus;

    return {
      state: this.getState(),
      reward,
      done: false,
      info: {
        crash: null,
        loopPenaltyApplied,
        revisitPenalty,
        ateFruit,
        timeToFruit,
        approachDelta,
      },
    };
  }

  getState() {
    const head = this.snake[0];
    const left = { x: -this.dir.y, y: this.dir.x };
    const right = { x: this.dir.y, y: -this.dir.x };
    const block = (dx, dy) => {
      const x = head.x + dx;
      const y = head.y + dy;
      return x < 0 || y < 0 || x >= this.cols || y >= this.rows || this.snakeSet.has(`${x},${y}`) ? 1 : 0;
    };
    const danger = [block(this.dir.x, this.dir.y), block(left.x, left.y), block(right.x, right.y)];
    const maxRange = Math.max(this.cols, this.rows) || 1;
    const bodyProximity = [this.dir, left, right].map((dirVec) => {
      let x = head.x;
      let y = head.y;
      let distance = 0;
      while (true) {
        x += dirVec.x;
        y += dirVec.y;
        distance += 1;
        if (x < 0 || y < 0 || x >= this.cols || y >= this.rows) return 0;
        if (this.snakeSet.has(`${x},${y}`)) {
          return 1 - Math.min(1, Math.max(0, distance - 1) / maxRange);
        }
      }
    });
    const dir = [
      this.dir.y === -1 ? 1 : 0,
      this.dir.y === 1 ? 1 : 0,
      this.dir.x === -1 ? 1 : 0,
      this.dir.x === 1 ? 1 : 0,
    ];
    const fruit = [
      this.fruit.y < head.y ? 1 : 0,
      this.fruit.y > head.y ? 1 : 0,
      this.fruit.x < head.x ? 1 : 0,
      this.fruit.x > head.x ? 1 : 0,
    ];
    const dists = [
      head.y / (this.rows - 1 || 1),
      (this.rows - 1 - head.y) / (this.rows - 1 || 1),
      head.x / (this.cols - 1 || 1),
      (this.cols - 1 - head.x) / (this.cols - 1 || 1),
    ];
    const dx = this.fruit.x - head.x;
    const dy = this.fruit.y - head.y;
    const len = Math.hypot(dx, dy) || 1;
    const crowd = [
      this.getVisit(head.x, head.y - 1),
      this.getVisit(head.x, head.y + 1),
      this.getVisit(head.x - 1, head.y),
      this.getVisit(head.x + 1, head.y),
    ];
    const normSpace = (dirVec) => {
      const tx = head.x + dirVec.x;
      const ty = head.y + dirVec.y;
      if (tx < 0 || ty < 0 || tx >= this.cols || ty >= this.rows) return 0;
      if (this.snakeSet.has(`${tx},${ty}`)) return 0;
      const willGrow = tx === this.fruit.x && ty === this.fruit.y;
      const space = this.freeSpaceFrom(tx, ty, !willGrow);
      return space / Math.max(1, this.cols * this.rows);
    };
    const spaceAhead = normSpace(this.dir);
    const spaceLeft = normSpace(left);
    const spaceRight = normSpace(right);
    const tail = this.snake[this.snake.length - 1];
    const tailPrev = this.snake[this.snake.length - 2] ?? tail;
    const tailVec = { x: tail.x - tailPrev.x, y: tail.y - tailPrev.y };
    const tailLen = Math.hypot(tailVec.x, tailVec.y) || 1;
    const tailDir = [tailVec.x / tailLen, tailVec.y / tailLen];
    const normLength = this.snake.length / Math.max(1, this.cols * this.rows);
    return Float32Array.from([
      ...danger,
      ...bodyProximity,
      ...dir,
      ...fruit,
      ...dists,
      dy / len,
      dx / len,
      ...crowd,
      spaceAhead,
      spaceLeft,
      spaceRight,
      ...tailDir,
      normLength,
    ]);
  }

  cloneRewardConfig() {
    return { ...this.reward };
  }
}

export function clampRewardConfig(config = {}) {
  return {
    ...config,
    loopPenalty: clamp(config.loopPenalty ?? REWARD_DEFAULTS.loopPenalty, 0, 1),
    compactWeight: clamp(config.compactWeight ?? REWARD_DEFAULTS.compactWeight, 0, 0.1),
    revisitPenalty: clamp(config.revisitPenalty ?? REWARD_DEFAULTS.revisitPenalty, 0, 0.1),
    selfPenalty: clamp(config.selfPenalty ?? REWARD_DEFAULTS.selfPenalty, 0, 30),
    turnPenalty: clamp(config.turnPenalty ?? REWARD_DEFAULTS.turnPenalty, 0, 0.02),
    approachBonus: clamp(config.approachBonus ?? REWARD_DEFAULTS.approachBonus, 0, 0.1),
    retreatPenalty: clamp(config.retreatPenalty ?? REWARD_DEFAULTS.retreatPenalty, 0, 0.1),
    growthBonus: clamp(config.growthBonus ?? REWARD_DEFAULTS.growthBonus, 0, 5),
  };
}
