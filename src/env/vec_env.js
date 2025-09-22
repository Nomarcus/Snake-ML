import { SnakeEnv } from './snake.js';

export class VecSnakeEnv {
  constructor(envCount = 1, { cols = 20, rows = 20, rewardConfig = {} } = {}) {
    this.envCount = Math.max(1, envCount | 0);
    this.cols = cols;
    this.rows = rows;
    this.rewardConfig = { ...rewardConfig };
    this.envs = Array.from({ length: this.envCount }, () => new SnakeEnv(cols, rows, rewardConfig));
  }

  setRewardConfig(config) {
    this.rewardConfig = { ...config };
    for (const env of this.envs) env.setRewardConfig(config);
  }

  setBoardSize(cols, rows) {
    this.cols = cols;
    this.rows = rows;
    this.envs = Array.from({ length: this.envCount }, () => new SnakeEnv(cols, rows, this.rewardConfig));
  }

  reset() {
    return this.envs.map((env) => env.reset());
  }

  resetEnv(index) {
    const env = this.envs[index];
    if (!env) throw new Error(`Env index ${index} out of range`);
    return env.reset();
  }

  step(actions) {
    if (!Array.isArray(actions) || actions.length !== this.envCount) {
      throw new Error(`Expected actions array length ${this.envCount}`);
    }
    const results = this.envs.map((env, i) => env.step(actions[i] ?? 0));
    return {
      nextStates: results.map((r) => r.state),
      rewards: results.map((r) => r.reward),
      dones: results.map((r) => r.done),
      infos: results.map((r) => r.info ?? {}),
    };
  }

  getStateDim() {
    const env = this.envs[0] ?? new SnakeEnv(this.cols, this.rows, this.rewardConfig);
    const state = env.getState();
    return state.length;
  }

  getActionDim() {
    return 3; // forward, left, right
  }
}
