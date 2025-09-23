import fs from 'fs/promises';
import path from 'path';
import { ensureDir, removeDir, withTempDir, nowISO, writeJsonWithFsync } from '../utils/fs_utils.js';

export class CheckpointManager {
  constructor({ saveDir, diskGuard, retain = 5 }) {
    this.saveDir = saveDir;
    this.diskGuard = diskGuard;
    this.retain = retain;
  }

  async init() {
    await ensureDir(this.saveDir);
  }

  async save({ agentState, rewardConfig, meta, isBest = false, reason }) {
    const timestamp = nowISO();
    const targetDir = path.join(this.saveDir, timestamp);
    await withTempDir(this.saveDir, async (tempDir) => {
      const payload = {
        createdAt: new Date().toISOString(),
        agent: agentState,
        rewardConfig,
        meta,
        reason,
        version: 1,
      };
      await writeJsonWithFsync(path.join(tempDir, 'checkpoint.json'), payload);
      await fs.rename(tempDir, targetDir);
    });
    await this._updatePointer('latest', targetDir);
    if (isBest) {
      await this._updatePointer('best', targetDir);
    }
    await this.diskGuard.pruneCheckpoints();
    return targetDir;
  }

  async _updatePointer(name, targetDir) {
    const pointerDir = path.join(this.saveDir, name);
    const tempPointer = path.join(this.saveDir, `${name}_tmp`);
    await removeDir(tempPointer);
    await ensureDir(tempPointer);
    await fs.cp(targetDir, tempPointer, { recursive: true, force: true });
    await removeDir(pointerDir);
    await fs.rename(tempPointer, pointerDir);
  }
}
