import fs from 'fs/promises';
import path from 'path';
import { createReadStream, createWriteStream } from 'fs';
import zlib from 'zlib';
import {
  directorySizeMB,
  ensureDir,
  listSubdirs,
  removeDir,
  safeUnlink,
} from './utils/fs_utils.js';

const LOG_ROTATE_BYTES = 10 * 1024 * 1024;
const HEAVY_TASK_COOLDOWN_MS = 30 * 1000;

export class DiskGuard {
  constructor({
    saveDir,
    retainCheckpoints = 5,
    saveCapMB = 1024,
    logDir,
    retainLogs = 3,
    logCapMB = 200,
  }) {
    this.saveDir = saveDir;
    this.retainCheckpoints = retainCheckpoints;
    this.saveCapMB = saveCapMB;
    this.logDir = logDir ?? path.join(saveDir, '..', 'logs');
    this.retainLogs = retainLogs;
    this.logCapMB = logCapMB;
    this._lastPrune = 0;
    this._lastLogPrune = 0;
  }

  async init() {
    await ensureDir(this.saveDir);
    await ensureDir(this.logDir);
  }

  async dirSizeMB(target) {
    return directorySizeMB(target);
  }

  async _sortCheckpoints() {
    const entries = await listSubdirs(this.saveDir);
    const checkpoints = [];
    for (const entry of entries) {
      if (['latest', 'best', 'tmp'].includes(entry.name)) continue;
      const stat = await fs.stat(entry.fullPath);
      checkpoints.push({ name: entry.name, path: entry.fullPath, mtime: stat.mtimeMs });
    }
    checkpoints.sort((a, b) => a.mtime - b.mtime);
    return checkpoints;
  }

  async pruneCheckpoints() {
    const now = Date.now();
    if (now - this._lastPrune < HEAVY_TASK_COOLDOWN_MS) return;
    this._lastPrune = now;

    const checkpoints = await this._sortCheckpoints();
    const toRemove = checkpoints.length - this.retainCheckpoints;
    if (toRemove > 0) {
      const victims = checkpoints.slice(0, toRemove);
      for (const victim of victims) {
        await removeDir(victim.path);
      }
    }

    let size = await directorySizeMB(this.saveDir);
    if (size <= this.saveCapMB) return;
    for (const checkpoint of checkpoints) {
      if (size <= this.saveCapMB) break;
      await removeDir(checkpoint.path);
      size = await directorySizeMB(this.saveDir);
    }
  }

  async rotateLogs(baseName = 'training_log.jsonl') {
    const now = Date.now();
    if (now - this._lastLogPrune < HEAVY_TASK_COOLDOWN_MS) return;
    this._lastLogPrune = now;

    const logPath = path.join(this.logDir, baseName);
    let stat;
    try {
      stat = await fs.stat(logPath);
    } catch (err) {
      if (err.code === 'ENOENT') return;
      throw err;
    }
    if (stat.size < LOG_ROTATE_BYTES) return;

    const entries = await fs.readdir(this.logDir);
    const gzipped = entries
      .filter((name) => name.startsWith(baseName.replace('.jsonl', '')) && name.endsWith('.gz'))
      .sort();
    const lastIndex = gzipped.length
      ? parseInt(gzipped[gzipped.length - 1].match(/_(\d+)\.jsonl\.gz$/)?.[1] ?? '0', 10)
      : 0;
    const nextIndex = lastIndex + 1;
    const targetName = `${baseName.replace('.jsonl', '')}_${nextIndex}.jsonl.gz`;
    const targetPath = path.join(this.logDir, targetName);

    await new Promise((resolve, reject) => {
      const source = createReadStream(logPath);
      const gzip = zlib.createGzip();
      const dest = createWriteStream(targetPath);
      source.on('error', reject);
      dest.on('error', reject);
      dest.on('finish', resolve);
      source.pipe(gzip).pipe(dest);
    });
    await safeUnlink(logPath);
    await fs.writeFile(logPath, '');

    const updated = await fs.readdir(this.logDir);
    const gzFiles = updated
      .filter((name) => name.endsWith('.jsonl.gz'))
      .map((name) => ({ name, path: path.join(this.logDir, name) }))
      .sort((a, b) => a.name.localeCompare(b.name));
    if (gzFiles.length > this.retainLogs) {
      const victims = gzFiles.slice(0, gzFiles.length - this.retainLogs);
      for (const victim of victims) {
        await safeUnlink(victim.path);
      }
    }

    let totalSize = await directorySizeMB(this.logDir);
    if (totalSize <= this.logCapMB) return;
    const remaining = await fs.readdir(this.logDir);
    const gzSorted = remaining
      .filter((name) => name.endsWith('.jsonl.gz'))
      .map((name) => ({ name, path: path.join(this.logDir, name) }))
      .sort((a, b) => a.name.localeCompare(b.name));
    for (const file of gzSorted) {
      if (totalSize <= this.logCapMB) break;
      await safeUnlink(file.path);
      totalSize = await directorySizeMB(this.logDir);
    }
  }
}
