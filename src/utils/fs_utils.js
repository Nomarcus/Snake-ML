import fs from 'fs/promises';
import path from 'path';

export async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

export async function writeJsonAtomic(targetPath, data) {
  const dir = path.dirname(targetPath);
  await ensureDir(dir);
  const tempPath = path.join(dir, `.tmp_${Date.now()}_${Math.random().toString(36).slice(2)}`);
  const json = JSON.stringify(data, null, 2);
  await fs.writeFile(tempPath, json);
  await fs.rename(tempPath, targetPath);
}

export async function writeJsonWithFsync(filePath, data) {
  const dir = path.dirname(filePath);
  await ensureDir(dir);
  const handle = await fs.open(filePath, 'w');
  try {
    await handle.writeFile(JSON.stringify(data, null, 2));
    await handle.sync();
  } finally {
    await handle.close();
  }
}

export async function writeFileAtomic(targetPath, contents) {
  const dir = path.dirname(targetPath);
  await ensureDir(dir);
  const tempPath = path.join(dir, `.tmp_${Date.now()}_${Math.random().toString(36).slice(2)}`);
  await fs.writeFile(tempPath, contents);
  await fs.rename(tempPath, targetPath);
}

export async function directorySizeMB(dirPath) {
  async function walk(current) {
    let size = 0;
    const entries = await fs.readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      const entryPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        size += await walk(entryPath);
      } else {
        const stat = await fs.stat(entryPath);
        size += stat.size;
      }
    }
    return size;
  }
  try {
    const sizeBytes = await walk(dirPath);
    return sizeBytes / (1024 * 1024);
  } catch (err) {
    if (err.code === 'ENOENT') return 0;
    throw err;
  }
}

export async function removeDir(target) {
  try {
    await fs.rm(target, { recursive: true, force: true });
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
  }
}

export async function listSubdirs(dirPath) {
  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    return entries.filter((entry) => entry.isDirectory()).map((entry) => ({
      name: entry.name,
      fullPath: path.join(dirPath, entry.name),
    }));
  } catch (err) {
    if (err.code === 'ENOENT') return [];
    throw err;
  }
}

export async function safeUnlink(filePath) {
  try {
    await fs.unlink(filePath);
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
  }
}

export async function withTempDir(baseDir, fn) {
  const tempDir = path.join(baseDir, `.tmp_${Date.now()}_${Math.random().toString(36).slice(2)}`);
  await ensureDir(tempDir);
  try {
    const result = await fn(tempDir);
    return result;
  } finally {
    await removeDir(tempDir);
  }
}

export function nowISO() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}
