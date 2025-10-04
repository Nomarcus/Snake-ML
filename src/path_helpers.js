export function bfsPath(gridSize, snake = [], fruit) {
  const size = Number.isFinite(gridSize) ? gridSize | 0 : 0;
  if (size <= 0 || !fruit || typeof fruit.x !== 'number' || typeof fruit.y !== 'number') {
    return [];
  }
  if (!Array.isArray(snake) || snake.length === 0) {
    return [];
  }
  const head = snake[0];
  if (typeof head?.x !== 'number' || typeof head?.y !== 'number') {
    return [];
  }
  const dirs = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ];
  const key = (x, y) => `${x},${y}`;
  const blocked = new Set();
  for (const segment of snake) {
    if (segment && Number.isFinite(segment.x) && Number.isFinite(segment.y)) {
      blocked.add(key(segment.x, segment.y));
    }
  }
  const visited = new Set();
  const prev = new Map();
  const queue = [{ x: head.x, y: head.y }];
  let qIndex = 0;
  visited.add(key(head.x, head.y));

  while (qIndex < queue.length) {
    const current = queue[qIndex++];
    const currentKey = key(current.x, current.y);
    if (current.x === fruit.x && current.y === fruit.y) {
      const path = [{ x: current.x, y: current.y }];
      let cursorKey = currentKey;
      while (prev.has(cursorKey)) {
        const parent = prev.get(cursorKey);
        cursorKey = key(parent.x, parent.y);
        path.unshift({ x: parent.x, y: parent.y });
      }
      return path;
    }
    for (const [dx, dy] of dirs) {
      const nx = current.x + dx;
      const ny = current.y + dy;
      if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
      const nextKey = key(nx, ny);
      if (visited.has(nextKey)) continue;
      if (blocked.has(nextKey) && !(nx === fruit.x && ny === fruit.y)) continue;
      visited.add(nextKey);
      prev.set(nextKey, { x: current.x, y: current.y });
      queue.push({ x: nx, y: ny });
    }
  }
  return [];
}

export function bfsDistance(gridSize, snake = [], fruit) {
  const path = bfsPath(gridSize, snake, fruit);
  return path.length ? path.length - 1 : -1;
}

export function generateHamiltonCycle(size) {
  const n = Number.isFinite(size) ? size | 0 : 0;
  if (n < 2 || n % 2 !== 0) {
    return [];
  }
  if (n === 2) {
    return [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ];
  }
  const cycle = [];
  for (let x = 0; x < n; x++) {
    cycle.push({ x, y: 0 });
  }
  for (let y = 1; y < n; y++) {
    cycle.push({ x: n - 1, y });
  }
  for (let x = n - 2; x >= 0; x--) {
    cycle.push({ x, y: n - 1 });
  }
  for (let y = n - 2; y >= 2; y--) {
    cycle.push({ x: 0, y });
  }
  const inner = generateHamiltonCycle(n - 2);
  for (let i = inner.length - 1; i >= 0; i--) {
    const point = inner[i];
    cycle.push({ x: point.x + 1, y: point.y + 1 });
  }
  cycle.push({ x: 0, y: 1 });
  return cycle;
}
