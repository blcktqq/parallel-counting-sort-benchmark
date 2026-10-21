import { Worker, isMainThread, parentPort } from 'node:worker_threads';
import { readFileSync, writeFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';

// Integer keys are int32. Float64 counters are exact for n <= 2^53 - 1.
function histogram(a, lo, k) {
  const h = new Float64Array(k);
  for (let i = 0; i < a.length; i++) {
    const j = a[i] - lo;
    if (j < 0 || j >= k) throw new RangeError('Key out of range');
    h[j]++;
  }
  return h;
}
function restore(h, lo) {
  let n = 0;
  for (const count of h) n += count;
  const out = new Int32Array(n);
  let pos = 0;
  for (let v = 0; v < h.length; v++) {
    const stop = pos + h[v];
    out.fill(lo + v, pos, stop);
    pos = stop;
  }
  return out;
}
function sequential(a, lo, k) {
  return restore(histogram(a, lo, k), lo);
}
function equal(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

class Pool {
  static async create(p) {
    const pool = new Pool();
    pool.workers = [];
    pool.pending = new Map();
    pool.nextId = 0;
    pool.closing = false;
    const ready = [];
    for (let t = 0; t < p; t++) {
      const worker = new Worker(new URL(import.meta.url));
      pool.workers.push(worker);
      ready.push(new Promise((resolve, reject) => {
        worker.on('message', message => {
          if (message.ready) { resolve(); return; }
          const task = pool.pending.get(message.id);
          if (!task) return;
          pool.pending.delete(message.id);
          if (message.error) task.reject(new Error(message.error));
          else task.resolve(message.histogram);
        });
        worker.on('error', error => {
          reject(error);
          for (const [id, task] of pool.pending) {
            if (task.worker === worker) { task.reject(error); pool.pending.delete(id); }
          }
        });
        worker.on('exit', code => {
          if (pool.closing) return;
          const error = new Error(`Worker exited unexpectedly: ${code}`);
          reject(error);
          for (const [id, task] of pool.pending) {
            if (task.worker === worker) { task.reject(error); pool.pending.delete(id); }
          }
        });
      }));
    }
    try { await Promise.all(ready); }
    catch (error) { await pool.close(); throw error; }
    return pool;
  }
  count(t, block, lo, k) {
    const worker = this.workers[t];
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, worker });
      // Transfer ownership of the already copied block, not a SharedArrayBuffer.
      worker.postMessage({ id, block, lo, k }, [block.buffer]);
    });
  }
  async close() {
    this.closing = true;
    await Promise.all(this.workers.map(worker => worker.terminate()));
  }
}

async function parallel(a, lo, k, p, pool) {
  const blocks = [];
  for (let t = 0; t < p; t++) {
    blocks.push(a.slice(Math.floor(a.length * t / p), Math.floor(a.length * (t + 1) / p)));
  }
  const partial = await Promise.all(blocks.map((block, t) => pool.count(t, block, lo, k)));
  const total = new Float64Array(k);
  for (const h of partial) for (let v = 0; v < k; v++) total[v] += h[v];
  return restore(total, lo);
}
async function cold(a, lo, k, p) {
  const pool = await Pool.create(p);
  try { return await parallel(a, lo, k, p, pool); }
  finally { await pool.close(); }
}
async function selfTest() {
  const pool = await Pool.create(3);
  let checks = 0, coldChecks = 0;
  try {
    for (const n of [0, 1, 2, 7, 100]) for (const k of [1, 7, 256]) {
      const lo = -Math.trunc(k / 2);
      let a = Int32Array.from({ length: n }, (_, i) => lo + (i * 37 + 11) % k);
      for (let shape = 0; shape < 4; shape++) {
        const expected = a.slice().sort();
        if (!equal(sequential(a, lo, k), expected) ||
            !equal(await parallel(a, lo, k, 3, pool), expected)) throw new Error('Self-test failed');
        checks++;
        if (k === 7 && shape === 0 && [0, 2, 100].includes(n)) {
          if (!equal(await cold(a, lo, k, 3), expected)) throw new Error('Cold test failed');
          coldChecks++;
        }
        if (shape === 0) a.sort();
        if (shape === 1) a.reverse();
        if (shape === 2) a.fill(lo);
      }
    }
  } finally { await pool.close(); }
  console.log(`Node.js PASS: ${checks} datasets x 2 modes, ${coldChecks} cold tests`);
}
function shuffle(a, state) {
  for (let i = a.length - 1; i > 0; i--) {
    state.value = (Math.imul(state.value, 1664525) + 1013904223) >>> 0;
    const j = state.value % (i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
}

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 1 && args[0] === '--self-test') { await selfTest(); return; }
  if (args.length !== 6) throw new Error('Usage: INPUT LO K REPEATS OUTPUT_CSV P_LIST');
  const [input, loText, kText, repeatText, output, pText] = args;
  const lo = Number(loText), k = Number(kText), repeats = Number(repeatText);
  const ps = pText.split(',').map(Number);
  const bytes = readFileSync(input);
  if (bytes.length % 4) throw new Error('Invalid i32 file');
  const a = new Int32Array(bytes.length / 4);
  for (let i = 0; i < a.length; i++) a[i] = bytes.readInt32LE(i * 4);
  const expected = a.slice().sort();
  const pools = new Map();
  try {
    for (const p of ps) pools.set(p, await Pool.create(p));
    const configs = [{ mode: 'seq', p: 1 }];
    for (const p of ps) for (const mode of ['warm', 'cold']) configs.push({ mode, p });
    const run = config => {
      if (config.mode === 'seq') return sequential(a, lo, k);
      if (config.mode === 'warm') return parallel(a, lo, k, config.p, pools.get(config.p));
      return cold(a, lo, k, config.p);
    };
    // Warm steady-state JS/V8 paths. A fresh cold worker naturally starts afresh.
    for (const config of configs) {
      const warmups = config.mode === 'cold' ? 1 : 10;
      for (let j = 0; j < warmups; j++) {
        if (!equal(await run(config), expected)) throw new Error('Warm-up mismatch');
      }
    }
    const rows = ['language,mode,p,repeat,ms'];
    const state = { value: 42 };
    for (let rep = 1; rep <= repeats; rep++) {
      shuffle(configs, state);
      for (const config of configs) {
        let result;
        const start = performance.now();
        if (config.mode === 'seq') result = sequential(a, lo, k);
        else result = await run(config);
        const ms = performance.now() - start;
        if (!equal(result, expected)) throw new Error('Result mismatch');
        rows.push(`Node.js,${config.mode},${config.p},${rep},${ms.toFixed(12)}`);
      }
    }
    writeFileSync(output, rows.join('\n') + '\n');
  } finally { await Promise.all([...pools.values()].map(pool => pool.close())); }
}

if (isMainThread) {
  main().catch(error => { console.error(error); process.exitCode = 1; });
} else {
  parentPort.on('message', ({ id, block, lo, k }) => {
    try {
      const h = histogram(block, lo, k);
      parentPort.postMessage({ id, histogram: h }, [h.buffer]);
    } catch (error) { parentPort.postMessage({ id, error: error.message }); }
  });
  parentPort.postMessage({ ready: true });
}
