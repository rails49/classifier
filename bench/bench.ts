// Time one batch of 50 patches in onnxruntime-web's WASM backend under Node.
// Usage: pnpm bench <threads> <model.onnx>...  (threads 0: onnxruntime chooses)
import { readFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import * as ort from "onnxruntime-web";

const BATCH = 50;
const WARMUP = 5;
const RUNS = 50;

const { positionals } = parseArgs({ allowPositionals: true });
const [threads, ...models] = positionals;
ort.env.wasm.numThreads = Number(threads);

for (const path of models) {
  const session = await ort.InferenceSession.create(await readFile(path), {
    executionProviders: ["wasm"],
  });
  const data = Float32Array.from({ length: BATCH * 3 * 48 * 96 }, Math.random);
  const feeds = { patches: new ort.Tensor("float32", data, [BATCH, 3, 48, 96]) };
  for (let i = 0; i < WARMUP; i++) await session.run(feeds);
  const times: number[] = [];
  for (let i = 0; i < RUNS; i++) {
    const start = performance.now();
    await session.run(feeds);
    times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  const median = (times[RUNS / 2 - 1] + times[RUNS / 2]) / 2;
  console.log(JSON.stringify({ model: path, threads: Number(threads), median_ms: median }));
  await session.release();
}
