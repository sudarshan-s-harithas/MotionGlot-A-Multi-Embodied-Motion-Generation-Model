import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs";
import { MotionTokenizer } from "./tokenizer.js?v=human1";
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
const sessions = new Map();
let tokenizer,
  config,
  modelBase,
  busy = false;
const notify = (message, progress = null) =>
  postMessage({ type: "status", message, progress });
async function bytes(name) {
  const url = modelBase + name;
  let cache;
  try {
    cache = await caches.open("motionglot-models-v1");
    const hit = await cache.match(url);
    if (hit) {
      notify(`Loading cached ${name}…`);
      return new Uint8Array(await hit.arrayBuffer());
    }
  } catch {}
  notify(`Downloading ${name}…`);
  const response = await fetch(url);
  if (!response.ok)
    throw Error(`Cannot download ${name} (HTTP ${response.status}).`);
  const total = Number(response.headers.get("content-length")) || 0,
    reader = response.body.getReader(),
    chunks = [];
  let read = 0,
    last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    read += value.length;
    if (Date.now() - last > 300) {
      notify(
        `Downloading ${name}: ${(read / 1e6).toFixed(0)} MB${total ? ` / ${(total / 1e6).toFixed(0)} MB` : ""}`,
        total ? read / total : null,
      );
      last = Date.now();
    }
  }
  const out = new Uint8Array(read);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  chunks.length = 0;
  try {
    await cache?.put(
      url,
      new Response(out, {
        headers: { "Content-Type": "application/octet-stream" },
      }),
    );
  } catch {}
  return out;
}
async function session(name) {
  if (!sessions.has(name)) {
    const data = await bytes(name);
    notify(`Preparing ${name}…`);
    sessions.set(
      name,
      await ort.InferenceSession.create(data, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }),
    );
  }
  return sessions.get(name);
}
function integers(values, dims) {
  return new ort.Tensor("int64", BigInt64Array.from(values, BigInt), dims);
}
function rows(tensor) {
  const [batch, n, d] = tensor.dims;
  return Array.from({ length: n }, (_, i) =>
    Array.from(tensor.data.slice(i * d, (i + 1) * d)),
  );
}
async function setup() {
  if (config) return;
  const settings = await (await fetch("./settings.json")).json();
  modelBase = settings.modelBase;
  config = JSON.parse(
    new TextDecoder().decode(await bytes("browser-config.json")),
  );
  tokenizer = new MotionTokenizer(
    JSON.parse(new TextDecoder().decode(await bytes("tokenizer.json"))),
    config.tokens,
  );
}
function random(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function choose(logits, allowed, sample, rng) {
  if (!sample)
    return allowed.reduce(
      (best, id) => (logits[id] > logits[best] ? id : best),
      allowed[0],
    );
  const sorted = allowed
      .map((id) => [id, logits[id] / 0.8])
      .sort((a, b) => b[1] - a[1]),
    max = sorted[0][1],
    weights = sorted.map((x) => Math.exp(x[1] - max)),
    sum = weights.reduce((a, b) => a + b, 0);
  let partial = 0,
    cut = weights.length;
  for (let i = 0; i < weights.length; i++) {
    partial += weights[i];
    if (partial / sum >= 0.9) {
      cut = i + 1;
      break;
    }
  }
  const keep = weights.slice(0, cut),
    total = keep.reduce((a, b) => a + b, 0);
  let draw = rng() * total;
  for (let i = 0; i < cut; i++) {
    draw -= keep[i];
    if (draw <= 0) return sorted[i][0];
  }
  return sorted[cut - 1][0];
}
async function generate(request) {
  if (
    !["text-to-human", "question-to-human", "human-to-text"].includes(
      request.task,
    )
  )
    throw Error("This demo supports human-motion tasks only.");
  await setup();
  const spec = config.tasks[request.task],
    tok = config.tokens;
  let source = request.text,
    features = request.features,
    embodiment = "human";
  if (spec.output === "text") {
    const dim = 263,
      input = features
        .flat()
        .map((v, i) => (v - config.mean[i % dim]) / config.std[i % dim]);
    const encoder = await session(`${embodiment}-encoder.onnx`);
    const encoded = await encoder.run({
      values: new ort.Tensor("float32", Float32Array.from(input), [
        1,
        features.length,
        dim,
      ]),
    });
    source = Array.from(
      encoded.output.data,
      (x) => `<${embodiment}_motion_id_${Number(x)}>`,
    ).join("");
    encoded.output.dispose();
  }
  if (!source?.trim()) throw Error("Enter a prompt.");
  const prompt =
      spec.prefix +
      source +
      "<eop>" +
      spec.start +
      (spec.output === "text" ? "" : " \n "),
    ids = tokenizer.encode(prompt),
    max = request.maxTokens;
  if (ids.length + max > config.context)
    throw Error(
      "Input and requested output exceed the 1024-token context. Use a shorter motion or token limit.",
    );
  const gpt = await session("gpt.onnx"),
    end = tok[spec.end],
    motionIds =
      spec.output === "text"
        ? null
        : Array.from(
            { length: 512 },
            (_, i) => tok[`<${spec.output}_motion_id_${i}>`],
          ),
    allowed = motionIds
      ? [...motionIds, end]
      : Array.from({ length: 50257 }, (_, i) => i).concat(end),
    cache = {},
    generated = [],
    rng = random(request.seed || 42);
  for (let i = 0; i < 12; i++)
    for (const kind of ["key", "value"])
      cache[`past_${i}_${kind}`] = new ort.Tensor(
        "float32",
        new Float32Array(0),
        [1, 12, 0, 64],
      );
  let current = ids,
    pastLength = 0;
  try {
    for (let step = 0; step < max; step++) {
      notify(`Generating ${request.task}: ${step + 1} / ${max} tokens…`);
      const input = integers(current, [1, current.length]),
        mask = integers(Array(pastLength + current.length).fill(1), [
          1,
          pastLength + current.length,
        ]),
        position = integers(
          current.map((_, i) => pastLength + i),
          [1, current.length],
        );
      const output = await gpt.run({
        input_ids: input,
        attention_mask: mask,
        position_ids: position,
        ...cache,
      });
      input.dispose();
      mask.dispose();
      position.dispose();
      let active = allowed;
      if (motionIds && step === 0) active = active.filter((id) => id !== end);
      const next = choose(output.logits.data, active, request.sample, rng);
      output.logits.dispose();
      generated.push(next);
      for (let i = 0; i < 12; i++)
        for (const kind of ["key", "value"]) {
          const key = `past_${i}_${kind}`;
          cache[key].dispose();
          cache[key] = output[`present_${i}_${kind}`];
        }
      pastLength += current.length;
      current = [next];
      if (next === end) break;
    }
  } finally {
    for (const tensor of Object.values(cache)) tensor.dispose();
  }
  const result = {
    task: request.task,
    terminated: generated.at(-1) === end,
    tokens: tokenizer.decode(generated),
    source:
      spec.output === "text"
        ? `Human motion · ${features.length} frames`
        : request.text,
    origin: "generated",
  };
  if (spec.output === "text") {
    result.caption = tokenizer.decode(generated.filter((id) => id !== end));
    result.features = features;
    result.embodiment = embodiment;
  } else {
    const codeMap = new Map(motionIds.map((id, i) => [id, i])),
      codes = generated
        .filter((id) => codeMap.has(id))
        .map((id) => codeMap.get(id));
    if (!codes.length) throw Error("No motion codes were generated.");
    const decoder = await session(`${spec.output}-decoder.onnx`),
      values = integers(codes, [1, codes.length]),
      out = await decoder.run({ values });
    values.dispose();
    features = rows(out.output);
    out.output.dispose();
    if (spec.output === "human")
      features = features.map((row) =>
        row.map((v, i) => v * config.std[i] + config.mean[i]),
      );
    Object.assign(result, { features, embodiment: "human" });
  }
  postMessage({ type: "result", result });
}
self.onmessage = async ({ data }) => {
  if (busy) return;
  busy = true;
  try {
    await generate(data);
  } catch (error) {
    postMessage({ type: "error", message: error.message || String(error) });
  } finally {
    busy = false;
  }
};
