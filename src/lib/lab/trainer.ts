import * as tf from "@tensorflow/tfjs";
import { ArchSpec, ToyConfig } from "./archs";
import { ToyTransformer } from "./model";
import { TaskSpec, VOCAB, decode, encode, mulberry32, sample } from "./tasks";

export interface TrainSettings {
  lr: number;
  batchSize: number;
  steps: number;
  warmup: number;
  evalEvery: number;
  clip: number;
  seed: number;
  backend: "webgl" | "cpu";
}

export interface SampleRow {
  prompt: string;
  target: string;
  pred: string;
  ok: boolean;
}

export interface TrainUpdate {
  step: number;
  totalSteps: number;
  loss: number;
  lr: number;
  lossCurve: [number, number][];
  accCurve: { step: number; train: number; test: number }[];
  samples: SampleRow[];
  expertUsage: number[][] | null;
  attn: { tokens: string[]; maps: number[][][][]; local: boolean[] } | null;
  elapsedMs: number;
  paramCount: number;
  activeParams: number;
  done: boolean;
  backend: string;
  error?: string;
}

let backendReady: string | null = null;
async function ensureBackend(pref: "webgl" | "cpu") {
  if (backendReady === pref) return pref;
  try {
    const ok = await tf.setBackend(pref);
    if (!ok) throw new Error("backend failed");
  } catch {
    await tf.setBackend("cpu");
  }
  await tf.ready();
  backendReady = tf.getBackend();
  return backendReady;
}

function makeBatch(task: TaskSpec, rng: () => number, B: number, split: "train" | "test") {
  const L = task.promptLen + task.answerLen;
  const inp = new Int32Array(B * (L - 1));
  const tgt = new Int32Array(B * (L - 1));
  for (let b = 0; b < B; b++) {
    const ex = sample(task, rng, split);
    const ids = encode(ex.prompt + ex.answer);
    for (let t = 0; t < L - 1; t++) {
      inp[b * (L - 1) + t] = ids[t];
      tgt[b * (L - 1) + t] = ids[t + 1];
    }
  }
  const mask = new Float32Array(L - 1);
  for (let t = task.promptLen - 1; t < L - 1; t++) mask[t] = 1;
  return { inp, tgt, mask, L };
}

function greedy(model: ToyTransformer, task: TaskSpec, prompts: string[]): string[] {
  const B = prompts.length;
  const seqs = prompts.map((p) => encode(p));
  for (let i = 0; i < task.answerLen; i++) {
    const len = seqs[0].length;
    const out = tf.tidy(() => {
      const x = tf.tensor2d(seqs.flat(), [B, len], "int32");
      const logits = model.forward(x);
      return logits.slice([0, len - 1, 0], [B, 1, model.vocab]).reshape([B, model.vocab]).argMax(-1);
    });
    const ids = out.dataSync();
    out.dispose();
    for (let b = 0; b < B; b++) seqs[b].push(ids[b]);
  }
  return seqs.map((s) => decode(s.slice(task.promptLen)));
}

function evaluate(model: ToyTransformer, task: TaskSpec, seed: number, split: "train" | "test", n: number) {
  const rng = mulberry32(seed);
  const exs = Array.from({ length: n }, () => sample(task, rng, split));
  const preds = greedy(model, task, exs.map((e) => e.prompt));
  let correct = 0;
  const rows: SampleRow[] = exs.map((e, i) => {
    const ok = preds[i] === e.answer;
    if (ok) correct++;
    return { prompt: e.prompt, target: e.answer, pred: preds[i], ok };
  });
  return { acc: correct / n, rows };
}

function captureAttention(model: ToyTransformer, task: TaskSpec, seed: number) {
  const ex = sample(task, mulberry32(seed), "test");
  const full = ex.prompt + ex.answer;
  const ids = encode(full).slice(0, -1);
  tf.tidy(() => {
    model.forward(tf.tensor2d(ids, [1, ids.length], "int32"), true);
  });
  const maps = model.captured ?? [];
  model.captured = null;
  return {
    tokens: [...full.slice(0, -1)],
    maps,
    local: Array.from({ length: model.cfg.layers }, (_, l) => model.isLocal(l)),
  };
}

export async function train(
  arch: ArchSpec,
  cfg: ToyConfig,
  task: TaskSpec,
  s: TrainSettings,
  onUpdate: (u: TrainUpdate) => void,
  shouldStop: () => boolean
): Promise<TrainUpdate> {
  const backend = await ensureBackend(s.backend);
  const maxLen = task.promptLen + task.answerLen;
  const model = new ToyTransformer(arch, cfg, VOCAB.length, maxLen, s.seed);
  const vars = model.variables;
  const opt = tf.train.adam(s.lr, 0.9, 0.98, 1e-8);
  const rng = mulberry32(s.seed * 7919 + 13);
  const start = performance.now();
  const lossCurve: [number, number][] = [];
  const accCurve: { step: number; train: number; test: number }[] = [];
  let samples: SampleRow[] = [];
  let attn: TrainUpdate["attn"] = null;
  let lastLoss = NaN;
  let lr = s.lr;
  const paramCount = model.paramCount();
  const activeParams = model.activeParamCount();

  const snapshot = (step: number, done: boolean, error?: string): TrainUpdate => ({
    step,
    totalSteps: s.steps,
    loss: lastLoss,
    lr,
    lossCurve: lossCurve.slice(),
    accCurve: accCurve.slice(),
    samples,
    expertUsage: arch.moe ? model.expertUsage.map((r) => r.slice()) : null,
    attn,
    elapsedMs: performance.now() - start,
    paramCount,
    activeParams,
    done,
    backend,
    error,
  });

  const runEval = (step: number) => {
    model.training = false;
    const tr = evaluate(model, task, 1234, "train", 96);
    const te = evaluate(model, task, 5678, "test", 96);
    accCurve.push({ step, train: tr.acc, test: te.acc });
    samples = te.rows.slice(0, 10);
    attn = captureAttention(model, task, step + 99);
  };

  let step = 0;
  try {
    for (step = 1; step <= s.steps; step++) {
      if (shouldStop()) break;
      // warmup + cosine decay
      lr =
        step <= s.warmup
          ? (s.lr * step) / s.warmup
          : s.lr * (0.1 + 0.9 * 0.5 * (1 + Math.cos((Math.PI * (step - s.warmup)) / Math.max(1, s.steps - s.warmup))));
      (opt as unknown as { learningRate: number }).learningRate = lr;

      const { inp, tgt, mask, L } = makeBatch(task, rng, s.batchSize, "train");
      model.training = true;
      const lossT = tf.tidy(() => {
        const x = tf.tensor2d(inp, [s.batchSize, L - 1], "int32");
        const y = tf.tensor1d(tgt, "int32");
        const m = tf.tensor1d(mask).reshape([1, L - 1]);
        const { value, grads } = tf.variableGrads(() => {
          const logits = model.forward(x);
          const logp = tf.logSoftmax(logits);
          const oh = tf.oneHot(y, model.vocab).reshape(logits.shape);
          const nll = oh.mul(logp).sum(-1).neg(); // [B, T]
          let loss = nll.mul(m).sum().div(m.sum().mul(s.batchSize));
          if (model.auxLoss) loss = loss.add(model.auxLoss.mul(0.01 / cfg.layers));
          return loss as tf.Scalar;
        }, vars);
        // global-norm gradient clipping
        const names = Object.keys(grads);
        const sq = tf.addN(names.map((n) => grads[n].square().sum()));
        const norm = sq.sqrt().dataSync()[0];
        const scale = norm > s.clip ? s.clip / norm : 1;
        const clipped: tf.NamedTensorMap = {};
        for (const n of names) clipped[n] = grads[n].mul(scale);
        opt.applyGradients(clipped as unknown as Parameters<typeof opt.applyGradients>[0]);
        return value;
      });
      lastLoss = (await lossT.data())[0];
      lossT.dispose();
      if (!Number.isFinite(lastLoss)) throw new Error("Loss diverged (NaN). Try a lower learning rate.");
      lossCurve.push([step, lastLoss]);

      if (step % s.evalEvery === 0 || step === s.steps) runEval(step);
      if (step % 5 === 0 || step === s.steps) {
        onUpdate(snapshot(step, false));
        await tf.nextFrame();
      }
    }
    if (accCurve.length === 0 || accCurve[accCurve.length - 1].step !== Math.min(step, s.steps)) {
      runEval(Math.min(step, s.steps));
    }
    const final = snapshot(Math.min(step, s.steps), true);
    onUpdate(final);
    return final;
  } catch (e) {
    const final = snapshot(step, true, e instanceof Error ? e.message : String(e));
    onUpdate(final);
    return final;
  } finally {
    model.dispose();
    opt.dispose();
  }
}
