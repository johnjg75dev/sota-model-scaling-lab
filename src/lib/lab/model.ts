import * as tf from "@tensorflow/tfjs";
import { ArchSpec, ToyConfig, toyKvHeads } from "./archs";

type T = tf.Tensor;

function gelu(x: T): T {
  // tanh approximation
  const c = Math.sqrt(2 / Math.PI);
  return x.mul(0.5).mul(tf.tanh(x.add(x.pow(3).mul(0.044715)).mul(c)).add(1));
}
function silu(x: T): T {
  return x.mul(tf.sigmoid(x));
}

export class ToyTransformer {
  params: Record<string, tf.Variable> = {};
  readonly D: number;
  readonly H: number;
  readonly Hkv: number;
  readonly hd: number;
  readonly ffn: number;
  readonly expertFfn: number;
  readonly latent: number;
  training = false;
  expertUsage: number[][] = [];
  auxLoss: T | null = null;
  captured: number[][][][] | null = null; // [layer][head][T][T]
  private seed: number;
  private cache = new Map<string, T>();

  constructor(
    public arch: ArchSpec,
    public cfg: ToyConfig,
    public vocab: number,
    public maxLen: number,
    seed = 1
  ) {
    this.seed = seed * 1000;
    this.D = cfg.dModel;
    this.H = cfg.heads;
    this.Hkv = arch.mla ? cfg.heads : toyKvHeads(arch, cfg.heads);
    this.hd = Math.floor(cfg.dModel / cfg.heads);
    if (this.hd % 2 !== 0) throw new Error("head dim must be even for RoPE");
    this.ffn = Math.max(8, Math.round(cfg.dModel * arch.ffnMult));
    this.expertFfn = arch.moe ? Math.max(8, Math.round(cfg.dModel * arch.moe.expertMult)) : 0;
    this.latent = arch.mla ? Math.max(8, Math.round(cfg.dModel * arch.mla.latentRatio)) : 0;
    this.build();
    this.expertUsage = Array.from({ length: cfg.layers }, () => new Array(arch.moe?.experts ?? 0).fill(0));
  }

  private w(name: string, shape: number[], std = 0.02) {
    this.params[name] = tf.variable(tf.randomNormal(shape, 0, std, "float32", this.seed++), true, name + "_" + this.seed);
  }
  private ones(name: string, n: number) {
    this.params[name] = tf.variable(tf.ones([n]), true, name + "_" + this.seed++);
  }
  private zeros(name: string, shape: number[]) {
    this.params[name] = tf.variable(tf.zeros(shape), true, name + "_" + this.seed++);
  }
  private normParams(name: string, n = this.D) {
    this.ones(name + ".w", n);
    if (this.arch.norm === "layernorm") this.zeros(name + ".b", [n]);
  }
  private mlpParams(p: string, hidden: number) {
    const { D } = this;
    const outStd = 0.02 / Math.sqrt(2 * this.cfg.layers);
    if (this.arch.mlp === "gelu") {
      this.w(p + "fc", [D, hidden]);
      this.w(p + "proj", [hidden, D], outStd);
      if (this.arch.qkvBias) {
        this.zeros(p + "fcb", [hidden]);
        this.zeros(p + "projb", [D]);
      }
    } else {
      this.w(p + "wg", [D, hidden]);
      this.w(p + "wu", [D, hidden]);
      this.w(p + "wd", [hidden, D], outStd);
    }
  }

  private build() {
    const { D, H, Hkv, hd, arch, vocab } = this;
    const outStd = 0.02 / Math.sqrt(2 * this.cfg.layers);
    this.w("wte", [vocab, D]);
    if (arch.pos === "learned") this.w("wpe", [this.maxLen, D], 0.01);
    for (let l = 0; l < this.cfg.layers; l++) {
      const p = `h${l}.`;
      this.normParams(p + "ln1");
      this.normParams(p + "ln2");
      if (arch.sandwichNorm) {
        this.normParams(p + "postAttn");
        this.normParams(p + "postMlp");
      }
      this.w(p + "wq", [D, H * hd]);
      if (arch.mla) {
        this.w(p + "wdkv", [D, this.latent]);
        this.ones(p + "kvn.w", this.latent);
        this.w(p + "wuk", [this.latent, Hkv * hd]);
        this.w(p + "wuv", [this.latent, Hkv * hd]);
      } else {
        this.w(p + "wk", [D, Hkv * hd]);
        this.w(p + "wv", [D, Hkv * hd]);
      }
      if (arch.qkvBias) {
        this.zeros(p + "bq", [H * hd]);
        if (!arch.mla) {
          this.zeros(p + "bk", [Hkv * hd]);
          this.zeros(p + "bv", [Hkv * hd]);
        }
        this.zeros(p + "bo", [D]);
      }
      this.w(p + "wo", [H * hd, D], outStd);
      if (arch.qkNorm) {
        this.ones(p + "qn.w", hd);
        this.ones(p + "kn.w", hd);
      }
      if (arch.sinks) this.zeros(p + "sink", [H]);
      if (arch.moe) {
        this.w(p + "router", [D, arch.moe.experts]);
        for (let e = 0; e < arch.moe.experts; e++) this.mlpParams(`${p}e${e}.`, this.expertFfn);
        for (let s = 0; s < arch.moe.shared; s++) this.mlpParams(`${p}s${s}.`, this.expertFfn);
      } else {
        this.mlpParams(p + "mlp.", this.ffn);
      }
    }
    this.normParams("lnf");
    if (!arch.tied) this.w("lmHead", [D, vocab]);
  }

  get variables(): tf.Variable[] {
    return Object.values(this.params);
  }

  paramCount(): number {
    return this.variables.reduce((s, v) => s + v.size, 0);
  }

  activeParamCount(): number {
    const total = this.paramCount();
    const moe = this.arch.moe;
    if (!moe) return total;
    const per = this.params["h0.e0.wg"]
      ? this.params["h0.e0.wg"].size + this.params["h0.e0.wu"].size + this.params["h0.e0.wd"].size
      : 0;
    return total - (moe.experts - moe.topK) * per * this.cfg.layers;
  }

  dispose() {
    this.variables.forEach((v) => v.dispose());
    this.cache.forEach((t) => t.dispose());
    this.cache.clear();
    this.auxLoss = null;
  }

  // ---------- building blocks ----------
  private linear(x: T, w: string, b?: string): T {
    const shape = x.shape as number[];
    const W = this.params[w];
    let y = tf.matMul(x.reshape([-1, shape[shape.length - 1]]), W);
    if (b && this.params[b]) y = y.add(this.params[b]);
    return y.reshape([...shape.slice(0, -1), W.shape[1] as number]);
  }

  private norm(x: T, name: string, forceRms = false): T {
    const w = this.params[name + ".w"];
    if (this.arch.norm === "layernorm" && !forceRms) {
      const { mean, variance } = tf.moments(x, -1, true);
      return x.sub(mean).mul(tf.rsqrt(variance.add(1e-5))).mul(w).add(this.params[name + ".b"]);
    }
    return x.mul(tf.rsqrt(x.square().mean(-1, true).add(1e-6))).mul(w);
  }

  private cached(key: string, make: () => T): T {
    let t = this.cache.get(key);
    if (!t) {
      t = tf.keep(make());
      this.cache.set(key, t);
    }
    return t;
  }

  private ropeTables(Tn: number): [T, T] {
    const half = this.hd / 2;
    const build = (fn: (a: number) => number) => {
      const arr = new Float32Array(Tn * half);
      for (let t = 0; t < Tn; t++)
        for (let i = 0; i < half; i++) {
          const inv = Math.pow(this.arch.ropeTheta, -(2 * i) / this.hd);
          arr[t * half + i] = fn(t * inv);
        }
      return tf.tensor2d(arr, [Tn, half]);
    };
    return [this.cached(`cos${Tn}`, () => build(Math.cos)), this.cached(`sin${Tn}`, () => build(Math.sin))];
  }

  private rope(x: T, Tn: number): T {
    const half = this.hd / 2;
    const [cos, sin] = this.ropeTables(Tn);
    const x1 = x.slice([0, 0, 0, 0], [-1, -1, -1, half]);
    const x2 = x.slice([0, 0, 0, half], [-1, -1, -1, half]);
    return tf.concat([x1.mul(cos).sub(x2.mul(sin)), x1.mul(sin).add(x2.mul(cos))], 3);
  }

  private mask(Tn: number, window: number | null): T {
    return this.cached(`mask${Tn}_${window}`, () => {
      const arr = new Float32Array(Tn * Tn);
      for (let i = 0; i < Tn; i++)
        for (let j = 0; j < Tn; j++) {
          const ok = j <= i && (window === null || i - j < window);
          arr[i * Tn + j] = ok ? 0 : -1e9;
        }
      return tf.tensor2d(arr, [Tn, Tn]);
    });
  }

  isLocal(l: number): boolean {
    const w = this.arch.window;
    if (!w) return false;
    return w.pattern === "all" || l % 2 === 0;
  }

  private attention(h: T, l: number, B: number, Tn: number, capture: boolean): T {
    const { H, Hkv, hd, arch } = this;
    const p = `h${l}.`;
    let q = this.linear(h, p + "wq", p + "bq").reshape([B, Tn, H, hd]).transpose([0, 2, 1, 3]);
    let k: T, v: T;
    if (arch.mla) {
      const c = this.norm(this.linear(h, p + "wdkv"), p + "kvn", true);
      k = this.linear(c, p + "wuk");
      v = this.linear(c, p + "wuv");
    } else {
      k = this.linear(h, p + "wk", p + "bk");
      v = this.linear(h, p + "wv", p + "bv");
    }
    k = k.reshape([B, Tn, Hkv, hd]).transpose([0, 2, 1, 3]);
    v = v.reshape([B, Tn, Hkv, hd]).transpose([0, 2, 1, 3]);
    if (arch.qkNorm) {
      q = this.norm(q, p + "qn", true);
      k = this.norm(k, p + "kn", true);
    }
    if (arch.pos === "rope") {
      q = this.rope(q, Tn);
      k = this.rope(k, Tn);
    }
    if (Hkv < H) {
      const g = H / Hkv;
      const rep = (t: T) => t.reshape([B, Hkv, 1, Tn * hd]).tile([1, 1, g, 1]).reshape([B, H, Tn, hd]);
      k = rep(k);
      v = rep(v);
    }
    let scores = tf.matMul(q, k, false, true).mul(1 / Math.sqrt(hd));
    scores = scores.add(this.mask(Tn, this.isLocal(l) ? arch.window!.size : null));
    let probs: T;
    if (arch.sinks) {
      const s = this.params[p + "sink"].reshape([1, H, 1, 1]).tile([B, 1, Tn, 1]);
      probs = tf.softmax(tf.concat([scores, s], 3), -1).slice([0, 0, 0, 0], [B, H, Tn, Tn]);
    } else {
      probs = tf.softmax(scores, -1);
    }
    if (capture && this.captured) {
      this.captured[l] = (probs.slice([0, 0, 0, 0], [1, H, Tn, Tn]).arraySync() as number[][][][])[0];
    }
    const out = tf.matMul(probs, v).transpose([0, 2, 1, 3]).reshape([B, Tn, H * hd]);
    return this.linear(out, p + "wo", p + "bo");
  }

  private mlp(x: T, p: string): T {
    if (this.arch.mlp === "gelu") {
      return this.linear(gelu(this.linear(x, p + "fc", p + "fcb")), p + "proj", p + "projb");
    }
    const act = this.arch.mlp === "swiglu" ? silu : gelu;
    return this.linear(act(this.linear(x, p + "wg")).mul(this.linear(x, p + "wu")), p + "wd");
  }

  private moeBlock(x: T, l: number, B: number, Tn: number): T {
    const moe = this.arch.moe!;
    const p = `h${l}.`;
    const N = B * Tn;
    const E = moe.experts;
    const flat = x.reshape([N, this.D]);
    const logits = this.linear(flat, p + "router"); // [N,E]
    // Top-k selection is non-differentiable: compute the mask outside the tape.
    const lv = logits.dataSync();
    const m = new Float32Array(N * E);
    for (let n = 0; n < N; n++) {
      const idx = Array.from({ length: E }, (_, e) => e).sort((a, b) => lv[n * E + b] - lv[n * E + a]);
      for (let k = 0; k < moe.topK; k++) {
        m[n * E + idx[k]] = 1;
        if (this.training) this.expertUsage[l][idx[k]]++;
      }
    }
    const maskT = tf.tensor2d(m, [N, E]);
    const gates = tf.softmax(logits.add(maskT.sub(1).mul(1e9)), -1);
    // Switch-style load-balancing auxiliary loss
    if (this.training) {
      const probs = tf.softmax(logits, -1).mean(0); // [E]
      const frac = maskT.mean(0).div(moe.topK); // [E]
      const aux = probs.mul(frac).sum().mul(E);
      this.auxLoss = this.auxLoss ? this.auxLoss.add(aux) : aux;
    }
    let out: T = tf.zerosLike(flat);
    for (let e = 0; e < E; e++) {
      out = out.add(this.mlp(flat, `${p}e${e}.`).mul(gates.slice([0, e], [N, 1])));
    }
    for (let s = 0; s < moe.shared; s++) out = out.add(this.mlp(flat, `${p}s${s}.`));
    return out.reshape([B, Tn, this.D]);
  }

  /** idx: int32 [B, T] -> logits [B, T, V] */
  forward(idx: tf.Tensor2D, capture = false): tf.Tensor3D {
    const [B, Tn] = idx.shape;
    const { D, arch } = this;
    this.auxLoss = null;
    if (capture) this.captured = [];
    let x: T = tf.gather(this.params.wte, idx.flatten()).reshape([B, Tn, D]);
    if (arch.embedScale) x = x.mul(Math.sqrt(D));
    if (arch.pos === "learned") x = x.add(this.params.wpe.slice([0, 0], [Tn, D]));
    for (let l = 0; l < this.cfg.layers; l++) {
      const p = `h${l}.`;
      let a = this.attention(this.norm(x, p + "ln1"), l, B, Tn, capture);
      if (arch.sandwichNorm) a = this.norm(a, p + "postAttn");
      x = x.add(a);
      const h2 = this.norm(x, p + "ln2");
      let m = arch.moe ? this.moeBlock(h2, l, B, Tn) : this.mlp(h2, p + "mlp.");
      if (arch.sandwichNorm) m = this.norm(m, p + "postMlp");
      x = x.add(m);
    }
    x = this.norm(x, "lnf");
    const flat = x.reshape([B * Tn, D]);
    let logits: T = arch.tied
      ? tf.matMul(flat, this.params.wte, false, true)
      : tf.matMul(flat, this.params.lmHead);
    if (arch.softcap) logits = tf.tanh(logits.div(arch.softcap)).mul(arch.softcap);
    return logits.reshape([B, Tn, this.vocab]) as tf.Tensor3D;
  }
}
