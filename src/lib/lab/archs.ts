export type NormType = "layernorm" | "rmsnorm";
export type MlpType = "gelu" | "swiglu" | "geglu";

export interface ArchSpec {
  id: string;
  name: string;
  org: string;
  year: number;
  color: string;
  blurb: string;
  real: {
    params: string;
    dModel: number;
    layers: number;
    heads: number;
    kvHeads: number;
    ffn: number;
    vocab: number;
    context: string;
    extra?: string;
  };
  norm: NormType;
  pos: "learned" | "rope";
  mlp: MlpType;
  /** query heads per KV head (1 = classic multi-head attention) */
  kvRatio: number;
  qkvBias: boolean;
  qkNorm: boolean;
  tied: boolean;
  window: { size: number; pattern: "all" | "alternate" } | null;
  moe: { experts: number; topK: number; shared: number; expertMult: number } | null;
  softcap: number | null;
  embedScale: boolean;
  sandwichNorm: boolean;
  ropeTheta: number;
  ffnMult: number;
  mla: { latentRatio: number } | null;
  sinks: boolean;
  features: string[];
}

export const ARCHS: ArchSpec[] = [
  {
    id: "gpt2",
    name: "GPT-2",
    org: "OpenAI",
    year: 2019,
    color: "#94a3b8",
    blurb: "The classic baseline: LayerNorm, learned absolute positions, GELU MLP, full multi-head attention.",
    real: { params: "124M", dModel: 768, layers: 12, heads: 12, kvHeads: 12, ffn: 3072, vocab: 50257, context: "1K" },
    norm: "layernorm",
    pos: "learned",
    mlp: "gelu",
    kvRatio: 1,
    qkvBias: true,
    qkNorm: false,
    tied: true,
    window: null,
    moe: null,
    softcap: null,
    embedScale: false,
    sandwichNorm: false,
    ropeTheta: 10000,
    ffnMult: 4,
    mla: null,
    sinks: false,
    features: ["LayerNorm", "Learned pos-emb", "GELU MLP", "MHA", "Biases", "Tied emb"],
  },
  {
    id: "llama3",
    name: "Llama 3.1 8B",
    org: "Meta",
    year: 2024,
    color: "#3b82f6",
    blurb: "The modern decoder template: pre-RMSNorm, RoPE (θ=500k), SwiGLU, grouped-query attention, no biases.",
    real: { params: "8.0B", dModel: 4096, layers: 32, heads: 32, kvHeads: 8, ffn: 14336, vocab: 128256, context: "128K" },
    norm: "rmsnorm",
    pos: "rope",
    mlp: "swiglu",
    kvRatio: 4,
    qkvBias: false,
    qkNorm: false,
    tied: false,
    window: null,
    moe: null,
    softcap: null,
    embedScale: false,
    sandwichNorm: false,
    ropeTheta: 500000,
    ffnMult: 3.5,
    mla: null,
    sinks: false,
    features: ["RMSNorm", "RoPE θ=500k", "SwiGLU", "GQA 4:1", "No bias"],
  },
  {
    id: "qwen3",
    name: "Qwen3 8B",
    org: "Alibaba",
    year: 2025,
    color: "#a855f7",
    blurb: "Llama-style backbone plus QK-Norm (RMSNorm on per-head queries & keys) for stable attention logits.",
    real: { params: "8.2B", dModel: 4096, layers: 36, heads: 32, kvHeads: 8, ffn: 12288, vocab: 151936, context: "128K" },
    norm: "rmsnorm",
    pos: "rope",
    mlp: "swiglu",
    kvRatio: 4,
    qkvBias: false,
    qkNorm: true,
    tied: false,
    window: null,
    moe: null,
    softcap: null,
    embedScale: false,
    sandwichNorm: false,
    ropeTheta: 1000000,
    ffnMult: 3,
    mla: null,
    sinks: false,
    features: ["RMSNorm", "QK-Norm", "RoPE θ=1M", "SwiGLU", "GQA 4:1"],
  },
  {
    id: "gemma3",
    name: "Gemma 3 12B",
    org: "Google",
    year: 2025,
    color: "#10b981",
    blurb: "Sandwich norms (pre+post), GeGLU, QK-Norm, √d embedding scale, tied embeddings, interleaved local/global attention.",
    real: { params: "12B", dModel: 3840, layers: 48, heads: 16, kvHeads: 8, ffn: 15360, vocab: 262144, context: "128K", extra: "5 local : 1 global, window 1024" },
    norm: "rmsnorm",
    pos: "rope",
    mlp: "geglu",
    kvRatio: 2,
    qkvBias: false,
    qkNorm: true,
    tied: true,
    window: { size: 6, pattern: "alternate" },
    moe: null,
    softcap: null,
    embedScale: true,
    sandwichNorm: true,
    ropeTheta: 1000000,
    ffnMult: 4,
    mla: null,
    sinks: false,
    features: ["Sandwich norm", "GeGLU", "QK-Norm", "Local/global attn", "√d emb scale", "Tied emb"],
  },
  {
    id: "mistral",
    name: "Mistral 7B",
    org: "Mistral AI",
    year: 2023,
    color: "#f97316",
    blurb: "Llama-like but every layer uses sliding-window attention; stacking layers widens the receptive field.",
    real: { params: "7.2B", dModel: 4096, layers: 32, heads: 32, kvHeads: 8, ffn: 14336, vocab: 32000, context: "32K", extra: "sliding window 4096" },
    norm: "rmsnorm",
    pos: "rope",
    mlp: "swiglu",
    kvRatio: 4,
    qkvBias: false,
    qkNorm: false,
    tied: false,
    window: { size: 6, pattern: "all" },
    moe: null,
    softcap: null,
    embedScale: false,
    sandwichNorm: false,
    ropeTheta: 10000,
    ffnMult: 3.5,
    mla: null,
    sinks: false,
    features: ["Sliding-window attn", "RMSNorm", "RoPE", "SwiGLU", "GQA 4:1"],
  },
  {
    id: "mixtral",
    name: "Mixtral 8x7B",
    org: "Mistral AI",
    year: 2023,
    color: "#ef4444",
    blurb: "Sparse mixture-of-experts: each token is routed to the top-2 of 8 SwiGLU experts per layer.",
    real: { params: "46.7B (12.9B active)", dModel: 4096, layers: 32, heads: 32, kvHeads: 8, ffn: 14336, vocab: 32000, context: "32K", extra: "8 experts, top-2" },
    norm: "rmsnorm",
    pos: "rope",
    mlp: "swiglu",
    kvRatio: 4,
    qkvBias: false,
    qkNorm: false,
    tied: false,
    window: null,
    moe: { experts: 4, topK: 2, shared: 0, expertMult: 2 },
    softcap: null,
    embedScale: false,
    sandwichNorm: false,
    ropeTheta: 1000000,
    ffnMult: 2,
    mla: null,
    sinks: false,
    features: ["MoE top-2", "Load-balance loss", "RMSNorm", "SwiGLU experts", "GQA"],
  },
  {
    id: "deepseek3",
    name: "DeepSeek-V3",
    org: "DeepSeek",
    year: 2024,
    color: "#06b6d4",
    blurb: "Fine-grained MoE with an always-on shared expert, plus Multi-head Latent Attention (low-rank compressed KV).",
    real: { params: "671B (37B active)", dModel: 7168, layers: 61, heads: 128, kvHeads: 128, ffn: 2048, vocab: 129280, context: "128K", extra: "256 routed experts top-8 + 1 shared, MLA kv_rank 512" },
    norm: "rmsnorm",
    pos: "rope",
    mlp: "swiglu",
    kvRatio: 1,
    qkvBias: false,
    qkNorm: false,
    tied: false,
    window: null,
    moe: { experts: 8, topK: 2, shared: 1, expertMult: 1 },
    softcap: null,
    embedScale: false,
    sandwichNorm: false,
    ropeTheta: 10000,
    ffnMult: 1,
    mla: { latentRatio: 0.25 },
    sinks: false,
    features: ["MLA (latent KV)", "Fine-grained MoE", "Shared expert", "RMSNorm", "SwiGLU"],
  },
  {
    id: "gptoss",
    name: "gpt-oss-20b",
    org: "OpenAI",
    year: 2025,
    color: "#eab308",
    blurb: "MoE with learned attention sinks, alternating sliding-window/full attention, and biased projections.",
    real: { params: "21B (3.6B active)", dModel: 2880, layers: 24, heads: 64, kvHeads: 8, ffn: 2880, vocab: 201088, context: "128K", extra: "32 experts top-4, window 128" },
    norm: "rmsnorm",
    pos: "rope",
    mlp: "swiglu",
    kvRatio: 2,
    qkvBias: true,
    qkNorm: false,
    tied: false,
    window: { size: 6, pattern: "alternate" },
    moe: { experts: 4, topK: 2, shared: 0, expertMult: 1.5 },
    softcap: null,
    embedScale: false,
    sandwichNorm: false,
    ropeTheta: 150000,
    ffnMult: 1.5,
    mla: null,
    sinks: true,
    features: ["Attention sinks", "Alt. sliding window", "MoE top-2", "GQA", "QKV bias"],
  },
];

export interface ToyConfig {
  dModel: number;
  layers: number;
  heads: number;
}

export function toyKvHeads(arch: ArchSpec, heads: number): number {
  const target = Math.max(1, Math.floor(heads / arch.kvRatio));
  for (let k = target; k >= 1; k--) if (heads % k === 0) return k;
  return 1;
}

export function shrinkFactor(arch: ArchSpec, cfg: ToyConfig): number {
  return arch.real.dModel / cfg.dModel;
}
