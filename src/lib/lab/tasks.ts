export const VOCAB = "_0123456789abcdefghijklmnop+=>?|()YN";
export const STOI: Record<string, number> = Object.fromEntries([...VOCAB].map((c, i) => [c, i]));
export const encode = (s: string) => [...s].map((c) => STOI[c] ?? 0);
export const decode = (ids: number[]) => ids.map((i) => VOCAB[i] ?? "?").join("");

export type Rng = () => number;
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ri = (rng: Rng, n: number) => Math.floor(rng() * n);
const pad = (n: number, w: number) => String(n).padStart(w, "0");

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export interface Example {
  prompt: string;
  answer: string;
}

export interface TaskSpec {
  id: string;
  name: string;
  skill: string;
  description: string;
  promptLen: number;
  answerLen: number;
  /** whether to hold out a deterministic 20% of the input space as a test set */
  heldOut: boolean;
  gen: (rng: Rng) => Example;
}

function isBalanced(s: string) {
  let d = 0;
  for (const c of s) {
    d += c === "(" ? 1 : -1;
    if (d < 0) return false;
  }
  return d === 0;
}

export const TASKS: TaskSpec[] = [
  {
    id: "reverse",
    name: "Reverse",
    skill: "Positional attention",
    description: "Reverse a 6-letter string. Requires attending to mirrored positions.",
    promptLen: 7,
    answerLen: 6,
    heldOut: true,
    gen: (rng) => {
      let s = "";
      for (let i = 0; i < 6; i++) s += "abcdefgh"[ri(rng, 8)];
      return { prompt: s + ">", answer: [...s].reverse().join("") };
    },
  },
  {
    id: "sort",
    name: "Sort",
    skill: "Counting / comparison",
    description: "Sort 6 digits ascending. The model must learn an implicit counting algorithm.",
    promptLen: 7,
    answerLen: 6,
    heldOut: true,
    gen: (rng) => {
      const d = Array.from({ length: 6 }, () => ri(rng, 10));
      return { prompt: d.join("") + ">", answer: [...d].sort((a, b) => a - b).join("") };
    },
  },
  {
    id: "add",
    name: "Addition",
    skill: "Carry arithmetic",
    description: "Add two 3-digit numbers (answer zero-padded to 4 digits). Carries make it non-local.",
    promptLen: 8,
    answerLen: 4,
    heldOut: true,
    gen: (rng) => {
      const a = ri(rng, 1000);
      const b = ri(rng, 1000);
      return { prompt: `${pad(a, 3)}+${pad(b, 3)}=`, answer: pad(a + b, 4) };
    },
  },
  {
    id: "modadd",
    name: "Mod-47 Add",
    skill: "Grokking",
    description: "(a + b) mod 47 on a finite table; 20% of pairs are never seen. Classic grokking setup.",
    promptLen: 6,
    answerLen: 2,
    heldOut: true,
    gen: (rng) => {
      const a = ri(rng, 47);
      const b = ri(rng, 47);
      return { prompt: `${pad(a, 2)}+${pad(b, 2)}=`, answer: pad((a + b) % 47, 2) };
    },
  },
  {
    id: "recall",
    name: "Assoc. Recall",
    skill: "Induction heads",
    description: "Given 4 key→value pairs then a query key, output its value. Tests content-based lookup.",
    promptLen: 10,
    answerLen: 1,
    heldOut: true,
    gen: (rng) => {
      const keys = [..."abcdefghijklmnop"];
      for (let i = keys.length - 1; i > 0; i--) {
        const j = ri(rng, i + 1);
        [keys[i], keys[j]] = [keys[j], keys[i]];
      }
      const ks = keys.slice(0, 4);
      const vs = ks.map(() => String(ri(rng, 10)));
      const q = ri(rng, 4);
      return { prompt: ks.map((k, i) => k + vs[i]).join("") + "?" + ks[q], answer: vs[q] };
    },
  },
  {
    id: "parens",
    name: "Balanced Parens",
    skill: "Stack / depth tracking",
    description: "Is a 10-char parenthesis string balanced? Y/N (50/50). Requires tracking running depth.",
    promptLen: 11,
    answerLen: 1,
    heldOut: false,
    gen: (rng) => {
      let s = "";
      if (rng() < 0.5) {
        // random balanced string
        let open = 0;
        let remainingOpen = 5;
        for (let i = 0; i < 10; i++) {
          const canOpen = remainingOpen > 0;
          const canClose = open > 0;
          if (canOpen && (!canClose || rng() < 0.5)) {
            s += "(";
            open++;
            remainingOpen--;
          } else {
            s += ")";
            open--;
          }
        }
      } else {
        do {
          s = "";
          for (let i = 0; i < 10; i++) s += rng() < 0.5 ? "(" : ")";
        } while (isBalanced(s));
      }
      return { prompt: s + "?", answer: isBalanced(s) ? "Y" : "N" };
    },
  },
];

export function isTestExample(task: TaskSpec, ex: Example) {
  return task.heldOut && hash(task.id + ex.prompt) % 5 === 0;
}

/** Draw an example from the requested split. */
export function sample(task: TaskSpec, rng: Rng, split: "train" | "test"): Example {
  for (let tries = 0; tries < 200; tries++) {
    const ex = task.gen(rng);
    if (!task.heldOut) return ex;
    if (isTestExample(task, ex) === (split === "test")) return ex;
  }
  return task.gen(rng);
}
