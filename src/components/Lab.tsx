"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ARCHS, ArchSpec, ToyConfig, toyKvHeads } from "@/lib/lab/archs";
import { TASKS, mulberry32, sample } from "@/lib/lab/tasks";
import type { TrainSettings, TrainUpdate } from "@/lib/lab/trainer";
import { Heatmap, LineChart, Series, ema } from "./Charts";

interface SessionRun {
  key: string;
  archId: string;
  taskId: string;
  cfg: ToyConfig;
  update: TrainUpdate | null;
  status: "queued" | "running" | "done" | "stopped" | "error";
}

interface DbRun {
  id: number;
  modelId: string;
  modelName: string;
  taskId: string;
  taskName: string;
  config: { dModel?: number; layers?: number; heads?: number; lr?: number };
  params: number;
  activeParams: number;
  steps: number;
  finalLoss: number;
  trainAcc: number;
  testAcc: number;
  durationMs: number;
  createdAt: string;
}

const archById = (id: string) => ARCHS.find((a) => a.id === id)!;
const taskById = (id: string) => TASKS.find((t) => t.id === id)!;

function fmtParams(n: number) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}
function fmtDim(n: number) {
  return n >= 1000 ? (n / 1000).toFixed(n % 1000 === 0 ? 0 : 1) + "K" : String(n);
}

function Pill({ children, active, onClick, color }: { children: React.ReactNode; active?: boolean; onClick?: () => void; color?: string }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-full border px-3 py-1 text-xs transition ${
        active ? "border-transparent text-slate-950 font-semibold" : "border-slate-700 text-slate-300 hover:border-slate-500"
      }`}
      style={active ? { background: color ?? "#38bdf8" } : undefined}
    >
      {children}
    </button>
  );
}

function NumField({
  label,
  value,
  onChange,
  options,
  step,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  options?: number[];
  step?: number;
  disabled?: boolean;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs text-slate-400">
      {label}
      {options ? (
        <select
          disabled={disabled}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-100"
        >
          {options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      ) : (
        <input
          disabled={disabled}
          type="number"
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-100"
        />
      )}
    </label>
  );
}

function ArchCard({ arch, cfg, selected, onToggle }: { arch: ArchSpec; cfg: ToyConfig; selected: boolean; onToggle: () => void }) {
  const kv = arch.mla ? cfg.heads : toyKvHeads(arch, cfg.heads);
  const ffn = arch.moe ? Math.round(cfg.dModel * arch.moe.expertMult) : Math.round(cfg.dModel * arch.ffnMult);
  const rows: [string, string, string][] = [
    ["d_model", fmtDim(arch.real.dModel), String(cfg.dModel)],
    ["layers", String(arch.real.layers), String(cfg.layers)],
    ["heads (q/kv)", `${arch.real.heads}/${arch.real.kvHeads}`, `${cfg.heads}/${kv}`],
    [arch.moe ? "expert ffn" : "ffn", fmtDim(arch.real.ffn), String(ffn)],
    ["vocab", fmtDim(arch.real.vocab), "36"],
  ];
  if (arch.moe) rows.push(["experts", arch.real.extra?.split(",")[0] ?? "", `${arch.moe.experts} top-${arch.moe.topK}${arch.moe.shared ? " +1 shared" : ""}`]);
  return (
    <button
      onClick={onToggle}
      className={`group relative flex flex-col rounded-xl border p-4 text-left transition ${
        selected ? "bg-slate-900" : "border-slate-800 bg-slate-900/40 hover:border-slate-600"
      }`}
      style={selected ? { borderColor: arch.color, boxShadow: `0 0 0 1px ${arch.color}55, 0 8px 30px -12px ${arch.color}88` } : undefined}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: arch.color }} />
            <span className="font-semibold text-slate-100">{arch.name}</span>
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            {arch.org} · {arch.year} · {arch.real.params}
          </div>
        </div>
        <span
          className={`rounded-md px-1.5 py-0.5 text-[10px] font-semibold ${selected ? "text-slate-950" : "bg-slate-800 text-slate-400"}`}
          style={selected ? { background: arch.color } : undefined}
        >
          {selected ? "IN ARENA" : "ADD"}
        </span>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-slate-400">{arch.blurb}</p>
      <table className="mt-3 w-full text-[11px]">
        <thead>
          <tr className="text-slate-500">
            <th className="text-left font-normal" />
            <th className="text-right font-normal">real</th>
            <th className="text-right font-normal">toy</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {rows.map(([k, r, t]) => (
            <tr key={k}>
              <td className="text-slate-500">{k}</td>
              <td className="text-right text-slate-500">{r}</td>
              <td className="text-right text-slate-200">{t}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 flex flex-wrap gap-1">
        {arch.features.map((f) => (
          <span key={f} className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300">
            {f}
          </span>
        ))}
      </div>
    </button>
  );
}

export default function Lab() {
  const [selected, setSelected] = useState<string[]>(["gpt2", "llama3", "mixtral"]);
  const [taskId, setTaskId] = useState("reverse");
  const [cfg, setCfg] = useState<ToyConfig>({ dModel: 64, layers: 2, heads: 4 });
  const [settings, setSettings] = useState<TrainSettings>({
    lr: 0.003,
    batchSize: 32,
    steps: 500,
    warmup: 40,
    evalEvery: 50,
    clip: 1,
    seed: 1,
    backend: "webgl",
  });
  const [session, setSession] = useState<SessionRun[]>([]);
  const [focusKey, setFocusKey] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const stopRef = useRef(false);
  const [dbRuns, setDbRuns] = useState<DbRun[]>([]);
  const [layerSel, setLayerSel] = useState(0);
  const [headSel, setHeadSel] = useState(0);

  const task = taskById(taskId);
  const examples = useMemo(() => {
    const rng = mulberry32(42);
    return Array.from({ length: 3 }, () => sample(task, rng, "train"));
  }, [task]);

  const loadRuns = useCallback(async () => {
    try {
      const r = await fetch("/api/runs", { cache: "no-store" });
      if (r.ok) setDbRuns(await r.json());
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  const toggle = (id: string) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  const start = async () => {
    if (!selected.length || running) return;
    const { train } = await import("@/lib/lab/trainer");
    stopRef.current = false;
    setRunning(true);
    const order = ARCHS.filter((a) => selected.includes(a.id));
    const stamp = Date.now();
    const queue: SessionRun[] = order.map((a) => ({
      key: `${stamp}-${a.id}`,
      archId: a.id,
      taskId,
      cfg: { ...cfg },
      update: null,
      status: "queued",
    }));
    setSession(queue);
    setFocusKey(queue[0].key);
    const patch = (key: string, p: Partial<SessionRun>) =>
      setSession((s) => s.map((r) => (r.key === key ? { ...r, ...p } : r)));

    for (const run of queue) {
      if (stopRef.current) {
        patch(run.key, { status: "stopped" });
        continue;
      }
      const arch = archById(run.archId);
      setFocusKey(run.key);
      patch(run.key, { status: "running" });
      const final = await train(
        arch,
        run.cfg,
        task,
        settings,
        (u) => patch(run.key, { update: u }),
        () => stopRef.current
      );
      const status = final.error ? "error" : stopRef.current ? "stopped" : "done";
      patch(run.key, { update: final, status });
      if (!final.error && final.step > 0) {
        const last = final.accCurve[final.accCurve.length - 1];
        const tail = final.lossCurve.slice(-20);
        const finalLoss = tail.reduce((s, p) => s + p[1], 0) / Math.max(1, tail.length);
        const stride = Math.max(1, Math.floor(final.lossCurve.length / 200));
        await fetch("/api/runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            modelId: arch.id,
            modelName: arch.name,
            taskId: task.id,
            taskName: task.name,
            config: { ...run.cfg, lr: settings.lr, batchSize: settings.batchSize, backend: final.backend },
            params: final.paramCount,
            activeParams: final.activeParams,
            steps: final.step,
            finalLoss,
            trainAcc: last?.train ?? 0,
            testAcc: last?.test ?? 0,
            durationMs: final.elapsedMs,
            lossCurve: final.lossCurve.filter((_, i) => i % stride === 0),
            accCurve: final.accCurve,
          }),
        }).catch(() => undefined);
        loadRuns();
      }
    }
    setRunning(false);
  };

  const focus = session.find((r) => r.key === focusKey) ?? session[0];
  const fu = focus?.update ?? null;
  const focusArch = focus ? archById(focus.archId) : null;

  const lossSeries: Series[] = session
    .filter((r) => r.update)
    .map((r) => ({ name: r.archId, color: archById(r.archId).color, points: ema(r.update!.lossCurve) }));
  const accSeries: Series[] = session.flatMap((r) =>
    r.update
      ? [
          { name: r.archId + " test", color: archById(r.archId).color, points: r.update.accCurve.map((a) => [a.step, a.test] as [number, number]) },
          {
            name: r.archId + " train",
            color: archById(r.archId).color,
            dashed: true,
            points: r.update.accCurve.map((a) => [a.step, a.train] as [number, number]),
          },
        ]
      : []
  );

  // leaderboard: best test acc per (model, task)
  const board = useMemo(() => {
    const m = new Map<string, DbRun>();
    for (const r of dbRuns) {
      const k = `${r.modelId}|${r.taskId}`;
      const cur = m.get(k);
      if (!cur || r.testAcc > cur.testAcc) m.set(k, r);
    }
    return m;
  }, [dbRuns]);

  const attnLayers = fu?.attn?.maps.length ?? 0;
  const L = Math.min(layerSel, Math.max(0, attnLayers - 1));
  const Hh = Math.min(headSel, Math.max(0, (fu?.attn?.maps[L]?.length ?? 1) - 1));

  const deleteRun = async (id: number) => {
    await fetch(`/api/runs/${id}`, { method: "DELETE" });
    loadRuns();
  };
  const clearRuns = async () => {
    if (!confirm("Delete all saved runs?")) return;
    await fetch(`/api/runs`, { method: "DELETE" });
    loadRuns();
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <header className="border-b border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950">
        <div className="mx-auto max-w-7xl px-6 py-8">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-400">TinyLab</div>
              <h1 className="mt-1 text-3xl font-bold text-white">SOTA open-weight architectures, shrunk to toy size</h1>
              <p className="mt-2 max-w-3xl text-sm text-slate-400">
                Each model keeps its real architectural recipe (RoPE, GQA, SwiGLU, QK-Norm, sliding windows, MoE routing, latent
                attention, attention sinks…) but at ~10⁻⁵ of the parameters, so it trains from scratch in your browser on WebGL.
                Pick models, pick a task, and race them.
              </p>
            </div>
            <div className="text-right text-xs text-slate-500">
              <div>{dbRuns.length} saved runs</div>
              <div>Training runs fully client-side with TensorFlow.js</div>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-8 px-6 py-8">
        {/* Model zoo */}
        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">1 · Model zoo</h2>
            <div className="flex gap-2 text-xs">
              <button className="text-slate-400 hover:text-white" onClick={() => setSelected(ARCHS.map((a) => a.id))}>
                select all
              </button>
              <span className="text-slate-700">|</span>
              <button className="text-slate-400 hover:text-white" onClick={() => setSelected([])}>
                clear
              </button>
            </div>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {ARCHS.map((a) => (
              <ArchCard key={a.id} arch={a} cfg={cfg} selected={selected.includes(a.id)} onToggle={() => toggle(a.id)} />
            ))}
          </div>
        </section>

        {/* Task + config */}
        <section className="grid gap-6 lg:grid-cols-5">
          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 lg:col-span-3">
            <h2 className="text-lg font-semibold text-white">2 · Task</h2>
            <div className="mt-3 flex flex-wrap gap-2">
              {TASKS.map((t) => (
                <Pill key={t.id} active={t.id === taskId} onClick={() => !running && setTaskId(t.id)}>
                  {t.name}
                </Pill>
              ))}
            </div>
            <div className="mt-4 flex flex-wrap items-start justify-between gap-4">
              <div className="max-w-md">
                <div className="text-xs uppercase tracking-wider text-sky-400">{task.skill}</div>
                <p className="mt-1 text-sm text-slate-300">{task.description}</p>
                <p className="mt-2 text-xs text-slate-500">
                  {task.heldOut
                    ? "Test accuracy is measured on a held-out 20% of inputs the model never trains on."
                    : "Test examples are freshly sampled each evaluation."}{" "}
                  Loss is computed only on answer tokens.
                </p>
              </div>
              <div className="rounded-lg bg-slate-950 p-3 font-mono text-sm">
                {examples.map((e, i) => (
                  <div key={i}>
                    <span className="text-slate-400">{e.prompt}</span>
                    <span className="text-emerald-400">{e.answer}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 lg:col-span-2">
            <h2 className="text-lg font-semibold text-white">3 · Scale &amp; train</h2>
            <div className="mt-3 grid grid-cols-3 gap-3">
              <NumField label="d_model" value={cfg.dModel} options={[32, 48, 64, 96, 128]} disabled={running} onChange={(v) => setCfg({ ...cfg, dModel: v })} />
              <NumField label="layers" value={cfg.layers} options={[1, 2, 3, 4]} disabled={running} onChange={(v) => setCfg({ ...cfg, layers: v })} />
              <NumField label="heads" value={cfg.heads} options={[2, 4, 8]} disabled={running} onChange={(v) => setCfg({ ...cfg, heads: v })} />
              <NumField label="steps" value={settings.steps} step={50} disabled={running} onChange={(v) => setSettings({ ...settings, steps: Math.max(10, v) })} />
              <NumField label="learning rate" value={settings.lr} step={0.0005} disabled={running} onChange={(v) => setSettings({ ...settings, lr: v })} />
              <NumField label="batch" value={settings.batchSize} options={[16, 32, 64, 128]} disabled={running} onChange={(v) => setSettings({ ...settings, batchSize: v })} />
              <NumField label="warmup" value={settings.warmup} step={10} disabled={running} onChange={(v) => setSettings({ ...settings, warmup: Math.max(1, v) })} />
              <NumField label="eval every" value={settings.evalEvery} step={10} disabled={running} onChange={(v) => setSettings({ ...settings, evalEvery: Math.max(10, v) })} />
              <NumField label="seed" value={settings.seed} disabled={running} onChange={(v) => setSettings({ ...settings, seed: Math.max(1, Math.round(v)) })} />
            </div>
            <div className="mt-3 flex items-center gap-2 text-xs text-slate-400">
              backend:
              {(["webgl", "cpu"] as const).map((b) => (
                <Pill key={b} active={settings.backend === b} onClick={() => !running && setSettings({ ...settings, backend: b })}>
                  {b}
                </Pill>
              ))}
            </div>
            <div className="mt-4 flex gap-2">
              <button
                onClick={start}
                disabled={running || !selected.length}
                className="flex-1 rounded-lg bg-sky-500 px-4 py-2.5 text-sm font-semibold text-slate-950 hover:bg-sky-400 disabled:opacity-40"
              >
                {running ? "Training…" : `Train ${selected.length} model${selected.length === 1 ? "" : "s"} on ${task.name}`}
              </button>
              {running && (
                <button
                  onClick={() => (stopRef.current = true)}
                  className="rounded-lg border border-rose-500 px-4 py-2.5 text-sm font-semibold text-rose-400 hover:bg-rose-500/10"
                >
                  Stop
                </button>
              )}
            </div>
            <p className="mt-2 text-[11px] text-slate-500">Models train one after another with identical data order &amp; seed for a fair race.</p>
          </div>
        </section>

        {/* Live arena */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-semibold text-white">4 · Arena</h2>
            <div className="flex flex-wrap gap-2">
              {session.map((r) => {
                const a = archById(r.archId);
                const u = r.update;
                const acc = u?.accCurve[u.accCurve.length - 1]?.test;
                return (
                  <button
                    key={r.key}
                    onClick={() => setFocusKey(r.key)}
                    className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs ${
                      focus?.key === r.key ? "border-slate-400 bg-slate-800" : "border-slate-800 hover:border-slate-600"
                    }`}
                  >
                    <span className="h-2 w-2 rounded-full" style={{ background: a.color }} />
                    <span className="text-slate-200">{a.name}</span>
                    <span className="text-slate-500">
                      {r.status === "running" && u ? `${u.step}/${u.totalSteps}` : r.status}
                    </span>
                    {acc !== undefined && <span className="font-mono text-emerald-400">{Math.round(acc * 100)}%</span>}
                  </button>
                );
              })}
            </div>
          </div>

          {session.length === 0 ? (
            <div className="mt-6 rounded-lg border border-dashed border-slate-800 p-10 text-center text-sm text-slate-500">
              Select models above and hit <span className="text-sky-400">Train</span> to start a race. Loss curves, held-out accuracy,
              attention maps and MoE routing stats will stream in here.
            </div>
          ) : (
            <>
              <div className="mt-4 grid gap-4 lg:grid-cols-2">
                <div className="rounded-lg bg-slate-950 p-3">
                  <div className="mb-1 text-xs text-slate-400">Training loss (answer tokens, EMA)</div>
                  <LineChart series={lossSeries} xMax={settings.steps} yMin={0.001} logY />
                </div>
                <div className="rounded-lg bg-slate-950 p-3">
                  <div className="mb-1 text-xs text-slate-400">Exact-match accuracy — solid: held-out test, dashed: train</div>
                  <LineChart series={accSeries} xMax={settings.steps} yMin={0} yMax={1} percent />
                </div>
              </div>

              {fu && focusArch && (
                <div className="mt-4 grid gap-4 lg:grid-cols-3">
                  <div className="rounded-lg bg-slate-950 p-4">
                    <div className="flex items-center gap-2">
                      <span className="h-2.5 w-2.5 rounded-full" style={{ background: focusArch.color }} />
                      <span className="font-semibold text-white">{focusArch.name}</span>
                      <span className="text-xs text-slate-500">toy</span>
                    </div>
                    {fu.error && <div className="mt-2 rounded bg-rose-500/10 p-2 text-xs text-rose-300">{fu.error}</div>}
                    <dl className="mt-3 grid grid-cols-2 gap-y-1.5 text-xs">
                      <dt className="text-slate-500">Parameters</dt>
                      <dd className="text-right font-mono">{fmtParams(fu.paramCount)}</dd>
                      <dt className="text-slate-500">Active / token</dt>
                      <dd className="text-right font-mono">{fmtParams(fu.activeParams)}</dd>
                      <dt className="text-slate-500">Real model</dt>
                      <dd className="text-right font-mono">{focusArch.real.params}</dd>
                      <dt className="text-slate-500">Step</dt>
                      <dd className="text-right font-mono">
                        {fu.step}/{fu.totalSteps}
                      </dd>
                      <dt className="text-slate-500">Loss</dt>
                      <dd className="text-right font-mono">{Number.isFinite(fu.loss) ? fu.loss.toFixed(4) : "—"}</dd>
                      <dt className="text-slate-500">LR</dt>
                      <dd className="text-right font-mono">{fu.lr.toExponential(2)}</dd>
                      <dt className="text-slate-500">Time</dt>
                      <dd className="text-right font-mono">{(fu.elapsedMs / 1000).toFixed(1)}s</dd>
                      <dt className="text-slate-500">Throughput</dt>
                      <dd className="text-right font-mono">{(fu.step / Math.max(0.001, fu.elapsedMs / 1000)).toFixed(1)} it/s</dd>
                      <dt className="text-slate-500">Backend</dt>
                      <dd className="text-right font-mono">{fu.backend}</dd>
                    </dl>
                    <div className="mt-3 h-1.5 overflow-hidden rounded bg-slate-800">
                      <div className="h-full transition-all" style={{ width: `${(fu.step / fu.totalSteps) * 100}%`, background: focusArch.color }} />
                    </div>

                    {fu.expertUsage && (
                      <div className="mt-4">
                        <div className="text-xs text-slate-400">MoE routing — share of tokens per expert</div>
                        {fu.expertUsage.map((row, l) => {
                          const tot = row.reduce((a, b) => a + b, 0) || 1;
                          return (
                            <div key={l} className="mt-2">
                              <div className="text-[10px] text-slate-500">layer {l}</div>
                              <div className="flex h-12 items-end gap-1">
                                {row.map((c, e) => (
                                  <div key={e} className="flex flex-1 flex-col items-center">
                                    <div
                                      className="w-full rounded-t"
                                      style={{ height: `${Math.max(2, (c / tot) * 100 * 1.8)}%`, background: focusArch.color, opacity: 0.85 }}
                                      title={`expert ${e}: ${((c / tot) * 100).toFixed(1)}%`}
                                    />
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  <div className="rounded-lg bg-slate-950 p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="text-xs text-slate-400">Attention map (held-out example)</div>
                      <div className="flex gap-1 text-[10px]">
                        {Array.from({ length: attnLayers }, (_, l) => (
                          <Pill key={l} active={l === L} color={focusArch.color} onClick={() => setLayerSel(l)}>
                            L{l}
                            {fu.attn?.local[l] ? "·win" : ""}
                          </Pill>
                        ))}
                      </div>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {Array.from({ length: fu.attn?.maps[L]?.length ?? 0 }, (_, h) => (
                        <Pill key={h} active={h === Hh} color={focusArch.color} onClick={() => setHeadSel(h)}>
                          h{h}
                        </Pill>
                      ))}
                    </div>
                    <div className="mt-3 overflow-auto">
                      {fu.attn && fu.attn.maps[L] ? (
                        <Heatmap map={fu.attn.maps[L][Hh]} tokens={fu.attn.tokens} color={focusArch.color} />
                      ) : (
                        <div className="text-xs text-slate-500">Waiting for first evaluation…</div>
                      )}
                    </div>
                    <p className="mt-2 text-[10px] text-slate-500">Rows = query position, columns = key position. Causal mask greys out the future.</p>
                  </div>

                  <div className="rounded-lg bg-slate-950 p-4">
                    <div className="text-xs text-slate-400">Held-out predictions</div>
                    <table className="mt-2 w-full font-mono text-xs">
                      <thead>
                        <tr className="text-slate-500">
                          <th className="text-left font-normal">prompt</th>
                          <th className="text-left font-normal">target</th>
                          <th className="text-left font-normal">model</th>
                        </tr>
                      </thead>
                      <tbody>
                        {fu.samples.map((s, i) => (
                          <tr key={i}>
                            <td className="py-0.5 text-slate-400">{s.prompt}</td>
                            <td className="text-slate-300">{s.target}</td>
                            <td className={s.ok ? "text-emerald-400" : "text-rose-400"}>
                              {s.pred} {s.ok ? "✓" : "✗"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {fu.samples.length === 0 && <div className="mt-2 text-xs text-slate-500">Waiting for first evaluation…</div>}
                  </div>
                </div>
              )}
            </>
          )}
        </section>

        {/* Leaderboard */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">5 · Leaderboard — best held-out accuracy</h2>
            {dbRuns.length > 0 && (
              <button onClick={clearRuns} className="text-xs text-slate-500 hover:text-rose-400">
                clear all
              </button>
            )}
          </div>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-500">
                  <th className="py-2 text-left font-normal">Model</th>
                  {TASKS.map((t) => (
                    <th key={t.id} className="px-2 py-2 text-center font-normal">
                      {t.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ARCHS.map((a) => (
                  <tr key={a.id} className="border-t border-slate-800">
                    <td className="py-2">
                      <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: a.color }} />
                      {a.name}
                    </td>
                    {TASKS.map((t) => {
                      const r = board.get(`${a.id}|${t.id}`);
                      return (
                        <td key={t.id} className="px-2 py-1.5 text-center">
                          {r ? (
                            <span
                              className="inline-block min-w-14 rounded px-2 py-1 font-mono text-xs"
                              style={{ background: `rgba(16,185,129,${0.1 + r.testAcc * 0.6})`, color: r.testAcc > 0.5 ? "#ecfdf5" : "#a7f3d0" }}
                              title={`${r.steps} steps · d=${r.config.dModel} L=${r.config.layers} · ${fmtParams(r.params)} params`}
                            >
                              {(r.testAcc * 100).toFixed(0)}%
                            </span>
                          ) : (
                            <span className="text-slate-700">—</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {dbRuns.length > 0 && (
            <details className="mt-5">
              <summary className="cursor-pointer text-xs text-slate-400 hover:text-white">Run history ({dbRuns.length})</summary>
              <div className="mt-3 max-h-80 overflow-auto">
                <table className="w-full text-xs">
                  <thead className="text-slate-500">
                    <tr>
                      <th className="text-left font-normal">When</th>
                      <th className="text-left font-normal">Model</th>
                      <th className="text-left font-normal">Task</th>
                      <th className="text-right font-normal">Config</th>
                      <th className="text-right font-normal">Params</th>
                      <th className="text-right font-normal">Steps</th>
                      <th className="text-right font-normal">Loss</th>
                      <th className="text-right font-normal">Train</th>
                      <th className="text-right font-normal">Test</th>
                      <th className="text-right font-normal">Time</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody className="font-mono">
                    {dbRuns.map((r) => (
                      <tr key={r.id} className="border-t border-slate-800/60">
                        <td className="py-1 text-slate-500">{new Date(r.createdAt).toLocaleString()}</td>
                        <td>{r.modelName}</td>
                        <td>{r.taskName}</td>
                        <td className="text-right text-slate-400">
                          d{r.config.dModel} L{r.config.layers} H{r.config.heads}
                        </td>
                        <td className="text-right">
                          {fmtParams(r.params)}
                          {r.activeParams !== r.params && <span className="text-slate-500"> ({fmtParams(r.activeParams)})</span>}
                        </td>
                        <td className="text-right">{r.steps}</td>
                        <td className="text-right">{r.finalLoss.toFixed(3)}</td>
                        <td className="text-right">{(r.trainAcc * 100).toFixed(0)}%</td>
                        <td className="text-right text-emerald-400">{(r.testAcc * 100).toFixed(0)}%</td>
                        <td className="text-right">{(r.durationMs / 1000).toFixed(0)}s</td>
                        <td className="text-right">
                          <button onClick={() => deleteRun(r.id)} className="text-slate-600 hover:text-rose-400">
                            ✕
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </section>

        <footer className="pb-8 text-center text-xs text-slate-600">
          Toy re-implementations for education — architectural features are faithful in spirit, simplified in detail (e.g. MLA without
          decoupled RoPE, dense-computed experts with top-k masking).
        </footer>
      </main>
    </div>
  );
}
