"use client";

export interface Series {
  name: string;
  color: string;
  points: [number, number][];
  dashed?: boolean;
}

export function LineChart({
  series,
  height = 220,
  yMin,
  yMax,
  xMax,
  yLabel,
  percent,
  logY,
}: {
  series: Series[];
  height?: number;
  yMin?: number;
  yMax?: number;
  xMax?: number;
  yLabel?: string;
  percent?: boolean;
  logY?: boolean;
}) {
  const W = 600;
  const H = height;
  const pad = { l: 44, r: 12, t: 10, b: 24 };
  const all = series.flatMap((s) => s.points);
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]).filter((v) => Number.isFinite(v));
  const x1 = xMax ?? Math.max(1, ...xs);
  const tf = (v: number) => (logY ? Math.log10(Math.max(v, 1e-4)) : v);
  let y0 = yMin ?? (ys.length ? Math.min(...ys) : 0);
  let y1 = yMax ?? (ys.length ? Math.max(...ys) : 1);
  if (y1 - y0 < 1e-9) y1 = y0 + 1;
  y0 = tf(y0);
  y1 = tf(y1);
  const sx = (x: number) => pad.l + (x / x1) * (W - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - (tf(y) - y0) / (y1 - y0)) * (H - pad.t - pad.b);
  const ticks = 4;
  const fmt = (v: number) => (percent ? `${Math.round(v * 100)}%` : v >= 10 ? v.toFixed(0) : v.toFixed(2));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none" style={{ height }}>
      {Array.from({ length: ticks + 1 }, (_, i) => {
        const t = y0 + ((y1 - y0) * i) / ticks;
        const v = logY ? Math.pow(10, t) : t;
        const y = pad.t + (1 - i / ticks) * (H - pad.t - pad.b);
        return (
          <g key={i}>
            <line x1={pad.l} x2={W - pad.r} y1={y} y2={y} stroke="#1e293b" strokeWidth={1} />
            <text x={pad.l - 6} y={y + 3} fontSize={10} fill="#64748b" textAnchor="end">
              {fmt(v)}
            </text>
          </g>
        );
      })}
      {Array.from({ length: 5 }, (_, i) => {
        const x = (x1 * i) / 4;
        return (
          <text key={i} x={sx(x)} y={H - 6} fontSize={10} fill="#64748b" textAnchor="middle">
            {Math.round(x)}
          </text>
        );
      })}
      {yLabel && (
        <text x={pad.l + 4} y={pad.t + 10} fontSize={10} fill="#475569">
          {yLabel}
        </text>
      )}
      {series.map((s, i) =>
        s.points.length > 1 ? (
          <polyline
            key={i}
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            strokeDasharray={s.dashed ? "5 4" : undefined}
            strokeLinejoin="round"
            points={s.points
              .filter((p) => Number.isFinite(p[1]))
              .map((p) => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`)
              .join(" ")}
          />
        ) : null
      )}
    </svg>
  );
}

export function Heatmap({ map, tokens, color = "#38bdf8" }: { map: number[][]; tokens: string[]; color?: string }) {
  const n = tokens.length;
  const cell = Math.max(14, Math.min(26, Math.floor(300 / Math.max(1, n))));
  return (
    <div className="inline-block">
      <div className="flex" style={{ marginLeft: cell }}>
        {tokens.map((t, j) => (
          <div key={j} style={{ width: cell }} className="text-center font-mono text-[10px] text-slate-400">
            {t}
          </div>
        ))}
      </div>
      {map.map((row, i) => (
        <div key={i} className="flex">
          <div style={{ width: cell }} className="text-center font-mono text-[10px] leading-none text-slate-400 flex items-center justify-center">
            {tokens[i]}
          </div>
          {row.map((v, j) => (
            <div
              key={j}
              title={`${tokens[i]} → ${tokens[j]}: ${v.toFixed(3)}`}
              style={{ width: cell, height: cell, background: color, opacity: j > i ? 0.03 : 0.06 + 0.94 * Math.sqrt(v) }}
              className="border border-slate-950"
            />
          ))}
        </div>
      ))}
    </div>
  );
}

export function ema(points: [number, number][], alpha = 0.08): [number, number][] {
  let m = NaN;
  return points.map(([x, y]) => {
    m = Number.isNaN(m) ? y : m + alpha * (y - m);
    return [x, m];
  });
}
