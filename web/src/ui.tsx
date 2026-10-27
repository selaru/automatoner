import React, { useEffect, useRef, useState } from "react";
import { FieldDef, TaskInfo, get, wsUrl } from "./api";

export function Badge({ kind, children }: { kind?: "ok" | "warn" | "bad" | "accent"; children: React.ReactNode }) {
  return <span className={`badge ${kind ?? ""}`}>{children}</span>;
}

export function Progress({ value }: { value?: number }) {
  if (value === undefined) return <div className="progress indet"><div /></div>;
  return <div className="progress"><div style={{ width: `${Math.round(value * 100)}%` }} /></div>;
}

export function Page({ title, sub, children, actions }: { title: string; sub?: string; children: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <>
      <div className="row between"><h1>{title}</h1>{actions}</div>
      {sub && <p className="sub">{sub}</p>}
      {children}
    </>
  );
}

export function Notice({ kind, children }: { kind?: "warn" | "bad" | "ok"; children: React.ReactNode }) {
  return <div className={`notice ${kind ?? ""}`}>{children}</div>;
}

export function LogBox({ text, follow = true }: { text: string; follow?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [text, follow]);
  const shown = text.split("\n").filter((l) => !l.startsWith("@@ ")).join("\n");
  return <div className="log" ref={ref}>{shown || "Waiting for output..."}</div>;
}

/** Follow a background task over a WebSocket, falling back to polling. */
export function useTask(taskId: string | null) {
  const [task, setTask] = useState<TaskInfo | null>(null);
  useEffect(() => {
    if (!taskId) { setTask(null); return; }
    let closed = false;
    let timer: number | undefined;
    const poll = async () => {
      try { const t = await get<TaskInfo>(`/api/tasks/${taskId}`); if (!closed) setTask(t); if (!closed && t.status === "running") timer = window.setTimeout(poll, 800); } catch { /* ignore */ }
    };
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl(`/ws/tasks/${taskId}`));
      ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.type === "task") setTask(m.task); };
      ws.onerror = () => { ws?.close(); poll(); };
    } catch { poll(); }
    return () => { closed = true; ws?.close(); if (timer) clearTimeout(timer); };
  }, [taskId]);
  return task;
}

export function TaskView({ task, showLog = true }: { task: TaskInfo | null; showLog?: boolean }) {
  if (!task) return null;
  return (
    <div className="stack">
      <div className="row between"><b>{task.title}</b>
        <Badge kind={task.status === "done" ? "ok" : task.status === "error" ? "bad" : "accent"}>{task.status}</Badge></div>
      {task.status === "running" && <Progress value={task.progress > 0 ? task.progress : undefined} />}
      {task.message && <div className="small muted">{task.message}</div>}
      {task.error && <Notice kind="bad">{task.error}</Notice>}
      {showLog && task.log.length > 0 && <LogBox text={task.log.join("\n")} />}
    </div>
  );
}

export function fmtBytes(n?: number) {
  if (n === undefined || n === null) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function fmtNum(v: any) {
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return String(v);
  return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3);
}

/** Render a form from a trainer or exporter field schema. */
export function FieldForm({ fields, values, onChange }: { fields: FieldDef[]; values: Record<string, any>; onChange: (v: Record<string, any>) => void }) {
  const [adv, setAdv] = useState(false);
  const set = (k: string, v: any) => onChange({ ...values, [k]: v });
  const shown = fields.filter((f) => adv || !f.advanced);
  return (
    <div className="stack">
      <div className="grid cols-2">
        {shown.map((f) => {
          const v = values[f.key] ?? f.default;
          if (f.type === "bool")
            return (
              <label key={f.key} className="check"><input type="checkbox" checked={!!v} onChange={(e) => set(f.key, e.target.checked)} />
                <span>{f.label}{f.help && <span className="help"> · {f.help}</span>}</span></label>
            );
          return (
            <label key={f.key} className="field"><span className="lbl">{f.label}</span>
              {f.type === "select" ? (
                <select value={v} onChange={(e) => set(f.key, e.target.value)}>{(f.options ?? []).map((o) => <option key={String(o)} value={o}>{String(o)}</option>)}</select>
              ) : (
                <input type={f.type === "text" ? "text" : "number"} step={f.type === "float" ? "any" : 1} min={f.min} max={f.max} value={v ?? ""}
                  onChange={(e) => set(f.key, f.type === "text" ? e.target.value : e.target.value === "" ? "" : Number(e.target.value))} />
              )}
              {f.help && <span className="help">{f.help}</span>}
            </label>
          );
        })}
      </div>
      {fields.some((f) => f.advanced) && <button className="ghost small" onClick={() => setAdv(!adv)}>{adv ? "Hide" : "Show"} advanced</button>}
    </div>
  );
}

export function Chart({ series, keys }: { series: Record<string, any>[]; keys?: string[] }) {
  const pick = keys ?? Array.from(new Set(series.flatMap((s) => Object.keys(s)))).filter((k) => /loss/.test(k));
  const colors = ["#2f7d5b", "#c0711a", "#3b6fb6", "#a43b73"];
  const pts = pick.map((k) => series.map((s, i) => ({ x: typeof s.step === "number" ? s.step : i, y: s[k] })).filter((p) => typeof p.y === "number"));
  const all = pts.flat();
  if (all.length < 2) return null;
  const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const W = 640, H = 190, pad = 30;
  const sx = (x: number) => pad + ((x - x0) / (x1 - x0 || 1)) * (W - pad - 8);
  const sy = (y: number) => H - 22 - ((y - y0) / (y1 - y0 || 1)) * (H - 40);
  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <line x1={pad} x2={W - 8} y1={H - 22} y2={H - 22} stroke="currentColor" opacity="0.2" />
        <text x={4} y={14} fontSize="11" fill="currentColor" opacity="0.6">{y1.toFixed(3)}</text>
        <text x={4} y={H - 24} fontSize="11" fill="currentColor" opacity="0.6">{y0.toFixed(3)}</text>
        {pts.map((p, i) => <polyline key={i} fill="none" stroke={colors[i % colors.length]} strokeWidth="2" points={p.map((q) => `${sx(q.x)},${sy(q.y)}`).join(" ")} />)}
      </svg>
      <div className="legend">{pick.map((k, i) => <span key={k}><i style={{ background: colors[i % colors.length] }} />{k}</span>)}</div>
    </div>
  );
}

export function copy(text: string) {
  navigator.clipboard?.writeText(text).catch(() => undefined);
}
