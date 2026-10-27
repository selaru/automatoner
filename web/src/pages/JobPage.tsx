import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Job, del, get, post, wsUrl } from "../api";
import { Badge, Chart, FieldForm, LogBox, Notice, Page, TaskView, copy, fmtBytes, fmtNum, useTask } from "../ui";

function MetricsTable({ metrics }: { metrics: Record<string, Record<string, any>> }) {
  const splits = Object.keys(metrics);
  const keys = Array.from(new Set(splits.flatMap((s) => Object.keys(metrics[s]).filter((k) => typeof metrics[s][k] === "number"))));
  if (!keys.length) return null;
  return (
    <div className="tablewrap"><table>
      <thead><tr><th>Metric</th>{splits.map((s) => <th key={s}>{s}</th>)}</tr></thead>
      <tbody>{keys.map((k) => <tr key={k}><td>{k.replace(/_/g, " ")}</td>{splits.map((s) => <td key={s}>{metrics[s][k] !== undefined ? fmtNum(metrics[s][k]) : ""}</td>)}</tr>)}</tbody></table></div>
  );
}

function ExportPanel({ job }: { job: Job }) {
  const [opts, setOpts] = useState<any[]>([]);
  const [sel, setSel] = useState<any>(null);
  const [vals, setVals] = useState<Record<string, any>>({});
  const [taskId, setTaskId] = useState<string | null>(null);
  const [exportId, setExportId] = useState("");
  const [err, setErr] = useState("");
  const task = useTask(taskId);
  useEffect(() => { get(`/api/jobs/${job.id}/exports`).then((o) => { setOpts(o); setSel(o.find((x: any) => x.available) ?? null); }); }, [job.id]);
  const run = async () => {
    setErr("");
    try { const r = await post("/api/exports", { job_id: job.id, exporter: sel.id, options: vals }); setTaskId(r.task.id); setExportId(r.export_id); }
    catch (e: any) { setErr(e.message); }
  };
  const result = task?.status === "done" ? task.result : null;
  return (
    <div className="stack">
      <div className="grid auto">{opts.map((o) => (
        <div key={o.id} className="card" style={{ cursor: o.available ? "pointer" : "default", opacity: o.available ? 1 : 0.55, outline: sel?.id === o.id ? "2px solid var(--accent)" : "none" }}
          onClick={() => o.available && (setSel(o), setVals({}))}>
          <div className="row between"><b>{o.name}</b>{!o.available && <Badge>unavailable</Badge>}</div>
          <div className="small muted">{o.available ? o.blurb : o.reason}</div>
        </div>))}</div>
      {sel && sel.options.length > 0 && <FieldForm fields={sel.options} values={vals} onChange={setVals} />}
      {sel && <div className="row"><button className="primary" disabled={task?.status === "running"} onClick={run}>Export: {sel.name}</button></div>}
      {err && <Notice kind="bad">{err}</Notice>}
      <TaskView task={task} showLog={task?.status === "error"} />
      {result && (
        <div className="card stack">
          <div className="row between"><b>Export ready</b>{result.download && <a className="btn primary" href={`/api/exports/${exportId}/download`}>Download zip</a>}</div>
          {result.location && <div className="small">Saved on this computer at <code>{result.location}</code></div>}
          {result.smoke_test && <div><div className="small muted">Try it:</div><div className="row"><code>{result.smoke_test}</code><button className="small" onClick={() => copy(result.smoke_test)}>Copy</button></div></div>}
          {result.notes && <Notice>{result.notes}</Notice>}
          {result.files && <div className="tablewrap"><table><tbody>{result.files.map((f: any) => <tr key={f.path}><td><code>{f.path}</code></td><td>{fmtBytes(f.bytes)}</td></tr>)}</tbody></table></div>}
        </div>)}
    </div>
  );
}

function Report({ id }: { id: string }) {
  const [md, setMd] = useState<string | null>(null);
  useEffect(() => { get(`/api/jobs/${id}/report`).then((r) => setMd(r.markdown)); }, [id]);
  if (!md) return null;
  return <><h2>Evaluation report</h2><pre className="log" style={{ maxHeight: 480 }}>{md}</pre></>;
}

function hasCurve(series?: Record<string, any>[]) {
  if (!series) return false;
  const keys = new Set(series.flatMap((s) => Object.keys(s)).filter((k) => /loss/.test(k)));
  return Array.from(keys).some((k) => series.filter((s) => typeof s[k] === "number").length > 1);
}

export default function JobPage() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [log, setLog] = useState("");
  const wsRef = useRef<WebSocket | null>(null);
  const load = () => get<Job>(`/api/jobs/${id}`).then(setJob);
  useEffect(() => {
    load();
    setLog("");
    const ws = new WebSocket(wsUrl(`/ws/jobs/${id}`));
    wsRef.current = ws;
    ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.type === "log") setLog((l) => l + m.text); if (m.type === "status") load(); };
    ws.onclose = () => load();
    return () => ws.close();
  }, [id]);
  if (!job) return null;
  const running = job.status === "running" || job.status === "queued";
  const metrics = job.manifest?.metrics;
  const badge = job.status === "done" ? "ok" : job.status === "error" ? "bad" : running ? "accent" : "warn";
  return (
    <Page title={`${job.model_name} on ${job.dataset_name}`} sub={`${job.category.replace(/_/g, " ")} · ${job.trainer} · ${job.id}`}
      actions={<div className="row"><Badge kind={badge}>{job.status}</Badge>
        {running && <button className="danger" onClick={async () => { await post(`/api/jobs/${id}/cancel`); load(); }}>Cancel</button>}
        {!running && <button className="danger" onClick={async () => { await del(`/api/jobs/${id}`); nav("/train"); }}>Delete</button>}</div>}>
      {job.error && <Notice kind="bad">{job.error}</Notice>}
      {hasCurve(job.series) && <><h2 style={{ marginTop: 0 }}>Progress</h2><div className="card"><Chart series={job.series!} /></div></>}
      {metrics && Object.keys(metrics).length > 0 && <><h2>Results</h2><MetricsTable metrics={metrics} />
        {job.manifest?.size_bytes && <p className="small muted">Model files: {fmtBytes(job.manifest.size_bytes)}{job.manifest.onnx_bytes ? `, ONNX ${fmtBytes(job.manifest.onnx_bytes)}` : ""}</p>}
        {job.manifest?.threshold_source && <p className="small muted">Alert threshold chosen by: {job.manifest.threshold_source}.</p>}</>}
      {job.manifest?.artifacts?.report && <Report id={id} />}
      {job.status === "done" && <><h2>Export</h2><ExportPanel job={job} /></>}
      <h2>Log</h2>
      <LogBox text={log} />
      <h2>Settings used</h2>
      <div className="card"><dl className="kv">{Object.entries(job.config).map(([k, v]) => <><dt key={k + "k"}>{k}</dt><dd key={k + "v"}>{typeof v === "object" ? JSON.stringify(v) : String(v)}</dd></>)}</dl></div>
      <p><Link to="/train">Back to training</Link></p>
    </Page>
  );
}
