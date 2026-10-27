import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Category, Dataset, FieldDef, Job, Model, get, post } from "../api";
import { Badge, FieldForm, Notice, Page, Progress } from "../ui";

export default function Train() {
  const nav = useNavigate();
  const [cats, setCats] = useState<Category[]>([]);
  const [cat, setCat] = useState("");
  const [models, setModels] = useState<Model[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [trainers, setTrainers] = useState<any[]>([]);
  const [modelId, setModelId] = useState("");
  const [dsId, setDsId] = useState("");
  const [trainer, setTrainer] = useState("");
  const [fields, setFields] = useState<FieldDef[]>([]);
  const [cfg, setCfg] = useState<Record<string, any>>({});
  const [jobs, setJobs] = useState<Job[]>([]);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const loadJobs = () => get("/api/jobs").then(setJobs);
  useEffect(() => { get("/api/categories").then((c: Category[]) => { setCats(c); setCat(c[0].id); }); get("/api/trainers").then(setTrainers); loadJobs(); }, []);
  useEffect(() => { const t = setInterval(loadJobs, 3000); return () => clearInterval(t); }, []);
  const loadModels = () => cat && get(`/api/models?category=${cat}`).then((ms: Model[]) => { setModels(ms); const first = ms.find((m) => m.trainable[cat]); setModelId((cur) => (ms.some((m) => m.id === cur) ? cur : first?.id ?? "")); });
  useEffect(() => {
    if (!cat) return;
    loadModels();
    get(`/api/datasets?category=${cat}`).then((ds: Dataset[]) => { const ready = ds.filter((d) => d.status === "ready"); setDatasets(ready); setDsId(ready[0]?.id ?? ""); });
  }, [cat]);

  const model = models.find((m) => m.id === modelId);
  const options = model?.trainer_options[cat] ?? [];
  useEffect(() => { setTrainer(options[0] ?? ""); }, [modelId, cat, models.length]);
  useEffect(() => {
    if (!trainer || !modelId) { setFields([]); return; }
    get(`/api/trainers/${trainer}/schema?category=${cat}&model_id=${modelId}`).then((f: FieldDef[]) => { setFields(f); setCfg({}); }).catch(() => setFields([]));
  }, [trainer, modelId, cat]);

  const tinfo = trainers.find((t) => t.id === trainer);
  const downloading = model?.state.status === "downloading";
  const needsDownload = model && model.state.status === "not_downloaded";
  useEffect(() => { if (!downloading) return; const t = setInterval(loadModels, 1500); return () => clearInterval(t); }, [downloading]);

  const start = async () => {
    setErr(""); setBusy(true);
    try { const j = await post<Job>("/api/jobs", { model_id: modelId, dataset_id: dsId, trainer, category: cat, config: cfg }); nav(`/jobs/${j.id}`); }
    catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  return (
    <Page title="Train & export" sub="Choose what to teach, which base model to start from, and which data to use.">
      <div className="tabs">{cats.map((c) => <div key={c.id} className={`tab ${c.id === cat ? "active" : ""}`} onClick={() => setCat(c.id)}>{c.name}<span className="n">{c.trainable}</span></div>)}</div>
      <div className="card stack">
        <div className="grid cols-2">
          <label className="field"><span className="lbl">Base model</span>
            <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {models.map((m) => <option key={m.id} value={m.id} disabled={!m.trainable[cat]}>{m.name}{m.trainable[cat] ? "" : " (training planned)"}</option>)}</select></label>
          <label className="field"><span className="lbl">Dataset</span>
            <select value={dsId} onChange={(e) => setDsId(e.target.value)}>
              {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.rows} rows)</option>)}</select></label>
        </div>
        {!datasets.length && <Notice kind="warn">No ready dataset for this category. <Link to="/datasets">Create one</Link>.</Notice>}
        {needsDownload && <Notice kind="warn"><div className="row between"><span>{model!.name} is not downloaded yet (about {model!.size_mb} MB).</span>
          <button className="primary" onClick={async () => { await post(`/api/models/${modelId}/download`); loadModels(); }}>Download</button></div></Notice>}
        {downloading && <div><Progress value={model!.state.progress || undefined} /><div className="small muted">{model!.state.message}</div></div>}
        {options.length > 1 && (
          <label className="field"><span className="lbl">Trainer</span><select value={trainer} onChange={(e) => setTrainer(e.target.value)}>
            {options.map((o) => <option key={o} value={o}>{trainers.find((t) => t.id === o)?.name ?? o}</option>)}</select></label>)}
        {tinfo && !tinfo.ready && <Notice kind="warn">
          {!tinfo.host_ok ? `${tinfo.name} does not run on this computer.` : <>{tinfo.name} needs the <b>{tinfo.extra}</b> extra (missing {tinfo.missing.join(", ")}). <Link to="/doctor">Install it from the Doctor page</Link>.</>}</Notice>}
        {tinfo && <div className="small muted">Runs on: {tinfo.hardware}</div>}
        {fields.length > 0 && <><h3>Settings</h3><FieldForm fields={fields} values={cfg} onChange={setCfg} /></>}
        {err && <Notice kind="bad">{err}</Notice>}
        <div><button className="primary" disabled={busy || !modelId || !dsId || !tinfo?.ready || needsDownload || downloading} onClick={start}>{busy ? "Starting..." : "Start training"}</button></div>
      </div>

      <h2>Jobs</h2>
      <div className="tablewrap"><table>
        <thead><tr><th>Started</th><th>Model</th><th>Dataset</th><th>Trainer</th><th>Status</th><th /></tr></thead>
        <tbody>{jobs.map((j) => (
          <tr key={j.id}><td>{(j.started || j.created).replace("T", " ").slice(0, 16)}</td><td>{j.model_name}</td><td>{j.dataset_name}</td><td>{j.trainer}</td>
            <td><Badge kind={j.status === "done" ? "ok" : j.status === "error" ? "bad" : j.status === "running" ? "accent" : "warn"}>{j.status}</Badge></td>
            <td><Link to={`/jobs/${j.id}`}>Open</Link></td></tr>))}
          {!jobs.length && <tr><td colSpan={6} className="muted">No jobs yet.</td></tr>}</tbody></table></div>
    </Page>
  );
}
