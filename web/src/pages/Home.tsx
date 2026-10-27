import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { get, post, TaskInfo } from "../api";
import { Badge, Notice, Page, TaskView, useTask } from "../ui";

function RecipeCard({ r }: { r: any }) {
  const [open, setOpen] = useState(false);
  const [params, setParams] = useState<Record<string, any>>({});
  const [providers, setProviders] = useState<any>(null);
  const [provider, setProvider] = useState("");
  const [taskId, setTaskId] = useState<string | null>(null);
  const [err, setErr] = useState("");
  const task = useTask(taskId);

  useEffect(() => { if (open && r.needs_llm && !providers) get("/api/providers").then((p) => { setProviders(p); setProvider(p.default); }); }, [open]);

  const run = async () => {
    setErr("");
    try { const t = await post<TaskInfo>(`/api/recipes/${r.id}/run`, { params, provider: r.needs_llm ? provider : undefined }); setTaskId(t.id); }
    catch (e: any) { setErr(e.message); }
  };

  return (
    <div className="card stack">
      <div className="row between"><h3>{r.name}</h3>{r.needs_llm ? <Badge kind="warn">needs an LLM</Badge> : <Badge kind="ok">runs offline</Badge>}</div>
      <p className="small muted" style={{ margin: 0 }}>{r.blurb}</p>
      <div className="chips"><Badge>{r.category.replace(/_/g, " ")}</Badge><Badge>{r.model_name}</Badge>{r.export && <Badge>→ {r.export}</Badge>}</div>
      {!open && !taskId && <div><button className="primary" onClick={() => setOpen(true)}>Set up and run</button></div>}
      {open && !taskId && (
        <div className="stack">
          {(r.params ?? []).map((p: any) => (
            <label key={p.key} className="field"><span className="lbl">{p.label}</span>
              <input value={params[p.key] ?? p.default ?? ""} onChange={(e) => setParams({ ...params, [p.key]: e.target.value })} /></label>
          ))}
          {r.needs_llm && providers && (
            <label className="field"><span className="lbl">LLM that writes the data</span>
              <select value={provider} onChange={(e) => setProvider(e.target.value)}>
                {providers.providers.map((p: any) => <option key={p.id} value={p.id} disabled={!p.configured}>{p.id}{p.configured ? ` (${p.model})` : " (no key)"}</option>)}
              </select></label>
          )}
          {r.model_state === "not_downloaded" && <Notice>{r.model_name} will be downloaded first.</Notice>}
          {err && <Notice kind="bad">{err}</Notice>}
          <div className="row"><button className="primary" onClick={run}>Run pipeline</button><button onClick={() => setOpen(false)}>Cancel</button></div>
        </div>
      )}
      {taskId && (
        <div className="stack">
          <TaskView task={task} />
          {task?.status === "done" && (
            <div className="row">
              <Link className="btn" to={`/jobs/${task.result.job_id}`}>Open training job</Link>
              {task.result.export_id && <a className="btn primary" href={`/api/exports/${task.result.export_id}/download`}>Download export</a>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [recipes, setRecipes] = useState<any[]>([]);
  useEffect(() => { get("/api/recipes").then(setRecipes); }, []);
  return (
    <Page title="Tiny models for one job each" sub="Pick a base model, bring or generate a dataset, train, and export for the place it will run. Recipes below are ready-made starting points.">
      <div className="steps">
        <Link className="step" to="/doctor"><b>Check setup</b><span>Everything installed?</span></Link>
        <Link className="step" to="/models"><b>Choose a model</b><span>By what it can learn</span></Link>
        <Link className="step" to="/datasets"><b>Build a dataset</b><span>Your files or AI-generated</span></Link>
        <Link className="step" to="/train"><b>Train</b><span>CPU, CUDA or Apple Silicon</span></Link>
        <Link className="step" to="/train"><b>Export</b><span>Lambda, Docker, device</span></Link>
      </div>
      <h2 style={{ marginTop: 0 }}>Recipes</h2>
      <div className="grid auto">{recipes.map((r) => <RecipeCard key={r.id} r={r} />)}</div>
    </Page>
  );
}
