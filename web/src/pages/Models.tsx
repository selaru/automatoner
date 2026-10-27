import { useCallback, useEffect, useState } from "react";
import { Category, Model, del, get, post } from "../api";
import { Badge, Notice, Page, Progress, fmtBytes } from "../ui";

function ModelCard({ m, cat, reload }: { m: Model; cat: string; reload: () => void }) {
  const st = m.state;
  const trainers = m.trainer_options[cat] ?? [];
  const [err, setErr] = useState("");
  const start = async () => { setErr(""); try { await post(`/api/models/${m.id}/download`); reload(); } catch (e: any) { setErr(e.message); } };
  useEffect(() => { if (st.status !== "downloading") return; const t = setInterval(reload, 1200); return () => clearInterval(t); }, [st.status]);
  return (
    <div className="card stack">
      <div className="row between"><h3>{m.name}</h3>
        {st.status === "downloaded" || st.status === "builtin" ? <Badge kind="ok">{st.status === "builtin" ? "built in" : "downloaded"}</Badge> : null}</div>
      <p className="small muted" style={{ margin: 0 }}>{m.good_for}</p>
      <div className="chips">
        <Badge>{m.params}</Badge><Badge>{m.size_mb} MB</Badge><Badge>{m.license}</Badge>
        {m.gated && <Badge kind="warn">gated</Badge>}
        {trainers.length ? <Badge kind="accent">train: {trainers.join(" / ")}</Badge> : <Badge>training planned</Badge>}
      </div>
      <div className="chips">{m.runs_on.map((r) => <Badge key={r}>{r}</Badge>)}</div>
      {m.notes && <div className="small muted">{m.notes}</div>}
      {m.download_note && <div className="small muted">{m.download_note}</div>}
      {st.status === "not_downloaded" && <div className="row"><button className="primary" onClick={start}>Download</button><span className="small muted">about {m.size_mb} MB from huggingface.co/{m.repo}</span></div>}
      {st.status === "downloading" && <><Progress value={st.progress || undefined} /><div className="small muted">{st.message || "starting..."}</div></>}
      {st.status === "downloaded" && <div className="row between"><span className="small muted">{fmtBytes(st.bytes)} on disk</span>
        <button className="small danger" onClick={async () => { if (confirm(`Delete ${m.name} from the Hugging Face cache on this computer?`)) { await del(`/api/models/${m.id}`); reload(); } }}>Remove</button></div>}
      {st.status === "error" && <Notice kind="bad">{st.error}<div><button className="small" onClick={start}>Retry</button></div></Notice>}
      {err && <Notice kind="bad">{err}</Notice>}
    </div>
  );
}

export default function Models() {
  const [cats, setCats] = useState<Category[]>([]);
  const [cat, setCat] = useState<string>("");
  const [models, setModels] = useState<Model[]>([]);
  useEffect(() => { get("/api/categories").then((c: Category[]) => { setCats(c); setCat(new URLSearchParams(location.search).get("category") || c[0]?.id); }); }, []);
  const reload = useCallback(() => { if (cat) get(`/api/models?category=${cat}`).then(setModels); }, [cat]);
  useEffect(reload, [reload]);
  const current = cats.find((c) => c.id === cat);
  return (
    <Page title="Models" sub="Grouped by what they can be trained to do. Nothing is downloaded until you press Download.">
      <div className="tabs">{cats.map((c) => <div key={c.id} className={`tab ${c.id === cat ? "active" : ""}`} onClick={() => setCat(c.id)}>{c.name}<span className="n">{c.models}</span></div>)}</div>
      {current && <p className="muted">{current.blurb} <span className="small">Dataset: {current.dataset}</span></p>}
      <div className="grid auto">{models.map((m) => <ModelCard key={m.id} m={m} cat={cat} reload={reload} />)}</div>
    </Page>
  );
}
