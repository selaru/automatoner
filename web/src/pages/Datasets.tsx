import { useCallback, useEffect, useState } from "react";
import { Category, Dataset, TaskInfo, del, get, post, upload } from "../api";
import { Badge, Notice, Page, TaskView, useTask } from "../ui";

const TABULAR_SPEC = {
  rows: 5000, anomaly_rate: 0.03, seed: 1, label_column: "label",
  columns: [
    { name: "hour", kind: "uniform_int", low: 0, high: 23 },
    { name: "requests_per_min", kind: "seasonal", base: 200, amplitude: 120, period_column: "hour", period: 24, noise: 15, min: 0, anomaly: { mult: 3 } },
    { name: "error_rate", kind: "normal", mean: 0.01, std: 0.005, min: 0, max: 1, anomaly: { shift_std: 8 } },
  ],
};
const LABELS = [{ name: "widened", description: "The change grants more access." }, { name: "narrowed", description: "The change removes access." }, { name: "unchanged", description: "No effect on access." }];
const TOOLS = [{ name: "set_light", description: "Turn a light on or off.", parameters: { type: "object", properties: { room: { type: "string" }, on: { type: "boolean" } }, required: ["room"] } }];
const RUBRIC = "Return one JSON object with keys summary, title, category, todos.\n- summary: one sentence in the input's language\n- title: at most 7 words\n- category: one of Work, Personal, Creative, Health, Finance, Other\n- todos: exactly 5 concrete first steps, each starting with a verb";

function ProviderPick({ value, onChange }: { value: { provider: string; model: string }; onChange: (v: { provider: string; model: string }) => void }) {
  const [p, setP] = useState<any>(null);
  useEffect(() => { get("/api/providers").then((r) => { setP(r); if (!value.provider) onChange({ provider: r.default, model: "" }); }); }, []);
  if (!p) return null;
  return (
    <div className="grid cols-2">
      <label className="field"><span className="lbl">LLM that writes the data</span>
        <select value={value.provider} onChange={(e) => onChange({ provider: e.target.value, model: "" })}>
          {p.providers.map((x: any) => <option key={x.id} value={x.id} disabled={x.allowed === false}>{x.id}{x.allowed === false ? " (blocked by local-only mode)" : x.configured ? "" : " (no key)"}</option>)}</select></label>
      <label className="field"><span className="lbl">Model (blank = {p.providers.find((x: any) => x.id === value.provider)?.model})</span>
        <input value={value.model} onChange={(e) => onChange({ ...value, model: e.target.value })} /></label>
    </div>
  );
}

function SyntheticForm({ category, name, onStarted }: { category: string; name: string; onStarted: (t: TaskInfo) => void }) {
  const [prov, setProv] = useState({ provider: "", model: "" });
  const [json, setJson] = useState("");
  const [describe, setDescribe] = useState("");
  const [useAI, setUseAI] = useState(false);
  const [extra, setExtra] = useState<Record<string, any>>({ per_label: 100, total: 200, language: "English", style: "short, realistic, varied wording", rubric: RUBRIC, system_prompt: "Turn the note into the JSON card.", seeds: "", input_description: "a short spoken note", cells: '{"lang": {"en": {"desc": "English", "share": 1}}}', output_keys: "summary,title,category,todos" });
  const [err, setErr] = useState("");
  const tabular = category.startsWith("tabular");
  useEffect(() => {
    if (tabular) setJson(JSON.stringify(TABULAR_SPEC, null, 2));
    else if (category === "text_classification") setJson(JSON.stringify(LABELS, null, 2));
    else if (category === "tool_calling" || category === "extraction") setJson(JSON.stringify(TOOLS, null, 2));
    else setJson("");
  }, [category]);
  const noGen = ["speech", "embeddings", "vision"].includes(category);
  if (noGen) return <Notice>Synthetic data for {category.replace("_", " ")} is not available yet. Upload your own files instead.</Notice>;
  const set = (k: string, v: any) => setExtra({ ...extra, [k]: v });
  const go = async () => {
    setErr("");
    try {
      let spec: any;
      if (tabular) spec = useAI ? { describe, rows: Number(extra.total) || 5000 } : JSON.parse(json);
      else if (category === "text_classification") spec = { labels: JSON.parse(json), per_label: Number(extra.per_label), language: extra.language, style: extra.style };
      else if (category === "tool_calling" || category === "extraction") spec = { tools: JSON.parse(json), total: Number(extra.total), language: extra.language };
      else spec = { total: Number(extra.total), rubric: extra.rubric, system_prompt: extra.system_prompt, input_description: extra.input_description,
        cells: JSON.parse(extra.cells), output_keys: extra.output_keys.split(",").map((s: string) => s.trim()).filter(Boolean),
        seeds: extra.seeds.split("\n").filter((l: string) => l.trim()).map((l: string) => (l.trim().startsWith("{") ? JSON.parse(l) : { text: l.trim() })) };
      const needsLLM = !tabular || useAI;
      const r = await post("/api/datasets/synthetic", { name: name || "Synthetic dataset", category, spec, provider: needsLLM ? prov.provider : undefined, model: prov.model || undefined });
      onStarted(r.task);
    } catch (e: any) { setErr(e.message); }
  };
  return (
    <div className="stack">
      {tabular && <label className="check"><input type="checkbox" checked={useAI} onChange={(e) => setUseAI(e.target.checked)} />Let an AI design the simulator from a description</label>}
      {tabular && useAI ? (
        <label className="field"><span className="lbl">Describe the system and what normal and strange look like</span>
          <textarea style={{ fontFamily: "inherit" }} value={describe} onChange={(e) => setDescribe(e.target.value)} placeholder="Lambda function behind an ALB for a web shop: traffic peaks at lunchtime, 5xx errors are rare, p95 latency around 100 ms..." /></label>
      ) : category === "text_to_json" ? null : (
        <label className="field"><span className="lbl">{tabular ? "Simulator spec (JSON)" : category === "text_classification" ? "Labels (JSON)" : "Tools (JSON schema)"}</span>
          <textarea rows={12} value={json} onChange={(e) => setJson(e.target.value)} /></label>
      )}
      {category === "text_classification" && <div className="grid cols-3">
        <label className="field"><span className="lbl">Examples per label</span><input type="number" value={extra.per_label} onChange={(e) => set("per_label", e.target.value)} /></label>
        <label className="field"><span className="lbl">Language</span><input value={extra.language} onChange={(e) => set("language", e.target.value)} /></label>
        <label className="field"><span className="lbl">Style</span><input value={extra.style} onChange={(e) => set("style", e.target.value)} /></label></div>}
      {(category === "tool_calling" || category === "extraction") && <div className="grid cols-2">
        <label className="field"><span className="lbl">Total examples</span><input type="number" value={extra.total} onChange={(e) => set("total", e.target.value)} /></label>
        <label className="field"><span className="lbl">Language</span><input value={extra.language} onChange={(e) => set("language", e.target.value)} /></label></div>}
      {tabular && useAI && <label className="field"><span className="lbl">Rows</span><input type="number" value={extra.total} onChange={(e) => set("total", e.target.value)} /></label>}
      {category === "text_to_json" && (
        <div className="stack">
          <div className="grid cols-2">
            <label className="field"><span className="lbl">Total examples</span><input type="number" value={extra.total} onChange={(e) => set("total", e.target.value)} /></label>
            <label className="field"><span className="lbl">What the inputs are</span><input value={extra.input_description} onChange={(e) => set("input_description", e.target.value)} /></label></div>
          <label className="field"><span className="lbl">System prompt (identical in every example, used at inference)</span><input value={extra.system_prompt} onChange={(e) => set("system_prompt", e.target.value)} /></label>
          <label className="field"><span className="lbl">Rubric the target outputs must follow</span><textarea rows={7} value={extra.rubric} onChange={(e) => set("rubric", e.target.value)} /></label>
          <label className="field"><span className="lbl">Output keys in order (the generator rejects anything else)</span><input value={extra.output_keys} onChange={(e) => set("output_keys", e.target.value)} /></label>
          <label className="field"><span className="lbl">Balance cells (JSON: dimension → value → desc, share)</span><textarea rows={5} value={extra.cells} onChange={(e) => set("cells", e.target.value)} /></label>
          <label className="field"><span className="lbl">Seeds: real examples, one per line (plain text or JSON with a "text" field). Optional but they set the style.</span>
            <textarea rows={5} value={extra.seeds} onChange={(e) => set("seeds", e.target.value)} /></label>
        </div>)}
      {(!tabular || useAI) && <ProviderPick value={prov} onChange={setProv} />}
      {err && <Notice kind="bad">{err}</Notice>}
      <div><button className="primary" onClick={go}>Generate</button></div>
    </div>
  );
}

function NewDataset({ cats, onDone }: { cats: Category[]; onDone: () => void }) {
  const [mode, setMode] = useState<"upload" | "hf" | "synthetic">("synthetic");
  const [category, setCategory] = useState("tabular_anomaly");
  const [name, setName] = useState("");
  const [files, setFiles] = useState<FileList | null>(null);
  const [label, setLabel] = useState("");
  const [hf, setHf] = useState({ repo: "", split: "train", subset: "", max_rows: 50000, cols: "" });
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const task = useTask(taskId);
  useEffect(() => { if (task && task.status !== "running") onDone(); }, [task?.status]);
  const cat = cats.find((c) => c.id === category);
  const tabular = category.startsWith("tabular");
  const options = () => ({ ...(label ? { label_column: label } : {}), ...(hf.cols ? { columns: JSON.parse(hf.cols) } : {}) });
  const doUpload = async () => {
    setErr(""); setBusy(true);
    try {
      const f = new FormData(); f.append("name", name || files?.[0]?.name || "Uploaded dataset"); f.append("category", category); f.append("options", JSON.stringify(options()));
      Array.from(files ?? []).forEach((x) => f.append("files", x));
      await upload("/api/datasets/upload", f); onDone(); setFiles(null);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  const doHF = async () => {
    setErr(""); setBusy(true);
    try { await post("/api/datasets/hf", { name: name || hf.repo, category, repo: hf.repo, split: hf.split, subset: hf.subset || null, max_rows: hf.max_rows, options: options() }); onDone(); }
    catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="card stack">
      <div className="tabs" style={{ margin: 0 }}>
        {(["synthetic", "upload", "hf"] as const).map((m) => <div key={m} className={`tab ${m === mode ? "active" : ""}`} onClick={() => setMode(m)}>{{ synthetic: "Generate with AI", upload: "Upload files", hf: "Hugging Face dataset" }[m]}</div>)}
      </div>
      <div className="grid cols-2">
        <label className="field"><span className="lbl">For</span><select value={category} onChange={(e) => setCategory(e.target.value)}>{cats.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}</select></label>
        <label className="field"><span className="lbl">Name</span><input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. checkout service, normal weeks" /></label>
      </div>
      {cat && <div className="small muted">Expected data: {cat.dataset}</div>}
      {mode === "synthetic" && !taskId && <SyntheticForm category={category} name={name} onStarted={(t) => setTaskId(t.id)} />}
      {mode === "synthetic" && taskId && <div className="stack"><TaskView task={task} />{task && task.status !== "running" && <div><button onClick={() => setTaskId(null)}>Make another</button></div>}</div>}
      {mode === "upload" && (
        <div className="stack">
          <input type="file" multiple onChange={(e) => setFiles(e.target.files)} />
          <div className="small muted">CSV, TSV, Parquet, JSON or JSONL. For speech, upload a zip with the audio files and a JSONL of audio, text and language.</div>
          {tabular && <label className="field"><span className="lbl">Label column (blank = detect label, is_anomaly, target...)</span><input value={label} onChange={(e) => setLabel(e.target.value)} /></label>}
          <div><button className="primary" disabled={!files?.length || busy} onClick={doUpload}>{busy ? "Importing..." : "Import"}</button></div>
        </div>)}
      {mode === "hf" && (
        <div className="stack">
          <div className="grid cols-2">
            <label className="field"><span className="lbl">Dataset repo</span><input value={hf.repo} placeholder="owner/name" onChange={(e) => setHf({ ...hf, repo: e.target.value })} /></label>
            <label className="field"><span className="lbl">Split</span><input value={hf.split} onChange={(e) => setHf({ ...hf, split: e.target.value })} /></label>
            <label className="field"><span className="lbl">Subset / config (optional)</span><input value={hf.subset} onChange={(e) => setHf({ ...hf, subset: e.target.value })} /></label>
            <label className="field"><span className="lbl">Max rows</span><input type="number" value={hf.max_rows} onChange={(e) => setHf({ ...hf, max_rows: Number(e.target.value) })} /></label>
          </div>
          <label className="field"><span className="lbl">Rename columns (JSON, target: source), e.g. {`{"text": "sentence", "label": "sentiment"}`}</span><input value={hf.cols} onChange={(e) => setHf({ ...hf, cols: e.target.value })} /></label>
          {tabular && <label className="field"><span className="lbl">Label column</span><input value={label} onChange={(e) => setLabel(e.target.value)} /></label>}
          <div><button className="primary" disabled={!hf.repo || busy} onClick={doHF}>{busy ? "Downloading..." : "Import"}</button></div>
        </div>)}
      {err && <Notice kind="bad">{err}</Notice>}
    </div>
  );
}

function DatasetRow({ d, reload }: { d: Dataset; reload: () => void }) {
  const [open, setOpen] = useState(false);
  const [prev, setPrev] = useState<any>(null);
  useEffect(() => { if (open && !prev && d.status === "ready") get(`/api/datasets/${d.id}/preview?limit=15`).then(setPrev); }, [open]);
  useEffect(() => { if (d.status !== "generating") return; const t = setInterval(reload, 2000); return () => clearInterval(t); }, [d.status]);
  const kind = d.status === "ready" ? "ok" : d.status === "generating" ? "accent" : "bad";
  return (
    <div className="card stack">
      <div className="row between">
        <div><b>{d.name}</b> <span className="small muted">{d.category.replace(/_/g, " ")} · {d.source}</span></div>
        <div className="row"><Badge kind={kind}>{d.status}</Badge>{d.rows !== undefined && <Badge>{d.rows} rows</Badge>}
          {d.splits && <Badge>{d.splits.train}/{d.splits.valid}/{d.splits.test} train/valid/test</Badge>}
          <button className="small" onClick={() => setOpen(!open)}>{open ? "Hide" : "Details"}</button>
          <button className="small danger" onClick={async () => { await del(`/api/datasets/${d.id}`); reload(); }}>Delete</button></div>
      </div>
      {d.error && <Notice kind="bad">{d.error}</Notice>}
      {d.report?.errors?.map((e, i) => <Notice key={i} kind="bad">{e}</Notice>)}
      {open && (
        <div className="stack">
          {d.report?.warnings?.map((w, i) => <Notice key={i} kind="warn">{w}</Notice>)}
          {d.label_column && <div className="small">Label column: <code>{d.label_column}</code></div>}
          {prev && prev.columns && <div className="tablewrap"><table><thead><tr>{prev.columns.map((c: string) => <th key={c}>{c}</th>)}</tr></thead>
            <tbody>{prev.rows.map((r: any[], i: number) => <tr key={i}>{r.map((v, j) => <td key={j}>{String(v)}</td>)}</tr>)}</tbody></table></div>}
          {prev && !prev.columns && <pre className="log">{prev.rows.slice(0, 6).map((r: any) => JSON.stringify(r, null, 1)).join("\n\n")}</pre>}
        </div>)}
    </div>
  );
}

export default function Datasets() {
  const [cats, setCats] = useState<Category[]>([]);
  const [list, setList] = useState<Dataset[]>([]);
  const reload = useCallback(() => { get("/api/datasets").then(setList); }, []);
  useEffect(() => { get("/api/categories").then(setCats); reload(); }, []);
  return (
    <Page title="Datasets" sub="Bring your own files, pull one from Hugging Face, or have an AI write one. Every dataset is checked and split into train, validation and test.">
      {cats.length > 0 && <NewDataset cats={cats} onDone={reload} />}
      <h2>Your datasets</h2>
      <div className="stack">{list.map((d) => <DatasetRow key={d.id} d={d} reload={reload} />)}{!list.length && <p className="muted">None yet.</p>}</div>
    </Page>
  );
}
