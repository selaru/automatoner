import { useEffect, useState } from "react";
import { get, post, TaskInfo } from "../api";
import { Badge, Notice, Page, TaskView, copy, useTask } from "../ui";

const groups: Record<string, string> = { system: "System", python: "Python packages", tools: "Tools", accounts: "Accounts" };

export default function Doctor() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [err, setErr] = useState("");
  const task = useTask(taskId);
  const load = () => { setLoading(true); get("/api/doctor").then(setData).finally(() => setLoading(false)); };
  useEffect(load, []);
  useEffect(() => { if (task && task.status !== "running") load(); }, [task?.status]);

  const fix = async (c: any) => {
    setErr("");
    const f = c.fix;
    try {
      let t: TaskInfo | null = null;
      if (f.type === "extra") t = await post(`/api/doctor/fix/extra/${f.extra}`);
      else if (f.type === "action") t = await post(`/api/doctor/fix/action/${f.action}`);
      if (t) setTaskId(t.id);
    } catch (e: any) { setErr(e.message); }
  };

  return (
    <Page title="Doctor" sub="Checks that everything needed to train and export is installed, and installs what is missing."
      actions={<button onClick={load} disabled={loading}>{loading ? "Checking..." : "Check again"}</button>}>
      {data && (
        <div className="row" style={{ marginBottom: 16 }}>
          <Badge kind="ok">{data.summary.ok} ok</Badge>
          {data.summary.warn > 0 && <Badge kind="warn">{data.summary.warn} warnings</Badge>}
          {data.summary.missing > 0 && <Badge kind="accent">{data.summary.missing} optional parts not installed</Badge>}
          {data.summary.fail > 0 && <Badge kind="bad">{data.summary.fail} problems</Badge>}
        </div>
      )}
      {err && <Notice kind="bad">{err}</Notice>}
      {task && <div className="card" style={{ marginBottom: 16 }}><TaskView task={task} /></div>}
      {data && Object.entries(groups).map(([g, label]) => {
        const items = data.checks.filter((c: any) => c.group === g);
        if (!items.length) return null;
        return (
          <div key={g}>
            <h2>{label}</h2>
            <div className="card" style={{ padding: "4px 16px" }}>
              {items.map((c: any) => (
                <div className="check-row" key={c.id}>
                  <span className={`dot ${c.status}`} />
                  <b>{c.name}</b>
                  <span className="small muted">{c.detail}</span>
                  <span>
                    {c.fix?.type === "extra" || c.fix?.type === "action" ? <button className="small primary" disabled={task?.status === "running"} onClick={() => fix(c)}>{c.fix.label}</button> : null}
                    {c.fix?.type === "link" && <a className="btn small" href={c.fix.url} target="_blank" rel="noreferrer">{c.fix.label}</a>}
                    {c.fix?.type === "command" && <button className="small" title={c.fix.command} onClick={() => copy(c.fix.command)}>Copy: {c.fix.label}</button>}
                  </span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </Page>
  );
}
