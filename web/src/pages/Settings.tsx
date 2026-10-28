import { useEffect, useState } from "react";
import { get, post, put } from "../api";
import { Badge, Notice, Page } from "../ui";

const KEYS: [string, string][] = [["anthropic", "Anthropic (Claude)"], ["gemini", "Google (Gemini)"], ["openai", "OpenAI"], ["openrouter", "OpenRouter"]];

export default function Settings() {
  const [s, setS] = useState<any>(null);
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [msg, setMsg] = useState<Record<string, string>>({});
  const [ollama, setOllama] = useState<string[]>([]);
  const [audit, setAudit] = useState<any[]>([]);
  const [saveErr, setSaveErr] = useState("");
  const load = () => { get("/api/settings").then(setS); get("/api/providers").then((p) => setOllama(p.ollama_models)); get("/api/audit?limit=50").then(setAudit); };
  useEffect(load, []);
  if (!s) return null;
  const save = async (patch: Record<string, any>) => {
    try { setSaveErr(""); setS(await put("/api/settings", patch)); get("/api/audit?limit=50").then(setAudit); } catch (e: any) { setSaveErr(e.message); }
  };
  const saveSecret = async (k: string) => { await save({ [k]: secrets[k] }); setSecrets({ ...secrets, [k]: "" }); setMsg({ ...msg, [k]: "Saved." }); };
  const test = async (name: string) => {
    setMsg({ ...msg, [name]: "Testing..." });
    const r = await post(`/api/providers/${name}/test`);
    setMsg({ ...msg, [name]: r.ok ? `Works. ${r.model} replied "${r.reply}".` : r.error });
  };
  const models = s.provider_models;
  return (
    <Page title="Settings" sub="Keys are stored on this computer only, in the Automatoner data folder with owner-only permissions.">
      <h2 style={{ marginTop: 0 }}>Hugging Face</h2>
      <div className="card stack">
        <p className="small muted" style={{ margin: 0 }}>A token lets you download gated models such as Gemma. Public models work without one. Create a read token at huggingface.co/settings/tokens.</p>
        <div className="row"><input type="password" style={{ maxWidth: 380 }} placeholder={s.hf_token_set ? "token saved (enter a new one to replace)" : "hf_..."} value={secrets.hf_token ?? ""} onChange={(e) => setSecrets({ ...secrets, hf_token: e.target.value })} />
          <button className="primary" disabled={!secrets.hf_token} onClick={() => saveSecret("hf_token")}>Save</button>
          <button onClick={async () => { const r = await post("/api/settings/hf-token/check"); setMsg({ ...msg, hf_token: r.detail }); }}>Check</button>
          {s.hf_token_set && <button className="danger" onClick={() => save({ hf_token: "" })}>Remove</button>}
        </div>
        {msg.hf_token && <div className="small">{msg.hf_token}</div>}
      </div>

      <h2>Privacy</h2>
      <div className="card stack">
        <label className="row" style={{ gap: 10 }}>
          <input type="checkbox" checked={!!s.local_only} disabled={s.local_only_locked} onChange={(e) => save({ local_only: e.target.checked })} />
          <span><b>Local-only mode.</b> Only Ollama on this computer or your private network may write synthetic data. Nothing is sent to Claude, Gemini, OpenAI or OpenRouter.</span>
        </label>
        {s.local_only_locked && <div className="small muted">An administrator fixed this setting on the server (AUTOMATONER_LOCAL_ONLY), so it cannot be changed here.</div>}
        {saveErr && <div className="small" style={{ color: "var(--bad, #c0392b)" }}>{saveErr}</div>}
        <div className="small muted">Model downloads from Hugging Face still work: they fetch files and send no data.</div>
      </div>

      <h2>Models that write synthetic data</h2>
      <div className="grid cols-2">
        <div className="card stack">
          <div className="row between"><h3>Ollama (local)</h3><Badge kind={ollama.length ? "ok" : "warn"}>{ollama.length ? `${ollama.length} models` : "not reachable"}</Badge></div>
          <label className="field"><span className="lbl">Server URL</span><input defaultValue={s.ollama_url} onBlur={(e) => save({ ollama_url: e.target.value })} /></label>
          <label className="field"><span className="lbl">Model</span>
            {ollama.length ? <select value={models.ollama} onChange={(e) => save({ provider_models: { ollama: e.target.value } })}>{ollama.map((m) => <option key={m}>{m}</option>)}</select>
              : <input defaultValue={models.ollama} onBlur={(e) => save({ provider_models: { ollama: e.target.value } })} />}</label>
          <div className="row"><button onClick={() => test("ollama")}>Test</button></div>
          {msg.ollama && <div className="small">{msg.ollama}</div>}
        </div>
        {KEYS.map(([k, label]) => (
          <div className="card stack" key={k}>
            <div className="row between"><h3>{label}</h3>{s[`${k}_api_key_set`] ? <Badge kind="ok">key saved</Badge> : <Badge>no key</Badge>}</div>
            <label className="field"><span className="lbl">API key</span>
              <input type="password" placeholder={s[`${k}_api_key_set`] ? "saved (enter a new one to replace)" : ""} value={secrets[`${k}_api_key`] ?? ""} onChange={(e) => setSecrets({ ...secrets, [`${k}_api_key`]: e.target.value })} /></label>
            <label className="field"><span className="lbl">Model</span><input defaultValue={models[k]} onBlur={(e) => save({ provider_models: { [k]: e.target.value } })} /></label>
            <div className="row"><button className="primary" disabled={!secrets[`${k}_api_key`]} onClick={() => saveSecret(`${k}_api_key`)}>Save</button>
              <button disabled={!s[`${k}_api_key_set`]} onClick={() => test(k)}>Test</button>
              {s[`${k}_api_key_set`] && <button className="danger" onClick={() => save({ [`${k}_api_key`]: "" })}>Remove</button>}</div>
            {msg[k] && <div className="small">{msg[k]}</div>}
          </div>
        ))}
      </div>
      <h2>Default provider</h2>
      <div className="card row">
        <select style={{ maxWidth: 240 }} value={s.default_provider} onChange={(e) => save({ default_provider: e.target.value })}>
          {["ollama", "anthropic", "gemini", "openai", "openrouter"].map((p) => <option key={p}>{p}</option>)}</select>
        <span className="small muted">Used when a form does not say which model should write the data.</span>
      </div>
      <h2>Audit log</h2>
      <div className="card stack">
        <p className="small muted" style={{ margin: 0 }}>The last 50 actions: who did what. Prompts, data and keys are never recorded. The full log is a file in the data folder; <code>automatoner audit</code> shows it in a terminal.</p>
        {audit.length === 0 ? <div className="small muted">Nothing yet.</div> : (
          <div style={{ maxHeight: 280, overflow: "auto" }}>
            <table className="small" style={{ width: "100%" }}>
              <tbody>{audit.map((e, i) => (
                <tr key={i}><td className="muted" style={{ whiteSpace: "nowrap" }}>{String(e.time).replace("T", " ").slice(0, 19)}</td><td>{e.actor}</td><td><b>{e.action}</b></td>
                  <td className="muted">{Object.entries(e).filter(([k]) => !["time", "actor", "action"].includes(k)).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" ")}</td></tr>))}</tbody>
            </table>
          </div>)}
        <div className="row"><button onClick={() => get("/api/audit?limit=50").then(setAudit)}>Refresh</button></div>
      </div>
      <Notice>Model names change often. If a test fails with "model not found", enter a current model name from the provider's documentation.</Notice>
    </Page>
  );
}
