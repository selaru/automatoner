import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { get, post } from "./api";
import Home from "./pages/Home";
import Models from "./pages/Models";
import Datasets from "./pages/Datasets";
import Train from "./pages/Train";
import JobPage from "./pages/JobPage";
import Doctor from "./pages/Doctor";
import Settings from "./pages/Settings";

const nav = [
  ["/", "Home"], ["/models", "Models"], ["/datasets", "Datasets"], ["/train", "Train & export"], ["/doctor", "Doctor"], ["/settings", "Settings"],
] as const;

function Login({ onDone }: { onDone: () => void }) {
  const [token, setToken] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setBusy(true); setErr("");
    try { await post("/api/auth/login", { token }); onDone(); } catch (e: any) { setErr(e.message || "Could not sign in"); }
    setBusy(false);
  };
  return (
    <div className="main" style={{ maxWidth: 460, margin: "12vh auto" }}>
      <div className="card stack">
        <h2 style={{ margin: 0 }}>Automatoner</h2>
        <p className="small muted" style={{ margin: 0 }}>This server needs an access token. Ask whoever runs it, or on the server run <code>automatoner token create</code> and read the token file.</p>
        <input type="password" autoFocus placeholder="Access token" value={token} onChange={(e) => setToken(e.target.value)} onKeyDown={(e) => e.key === "Enter" && token && submit()} />
        <div className="row"><button className="primary" disabled={!token || busy} onClick={submit}>Sign in</button></div>
        {err && <div className="small" style={{ color: "var(--bad, #c0392b)" }}>{err}</div>}
      </div>
    </div>
  );
}

export default function App() {
  const [auth, setAuth] = useState<{ required: boolean; authenticated: boolean } | null>(null);
  const check = () => get("/api/auth/status").then(setAuth).catch(() => setAuth({ required: false, authenticated: true }));
  useEffect(() => {
    check();
    const again = () => setAuth((a) => (a ? { ...a, authenticated: false } : a));
    window.addEventListener("auth-required", again);
    return () => window.removeEventListener("auth-required", again);
  }, []);
  if (!auth) return null;
  if (auth.required && !auth.authenticated) return <Login onDone={check} />;
  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <svg viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="var(--accent)" /><path d="M9 22V12l7-4 7 4v10l-7 3z" fill="none" stroke="var(--accent-ink)" strokeWidth="2" strokeLinejoin="round" /></svg>
          Automatoner
        </div>
        {nav.map(([to, label]) => <NavLink key={to} to={to} end={to === "/"}>{label}</NavLink>)}
        <div className="spacer" />
        {auth.required && <button className="small" onClick={async () => { await post("/api/auth/logout"); check(); }}>Sign out</button>}
        <div className="foot">Tiny models for one job each.</div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/models" element={<Models />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/train" element={<Train />} />
          <Route path="/jobs/:id" element={<JobPage />} />
          <Route path="/doctor" element={<Doctor />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  );
}
