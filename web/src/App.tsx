import { NavLink, Route, Routes } from "react-router-dom";
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

export default function App() {
  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <svg viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="var(--accent)" /><path d="M9 22V12l7-4 7 4v10l-7 3z" fill="none" stroke="var(--accent-ink)" strokeWidth="2" strokeLinejoin="round" /></svg>
          Automatoner
        </div>
        {nav.map(([to, label]) => <NavLink key={to} to={to} end={to === "/"}>{label}</NavLink>)}
        <div className="spacer" />
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
