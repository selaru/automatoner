export class ApiError extends Error {}

async function req<T>(method: string, url: string, body?: unknown, form?: FormData): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body !== undefined ? { "content-type": "application/json" } : undefined,
    body: form ?? (body !== undefined ? JSON.stringify(body) : undefined),
  });
  if (r.status === 401 && !url.startsWith("/api/auth/")) window.dispatchEvent(new Event("auth-required"));
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j); } catch { /* keep status text */ }
    throw new ApiError(msg);
  }
  return r.json() as Promise<T>;
}

export const get = <T = any>(url: string) => req<T>("GET", url);
export const post = <T = any>(url: string, body: unknown = {}) => req<T>("POST", url, body);
export const put = <T = any>(url: string, body: unknown) => req<T>("PUT", url, body);
export const del = <T = any>(url: string) => req<T>("DELETE", url);
export const upload = <T = any>(url: string, form: FormData) => req<T>("POST", url, undefined, form);

export function wsUrl(path: string) {
  const p = location.protocol === "https:" ? "wss:" : "ws:";
  return `${p}//${location.host}${path}`;
}

export interface Model {
  id: string; name: string; source: "hf" | "builtin"; repo?: string; categories: string[]; params: string; size_mb: number;
  license: string; gated?: boolean; notes?: string; download_note?: string; good_for: string; runs_on: string[];
  trainable: Record<string, boolean>; trainer_options: Record<string, string[]>;
  state: { status: string; progress?: number; message?: string; error?: string; bytes?: number };
}
export interface Category { id: string; name: string; blurb: string; dataset: string; models: number; trainable: number }
export interface Dataset {
  id: string; name: string; category: string; source: string; status: string; rows?: number; splits?: Record<string, number>;
  columns?: string[]; label_column?: string | null; report?: { errors: string[]; warnings: string[]; dropped: number }; created: string; error?: string;
  format?: string; labels?: string[];
}
export interface Job {
  id: string; status: string; model_id: string; model_name: string; dataset_id: string; dataset_name: string; trainer: string; category: string;
  config: Record<string, any>; created: string; started?: string; finished?: string; error?: string | null; summary?: any;
  series?: Record<string, any>[]; manifest?: any;
}
export interface TaskInfo { id: string; kind: string; title: string; status: string; progress: number; message: string; result: any; error?: string; log: string[] }
export interface FieldDef { key: string; label: string; type: "int" | "float" | "bool" | "select" | "text"; default: any; help?: string; min?: number; max?: number; options?: any[]; advanced?: boolean }
