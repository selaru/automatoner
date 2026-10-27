import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` proxies /api and /ws to the Python server on 8765.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../automatoner/web/dist", emptyOutDir: true },
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8765", "/ws": { target: "ws://127.0.0.1:8765", ws: true } } },
});
