from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import routes
from .training import runner

DIST = Path(__file__).resolve().parent / "web" / "dist"

PLACEHOLDER = """<!doctype html><meta charset=utf-8><title>Automatoner</title>
<body style="font-family:system-ui;max-width:40rem;margin:3rem auto">
<h1>Automatoner is running</h1>
<p>The web interface has not been built. Run <code>npm install && npm run build</code> in <code>web/</code>,
or use <code>npm run dev</code> and open the Vite address. The API is available at <a href="/docs">/docs</a>.</p>"""


LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}
DEV_ORIGINS = ["http://127.0.0.1:5173", "http://localhost:5173"]


def _host_only(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("["):  # [::1]:8765
        return value.split("]")[0] + "]"
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


class LocalOnly:
    """Refuse requests a web page could forge against this local server.

    - DNS rebinding: the Host header must be a loopback name (unless the server was started on a non-loopback address on purpose).
    - Cross-site requests and WebSockets: when the browser sends an Origin, it must be this server (or the Vite dev server).
    """

    def __init__(self, app, allow_any_host: bool) -> None:
        self.app, self.allow_any_host = app, allow_any_host

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
            host = headers.get("host", "")
            bad = None
            if not self.allow_any_host and _host_only(host) not in LOOPBACK:
                bad = "unexpected Host header"
            origin = headers.get("origin")
            unsafe = scope["type"] == "websocket" or scope.get("method") not in ("GET", "HEAD", "OPTIONS")
            if bad is None and origin and unsafe:
                o = origin.split("://", 1)[-1]
                if o != host and origin not in DEV_ORIGINS:
                    bad = "cross-site request refused"
            if bad:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    body = json.dumps({"detail": bad}).encode()
                    await send({"type": "http.response.start", "status": 403, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
                    await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def create_app(allow_any_host: bool = False) -> FastAPI:
    app = FastAPI(title="Automatoner", version=__version__)
    # Dev server (Vite) runs on another port; production serves the built UI from this app.
    app.add_middleware(CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(LocalOnly, allow_any_host=allow_any_host)
    app.include_router(routes.router)
    app.include_router(routes.ws_router)
    runner.mark_interrupted()

    if (DIST / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            f = (DIST / path).resolve()
            if path and DIST.resolve() in f.parents and f.is_file():
                return FileResponse(f)
            return FileResponse(DIST / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def index():
            return HTMLResponse(PLACEHOLDER)
    return app
