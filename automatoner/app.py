from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, audit, auth, util
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


def _deny(scope, send_status: int, detail: str):
    async def go(send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            body = json.dumps({"detail": detail}).encode()
            await send({"type": "http.response.start", "status": send_status, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
    return go


def _cookie(headers: dict[str, str], name: str) -> str | None:
    for part in headers.get("cookie", "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == name:
            return v
    return None


class TokenAuth:
    """Require the access token for the API and WebSockets. The built UI files and the sign-in routes stay open."""

    OPEN = ("/api/auth/status", "/api/auth/login", "/api/auth/logout")

    def __init__(self, app, token: str | None) -> None:
        self.app, self.token = app, token
        self._last_denied: dict[str, float] = {}

    async def __call__(self, scope, receive, send):
        if self.token and scope["type"] in ("http", "websocket"):
            path = scope["path"]
            guarded = (path.startswith("/api/") or path.startswith("/ws/")) and path not in self.OPEN
            if guarded:
                headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
                if not auth.authorised(self.token, headers.get("authorization"), _cookie(headers, auth.COOKIE)):
                    ip = (scope.get("client") or ("?", 0))[0]
                    if time.time() - self._last_denied.get(ip, 0) > 10:  # one line per address every 10 s, so probing cannot flood the log
                        self._last_denied[ip] = time.time()
                        audit.record("auth.denied", client=ip, path=path)
                    await _deny(scope, 401, "authentication required")(send)
                    return
        await self.app(scope, receive, send)


class Actor:
    """Tag everything done while handling a request with who asked, for the audit log."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            tok = audit.set_actor(f"web:{(scope.get('client') or ('?', 0))[0]}")
            try:
                await self.app(scope, receive, send)
            finally:
                audit.reset_actor(tok)
            return
        await self.app(scope, receive, send)


def create_app(allow_any_host: bool = False, token: str | None = None) -> FastAPI:
    token = token or auth.configured_token()
    if allow_any_host:
        auth.require_for_remote(token)
    app = FastAPI(title="Automatoner", version=__version__)
    # Dev server (Vite) runs on another port; production serves the built UI from this app.
    # Added last = runs first: host and origin checks, then the token, then the actor tag.
    app.add_middleware(Actor)
    app.add_middleware(TokenAuth, token=token)
    app.add_middleware(CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(LocalOnly, allow_any_host=allow_any_host)
    fails: dict[str, list[float]] = {}

    @app.get("/api/auth/status", include_in_schema=False)
    def auth_status(request: Request):
        ok = not token or auth.authorised(token, request.headers.get("authorization"), request.cookies.get(auth.COOKIE))
        return {"required": bool(token), "authenticated": ok}

    @app.post("/api/auth/login", include_in_schema=False)
    async def auth_login(request: Request, body: dict):
        if not token:
            return {"required": False, "authenticated": True}
        ip = request.client.host if request.client else "?"
        recent = [t for t in fails.get(ip, []) if time.time() - t < 60]
        if len(recent) >= 10:
            return JSONResponse({"detail": "too many failed attempts, wait a minute"}, status_code=429)
        if not auth.same(str(body.get("token", "")).strip(), token):
            fails[ip] = recent + [time.time()]
            audit.record("auth.failed", client=ip)
            await asyncio.sleep(1)
            return JSONResponse({"detail": "wrong token"}, status_code=401)
        audit.record("auth.login", client=ip)
        r = JSONResponse({"required": True, "authenticated": True})
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        r.set_cookie(auth.COOKIE, auth.session_value(token), httponly=True, samesite="strict", secure=secure, max_age=12 * 3600, path="/")
        return r

    @app.post("/api/auth/logout", include_in_schema=False)
    def auth_logout():
        r = JSONResponse({"ok": True})
        r.delete_cookie(auth.COOKIE, path="/")
        return r

    @app.exception_handler(util.InvalidId)
    async def _invalid_id(_request, _exc):
        return JSONResponse({"detail": "unknown id"}, status_code=404)

    app.include_router(routes.router)
    app.include_router(routes.ws_router)
    runner.mark_interrupted()

    if (DIST / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            root = DIST.resolve()
            f = os.path.realpath(os.path.join(root, path))
            if path and f.startswith(str(root) + os.sep) and os.path.isfile(f):
                return FileResponse(f)
            return FileResponse(root / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def index():
            return HTMLResponse(PLACEHOLDER)
    return app
