# Author: Alex Picon <alexnpc@me.com>
"""Standalone HTTP application; inference remains in offline Docker workers."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from server.routers.hearsay_compare import router

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="HEARSAY", docs_url=None, redoc_url=None)
app.include_router(router)

@app.get("/healthz")
def health():
    return {"status": "ok", "service": "hearsay"}

@app.get("/hearsay-compare/")
def old_entrypoint():
    return RedirectResponse("/", status_code=307)

# Legacy deep links remain valid while the app also lives at the domain root.
app.mount("/hearsay-compare", StaticFiles(directory=ROOT / "apps/hearsay-compare", html=True), name="legacy")
app.mount("/", StaticFiles(directory=ROOT / "apps/hearsay-compare", html=True), name="web")
