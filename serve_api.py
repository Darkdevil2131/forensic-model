from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Header,
                     Request, UploadFile, status)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, HTMLResponse

# --------------------------------------------------------------------------- config

MODEL_PATH = os.environ.get("FORENSIC_MODEL", "denoiser_inference.pth")
JOB_ROOT = os.environ.get("FORENSIC_JOB_DIR", os.path.join(tempfile.gettempdir(),
                                                           "forensic_jobs"))
MAX_UPLOAD_MB = float(os.environ.get("FORENSIC_MAX_UPLOAD_MB", "200"))
MAX_FRAMES = int(os.environ.get("FORENSIC_MAX_FRAMES", "200"))
JOB_TTL_SECONDS = int(os.environ.get("FORENSIC_JOB_TTL", str(6 * 3600)))

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts"}


def _load_api_keys() -> set:
    raw = os.environ.get("FORENSIC_API_KEYS", "").strip()
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    if not keys:
        k = "dev_" + secrets.token_urlsafe(24)
        keys = {k}
        print("\n" + "!" * 78, flush=True)
        print("FORENSIC_API_KEYS is not set. Generated a throwaway key for this process:",
              flush=True)
        print(f"    {k}", flush=True)
        print("It changes every restart. Set FORENSIC_API_KEYS in the environment before",
              flush=True)
        print("you point a website at this.", flush=True)
        print("!" * 78 + "\n", flush=True)
    return keys


API_KEYS = _load_api_keys()


def require_key(x_api_key: Optional[str] = Header(default=None),
                authorization: Optional[str] = Header(default=None)) -> str:
    supplied = x_api_key
    if not supplied and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            "Missing API key. Send X-API-Key or Authorization: Bearer.")
    # constant-time compare against every key, so a wrong key cannot be timed out of us
    ok = False
    for k in API_KEYS:
        if secrets.compare_digest(supplied, k):
            ok = True
    if not ok:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid API key.")
    return supplied


# --------------------------------------------------------------------------- model

class Engine:
    """Loads the restorer once and serves it. Thread-safe: one lock around inference,
    because a single GPU processing two requests concurrently is slower than queueing."""

    def __init__(self):
        self.net = None
        self.meta = {}
        self.device = "cpu"
        self.checkpoint = None
        self.sha256 = None
        self.error = None
        self.lock = threading.Lock()

    def load(self, path):
        self.checkpoint = os.path.abspath(path)
        try:
            import torch
            # the model definition normally sits next to the checkpoint
            for d in (os.path.dirname(self.checkpoint), os.getcwd(),
                      os.path.dirname(os.path.abspath(__file__))):
                if d and os.path.exists(os.path.join(d, "train_denoiser_model.py")):
                    if d not in sys.path:
                        sys.path.insert(0, d)
                    break
            from train_denoiser_model import load_denoiser

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.net = load_denoiser(self.checkpoint, device=self.device, prefer_ema=True)
            self.meta = dict(getattr(self.net, "meta", {}) or {})
            h = hashlib.sha256()
            with open(self.checkpoint, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            self.sha256 = h.hexdigest()
            self.error = None
            print(f"[engine] loaded {self.checkpoint}\n"
                  f"         arch={self.meta.get('arch')} "
                  f"params={self.meta.get('params')} step={self.meta.get('step')} "
                  f"val_psnr={self.meta.get('best_psnr')} device={self.device}", flush=True)
        except Exception as e:
            self.net = None
            self.error = f"{type(e).__name__}: {e}"
            print(f"[engine] MODEL NOT LOADED: {self.error}", flush=True)
            print("         /v1/restore will return 503. Fusion still works without it.",
                  flush=True)

    @property
    def ready(self):
        return self.net is not None

    def restore(self, bgr, tile=0):
        if not self.ready:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                f"Restoration model not loaded: {self.error}")
        from train_denoiser_model import restore_image
        with self.lock:
            return restore_image(self.net, bgr, device=self.device,
                                 amp=(self.device == "cuda"), tile=tile)

    def info(self):
        return {
            "loaded": self.ready,
            "error": self.error,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.sha256,
            "device": self.device,
            **{k: v for k, v in self.meta.items() if not k.endswith("state_dict")},
        }


def _json_safe(o):
    """JSON has no NaN/Infinity and Starlette refuses to emit them. Checkpoint metadata
    and fusion reports can both contain them, so scrub on the way out."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


ENGINE = Engine()

# --------------------------------------------------------------------------- jobs

JOBS = {}
JOBS_LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fusion")


def _job_dir(job_id):
    return os.path.join(JOB_ROOT, job_id)


def _safe_name(name, fallback="upload"):
    base = os.path.basename(name or "")
    base = "".join(c for c in base if c.isalnum() or c in "._- ")[:80].strip()
    return base or fallback


def _set(job_id, **kw):
    with JOBS_LOCK:
        JOBS[job_id].update(kw)


def _run_fusion(job_id, input_path, params):
    _set(job_id, status="running", started_utc=datetime.now(timezone.utc).isoformat())
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from face_fusion_v2 import reconstruct

        outdir = os.path.join(_job_dir(job_id), "out")
        t0 = time.time()
        report = reconstruct(
            input_path, outdir,
            max_images=params["max_images"],
            denoiser_checkpoint=(ENGINE.checkpoint if params["use_denoiser"]
                                 and ENGINE.ready else None),
            pose_tol=params["pose_tol"],
            weights=None,
            min_region_score=params["min_region_score"],
        )
        files = sorted(os.listdir(outdir)) if os.path.isdir(outdir) else []
        _set(job_id, status="done", report=report, files=files,
             seconds=round(time.time() - t0, 2),
             finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as e:
        _set(job_id, status="error", error=f"{type(e).__name__}: {e}",
             traceback=traceback.format_exc()[-4000:],
             finished_utc=datetime.now(timezone.utc).isoformat())


def _reap_old_jobs():
    now = time.time()
    with JOBS_LOCK:
        stale = [j for j, v in JOBS.items()
                 if now - v.get("created_ts", now) > JOB_TTL_SECONDS]
        for j in stale:
            JOBS.pop(j, None)
    for j in stale:
        shutil.rmtree(_job_dir(j), ignore_errors=True)


# --------------------------------------------------------------------------- app

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app_):
    os.makedirs(JOB_ROOT, exist_ok=True)
    ENGINE.load(MODEL_PATH)
    yield
    POOL.shutdown(wait=False)


app = FastAPI(title="Forensic face restoration API", version="4.0",
              lifespan=_lifespan,
              description=(__doc__ or "").split("Endpoints")[0].strip())

_origins = [o.strip() for o in os.environ.get("FORENSIC_CORS_ORIGINS", "*").split(",")
            if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                   allow_headers=["X-API-Key", "Authorization", "Content-Type"])


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Forensic AI API</title>
    </head>
    <body style="font-family:Arial;padding:40px;text-align:center;">
        <h1>🧠 Forensic AI Restoration API</h1>
        <p><b>Status:</b> Running ✅</p>

        <h3>Endpoints</h3>

        <p><a href="/health">/health</a></p>
        <p>/v1/model (requires API key)</p>
        <p>/v1/restore (POST)</p>
        <p>/v1/reconstruct (POST)</p>

        <hr>
        <p>Powered by your trained NAFNet restoration model.</p>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ENGINE.ready, "device": ENGINE.device,
            "arch": ENGINE.meta.get("arch"), "jobs": len(JOBS),
            "utc": datetime.now(timezone.utc).isoformat()}


@app.get("/v1/model")
def model_info(_: str = Depends(require_key)):
    return _json_safe(ENGINE.info())


async def _read_upload(f: UploadFile, limit_mb=MAX_UPLOAD_MB) -> bytes:
    data = await f.read()
    if len(data) > limit_mb * 1e6:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"{f.filename} is over {limit_mb:.0f} MB.")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{f.filename} is empty.")
    return data


@app.post("/v1/restore")
async def restore(file: UploadFile = File(...), tile: int = Form(0),
                  _: str = Depends(require_key)):
    """One image in, the restored image out as PNG. Any resolution; pass tile=512 for
    frames too large to fit in VRAM in one pass."""
    data = await _read_upload(file)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Could not decode that file as an image.")
    t0 = time.time()
    out = ENGINE.restore(img, tile=int(tile))
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise HTTPException(500, "PNG encode failed.")
    return Response(
        content=buf.tobytes(), media_type="image/png",
        headers={
            "X-Input-Size": f"{img.shape[1]}x{img.shape[0]}",
            "X-Elapsed-Seconds": f"{time.time()-t0:.3f}",
            "X-Model-Arch": str(ENGINE.meta.get("arch")),
            "X-Model-Sha256": (ENGINE.sha256 or "")[:16],
            "Content-Disposition": 'inline; filename="restored.png"',
        })


@app.post("/v1/reconstruct", status_code=status.HTTP_202_ACCEPTED)
async def reconstruct_endpoint(
        files: list[UploadFile] = File(...),
        max_images: int = Form(MAX_FRAMES),
        pose_tol: float = Form(20.0),
        min_region_score: float = Form(0.25),
        use_denoiser: bool = Form(True),
        _: str = Depends(require_key)):
    """Multi-frame forensic reconstruction. Send either one video file or many stills.
    Returns a job id straight away; poll GET /v1/jobs/{id}."""
    _reap_old_jobs()
    if not files:
        raise HTTPException(400, "No files.")

    job_id = uuid.uuid4().hex[:16]
    jd = _job_dir(job_id)
    updir = os.path.join(jd, "input")
    os.makedirs(updir, exist_ok=True)

    saved, kinds = [], set()
    for i, f in enumerate(files[:MAX_FRAMES]):
        data = await _read_upload(f)
        name = _safe_name(f.filename, f"frame_{i:04d}")
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXT | VIDEO_EXT:
            shutil.rmtree(jd, ignore_errors=True)
            raise HTTPException(415, f"Unsupported file type: {name or f.filename}")
        kinds.add("video" if ext in VIDEO_EXT else "image")
        p = os.path.join(updir, f"{i:04d}_{name}")
        with open(p, "wb") as fh:
            fh.write(data)
        saved.append(p)

    if "video" in kinds and len(saved) > 1:
        shutil.rmtree(jd, ignore_errors=True)
        raise HTTPException(400, "Send one video, or several stills - not both.")

    # face_fusion_v2 takes a video path or a directory of images
    input_path = saved[0] if "video" in kinds else updir

    with JOBS_LOCK:
        JOBS[job_id] = {"id": job_id, "status": "queued", "files_received": len(saved),
                        "input_kind": ("video" if "video" in kinds else "images"),
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "created_ts": time.time()}
    params = {"max_images": int(max_images), "pose_tol": float(pose_tol),
              "min_region_score": float(min_region_score),
              "use_denoiser": bool(use_denoiser)}
    POOL.submit(_run_fusion, job_id, input_path, params)

    return {"job_id": job_id, "status": "queued",
            "poll": f"/v1/jobs/{job_id}", "files_received": len(saved)}


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str, _: str = Depends(require_key)):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such job (or it expired).")
    out = dict(job)
    out.pop("created_ts", None)
    if job.get("status") == "done":
        out["artifacts"] = {n: f"/v1/jobs/{job_id}/files/{n}" for n in job.get("files", [])}
    return _json_safe(out)


@app.get("/v1/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str, _: str = Depends(require_key)):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such job.")
    safe = _safe_name(name)
    path = os.path.join(_job_dir(job_id), "out", safe)
    # belt and braces: the resolved path must still be inside the job directory
    if not os.path.realpath(path).startswith(os.path.realpath(_job_dir(job_id))):
        raise HTTPException(400, "Bad filename.")
    if not os.path.isfile(path):
        raise HTTPException(404, f"No artifact named {safe}.")
    media = "image/png" if safe.endswith(".png") else (
        "application/json" if safe.endswith(".json") else "application/octet-stream")
    return FileResponse(path, media_type=media, filename=safe)


@app.delete("/v1/jobs/{job_id}")
def job_delete(job_id: str, _: str = Depends(require_key)):
    with JOBS_LOCK:
        existed = JOBS.pop(job_id, None) is not None
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    if not existed:
        raise HTTPException(404, "No such job.")
    return {"deleted": job_id}


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    return JSONResponse(status_code=500,
                        content={"detail": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Railway (and Render, Fly, most PaaS hosts) injects a PORT env var and only
    # routes traffic to that exact port on 0.0.0.0 -- binding to 127.0.0.1 or a
    # hardcoded port is the single most common reason a deploy shows "Application
    # failed to respond" even though the build succeeded. These defaults make
    # `python serve_api.py` with NO flags correct both locally and on Railway.
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    os.environ["FORENSIC_MODEL"] = os.path.abspath(a.model)
    MODEL_PATH = os.path.abspath(a.model)
    import uvicorn
    print(f"[serve_api] binding {a.host}:{a.port} "
          f"(PORT env var: {os.environ.get('PORT', '<not set, using default>')})")
    uvicorn.run("serve_api:app" if a.reload else app, host=a.host, port=a.port,
                reload=a.reload, log_level="info")