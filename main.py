import asyncio
import os
import uuid
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from pipeline import stitch_images

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="360 Stitch Prototype")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("STITCH_WORKERS", "2")))
_session_stage: dict[str, str] = {}
_session_errors: dict[str, str] = {}


@app.get("/")
def root():
    return {"message": "360 stitch server running. Open /static/viewer.html"}


# ── Per-shot upload ──────────────────────────────────────────────
@app.post("/upload/{session_id}")
async def upload_shot(session_id: str, file: UploadFile = File(...)):
    """Receive one image at a time — called immediately after each capture."""
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    existing = len([f for f in session_dir.iterdir() if f.suffix in ('.jpg', '.jpeg', '.png')])
    ext = Path(file.filename or "img.jpg").suffix or ".jpg"
    filename = Path(file.filename).name if file.filename else f"{existing:03d}{ext}"
    dest = session_dir / filename
    dest.write_bytes(await file.read())
    return {"saved": dest.name, "index": existing, "total": existing + 1}


# ── Stitch by session ID (no file upload needed) ─────────────────
@app.post("/stitch/{session_id}")
async def stitch_session(session_id: str, fov: float = Query(default=75.0)):
    """Stitch all images already uploaded for this session."""
    session_dir = (UPLOAD_DIR / session_id).resolve()
    if not session_dir.exists():
        raise HTTPException(404, "Session not found — no images uploaded yet")

    def spatial_key(p: Path):
        # Sort by elevation asc (horizon first), then azimuth asc (left-to-right sweep)
        # Filename format: el50_az060.jpg or el n45_az030.jpg (n = negative)
        name = p.stem  # e.g. "el50_az060" or "eln45_az030"
        try:
            el_part, az_part = name.split('_az')
            el = int(el_part.replace('el','').replace('n','-'))
            az = int(az_part)
            return (abs(el), az)  # horizon (el=0) first, then by azimuth
        except Exception:
            return (999, name)  # fallback: sort legacy filenames alphabetically

    paths = [
        str(p) for p in sorted(
            (p for p in session_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')),
            key=spatial_key
        )
    ]
    if len(paths) < 2:
        raise HTTPException(400, f"Only {len(paths)} image(s) in session, need at least 2")

    output_path = (OUTPUT_DIR / f"{session_id}.jpg").resolve()
    _session_stage[session_id] = "starting"

    def _run():
        try:
            success, res = stitch_images(session_id, session_dir, output_path, fov=fov)
            _session_stage[session_id] = "done" if success else "error"
            if not success:
                _session_errors[session_id] = res
        except Exception as e:
            _session_stage[session_id] = "error"
            _session_errors[session_id] = str(e)

    _executor.submit(_run)
    return {"status": "queued", "session": session_id}


# ── Legacy bulk stitch (used by viewer.html) ─────────────────────
@app.post("/stitch")
async def stitch_bulk(
    files: list[UploadFile] = File(...),
    session_id: Optional[str] = Query(default=None),
):
    if len(files) < 2:
        raise HTTPException(400, "Send at least 2 images")

    sid = session_id or uuid.uuid4().hex[:10]
    session_dir = UPLOAD_DIR / sid
    session_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, f in enumerate(files):
        ext = Path(f.filename or "img.jpg").suffix or ".jpg"
        dest = session_dir / f"{i:03d}{ext}"
        dest.write_bytes(await f.read())
        paths.append(str(dest))

    output_path = str(OUTPUT_DIR / f"{sid}.jpg")
    _session_stage[sid] = "starting"

    def _run():
        success, res = stitch_images(sid, session_dir, Path(output_path))
        if not success:
            raise Exception(res)
        return {}

    loop = asyncio.get_event_loop()
    try:
        stats = await loop.run_in_executor(_executor, _run)
        _session_stage[sid] = "done"
    except Exception as e:
        _session_stage[sid] = "error"
        return JSONResponse({"status": "error", "message": str(e)}, status_code=422)

    return {"status": "ok", "url": f"/result/{sid}.jpg", "session": sid, **stats}


@app.get("/status/{session_id}")
def session_status(session_id: str):
    stage = _session_stage.get(session_id, "unknown")
    resp = {"stage": stage}
    if stage == "done":
        resp["url"] = f"/result/{session_id}.jpg"
    if stage == "error":
        resp["message"] = _session_errors.get(session_id, "unknown error")
    return resp


@app.get("/debug/{session_id}")
def debug_session(session_id: str):
    session_dir = UPLOAD_DIR / session_id
    if not session_dir.exists():
        return {"error": "session not found"}
    files = sorted(session_dir.iterdir())
    return {
        "session_id": session_id,
        "image_count": len(files),
        "stage": _session_stage.get(session_id, "unknown"),
        "images": [{"name": f.name, "size_kb": round(f.stat().st_size / 1024)} for f in files],
    }


@app.get("/result/{filename}")
def get_result(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Result not found")
    return FileResponse(str(path), media_type="image/jpeg")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "localhost"

    print("\n" + "=" * 52)
    print("  360° Stitch Prototype Server")
    print("=" * 52)
    print(f"  Desktop viewer : http://localhost:{port}/static/viewer.html")
    print(f"  Mobile capture : http://{local_ip}:{port}/static/capture.html")
    print("=" * 52 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
