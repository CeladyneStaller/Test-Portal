"""
Analysis Portal (Lite) — FastAPI application.

Self-contained: no database, no Redis, no cloud storage.
Jobs run in a ProcessPoolExecutor and results live in /tmp until auto-cleaned.

Routes:
  GET  /                              → Frontend UI
  GET  /api/scripts                   → Available analysis scripts
  POST /api/upload                    → Upload CSVs + start analysis
  GET  /api/jobs/{id}                 → Job status
  GET  /api/jobs                      → All jobs (for page reload recovery)
  GET  /api/download/{id}/{filename}  → Download a result file
"""

import os
import uuid
import shutil
import time
import traceback
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
JOB_TTL_HOURS = int(os.getenv("JOB_TTL_HOURS", "24"))
JOBS_DIR = Path(os.getenv("JOBS_DIR", "/tmp/analysis-portal-jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# In-memory job store
# ═══════════════════════════════════════════════════════════════════

jobs: dict = {}  # job_id → job metadata dict
jobs_lock = threading.Lock()

app = FastAPI(title="Analysis Portal")
executor = ProcessPoolExecutor(max_workers=MAX_WORKERS, max_tasks_per_child=1)

TEMPLATE_DIR = Path(__file__).parent / "templates"


# ═══════════════════════════════════════════════════════════════════
# Worker function (runs in a separate process)
# ═══════════════════════════════════════════════════════════════════

def _run_job(job_id: str, script_name: str, input_dir: str, output_dir: str,
             params: dict = None) -> dict:
    """
    Executed in the process pool. Imports the script, runs it,
    renames output files with sample name prefix, and returns result.
    """
    import importlib
    import scripts

    registry = scripts.SCRIPT_REGISTRY

    if script_name not in registry:
        raise ValueError(f"Unknown script: {script_name}")

    run_fn = registry[script_name]
    result = run_fn(input_dir=input_dir, output_dir=output_dir, params=params or {})

    # ── Resolve sample name ──
    p = params or {}
    sample_name = p.get('sample_name', '').strip()
    if not sample_name:
        # Derive from input: use first directory name or first filename
        inp = Path(input_dir)
        entries = sorted(inp.iterdir())
        if entries:
            first = entries[0]
            if first.is_dir():
                sample_name = first.name[:18]
            else:
                sample_name = first.stem[:18]

    # Script short label
    short = scripts.SCRIPT_SHORT.get(script_name, script_name.replace(' ', ''))

    # ── Rename output files with prefix ──
    out = Path(output_dir)
    prefix = f"{sample_name}-{short}" if sample_name else short

    for f in sorted(out.rglob("*"), reverse=True):  # reverse to handle files before dirs
        if not f.is_file():
            continue
        new_name = f"{prefix}-{f.name}"
        new_path = f.parent / new_name
        f.rename(new_path)

    # ── Collect and group output files ──
    all_output = [f for f in out.rglob("*") if f.is_file()]

    DIR_LABELS = {
        'ecsa': 'ECSA',
        'eis': 'EIS',
        'crossover': 'H₂ Crossover',
        'polcurve': 'Polarization Curve',
        'ocv': 'OCV',
        'durability': 'Durability',
    }

    grouped = {}
    flat_list = []
    for f in sorted(all_output):
        rel = f.relative_to(out)
        flat_list.append(str(rel))
        parts = rel.parts
        if len(parts) > 1:
            dir_key = parts[0]
            label = DIR_LABELS.get(dir_key, dir_key.replace('_', ' ').title())
        else:
            label = script_name
        grouped.setdefault(label, []).append(str(rel))

    # Clean up matplotlib and free memory before worker exits
    try:
        import matplotlib.pyplot as _plt
        _plt.close('all')
    except Exception:
        pass
    import gc
    gc.collect()

    return {
        "output_files": flat_list,
        "output_groups": grouped,
        "script_result": result or {},
        "sample_name": sample_name,
    }


def _on_job_done(job_id: str, future):
    """Callback when a job finishes (success or failure)."""
    with jobs_lock:
        if job_id not in jobs:
            return
        try:
            result = future.result()
            jobs[job_id].update({
                "status": "complete",
                "message": "Analysis complete",
                "output_files": result["output_files"],
                "output_groups": result.get("output_groups", {}),
                "script_result": result["script_result"],
                "sample_name": result.get("sample_name", ""),
                "completed_at": datetime.now().isoformat(),
            })
        except Exception as e:
            jobs[job_id].update({
                "status": "failed",
                "message": str(e),
                "error": traceback.format_exc(),
                "completed_at": datetime.now().isoformat(),
            })


# ═══════════════════════════════════════════════════════════════════
# Cleanup: remove expired jobs
# ═══════════════════════════════════════════════════════════════════

def _cleanup_old_jobs():
    """Remove jobs and their files after JOB_TTL_HOURS."""
    ttl_seconds = JOB_TTL_HOURS * 3600
    now = time.time()

    with jobs_lock:
        expired = []
        for jid, meta in jobs.items():
            job_dir = JOBS_DIR / jid
            if job_dir.exists():
                age = now - job_dir.stat().st_mtime
                if age > ttl_seconds and meta["status"] in ("complete", "failed"):
                    expired.append(jid)

        for jid in expired:
            shutil.rmtree(JOBS_DIR / jid, ignore_errors=True)
            del jobs[jid]


def _start_cleanup_timer():
    _cleanup_old_jobs()
    timer = threading.Timer(3600, _start_cleanup_timer)  # Run every hour
    timer.daemon = True
    timer.start()


@app.on_event("startup")
def startup():
    _start_cleanup_timer()


# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((TEMPLATE_DIR / "index.html").read_text())


@app.get("/api/scripts")
async def list_scripts():
    from scripts import SCRIPT_REGISTRY, SCRIPT_PARAMS
    return {
        "scripts": [
            {
                "name": name,
                "description": (fn.__doc__ or "").strip().split("\n")[0],
                "params": SCRIPT_PARAMS.get(name, []),
            }
            for name, fn in SCRIPT_REGISTRY.items()
        ]
    }


@app.post("/api/upload")
async def upload_and_run(
    script: str = Form(...),
    params: str = Form("{}"),
    files: list[UploadFile] = File(None),
    zipfile: UploadFile = File(None),
):
    import json
    import zipfile as zipmod
    from scripts import SCRIPT_REGISTRY
    if script not in SCRIPT_REGISTRY:
        raise HTTPException(400, f"Unknown script: {script}")
    if not files and not zipfile:
        raise HTTPException(400, "No files uploaded")

    # Parse params JSON
    try:
        user_params = json.loads(params)
    except json.JSONDecodeError:
        user_params = {}

    # Create job directories
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    input_dir = JOBS_DIR / job_id / "input"
    output_dir = JOBS_DIR / job_id / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    filenames = []

    if zipfile and zipfile.filename:
        # ── Zip upload: extract preserving folder structure ──
        zip_bytes = await zipfile.read()
        import io
        with zipmod.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                safe_path = Path(info.filename)
                if '..' in safe_path.parts:
                    continue
                dest = input_dir / safe_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(info))
                filenames.append(info.filename)
    elif files:
        # ── Individual file upload ──
        for f in files:
            content = await f.read()
            if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
                shutil.rmtree(JOBS_DIR / job_id)
                raise HTTPException(413, f"{f.filename} exceeds {MAX_UPLOAD_MB}MB limit")
            safe_path = Path(f.filename)
            if '..' in safe_path.parts:
                continue
            dest = input_dir / safe_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            filenames.append(f.filename)

    if not filenames:
        shutil.rmtree(JOBS_DIR / job_id)
        raise HTTPException(400, "No files received")

    # Register job
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "message": f"Running {script}...",
            "script": script,
            "input_files": filenames,
            "submitted_at": datetime.now().isoformat(),
        }

    # Submit to process pool
    future = executor.submit(
        _run_job, job_id, script, str(input_dir), str(output_dir), user_params
    )
    future.add_done_callback(lambda f: _on_job_done(job_id, f))

    return {"job_id": job_id, "status": "running", "files_received": filenames}


@app.get("/api/jobs")
async def all_jobs():
    with jobs_lock:
        return {"jobs": list(jobs.values())}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job not found")
        return jobs[job_id]


@app.get("/api/download/{job_id}/{filepath:path}")
async def download_result(job_id: str, filepath: str):
    # Resolve and validate path to prevent traversal
    base = JOBS_DIR / job_id / "output"
    file_path = (base / filepath).resolve()

    # Ensure the resolved path is still inside the output directory
    if not str(file_path).startswith(str(base.resolve())):
        raise HTTPException(400, "Invalid path")

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    media_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "pdf": "application/pdf",
    }
    ext = file_path.suffix.lstrip(".").lower()
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=file_path.name,
    )


@app.get("/api/download-zip/{job_id}")
async def download_zip(job_id: str, group: str = Query(None)):
    """Download all results (or a group) as a ZIP file."""
    import zipfile
    import io

    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job not found")
        job = jobs[job_id]

    base = JOBS_DIR / job_id / "output"
    if not base.exists():
        raise HTTPException(404, "Output directory not found")

    # Get the file list for the requested group
    groups = job.get("output_groups", {})
    sample = job.get("sample_name", "")
    if group and group in groups:
        file_list = groups[group]
        prefix = f"{sample}-" if sample else ""
        zip_name = f"{prefix}{group.lower().replace(' ', '_').replace('₂', '2')}_results.zip"
    else:
        file_list = job.get("output_files", [])
        zip_name = f"{sample}_results.zip" if sample else f"results_{job_id}.zip"

    if not file_list:
        raise HTTPException(404, "No files to download")

    # Build ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for rel_path in file_list:
            file_path = (base / rel_path).resolve()
            if not str(file_path).startswith(str(base.resolve())):
                continue
            if file_path.exists():
                zf.write(file_path, arcname=rel_path)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


# ═══════════════════════════════════════════════════════════════════
# Multi-sample upload
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/upload-multi")
async def upload_multi(
    script: str = Form(...),
    manifest: str = Form(...),
    files: list[UploadFile] = File(None),
    zipfile: UploadFile = File(None),
):
    """
    Upload all files from a parent folder and split into per-sample jobs.

    manifest is a JSON array:
    [{"folder": "Parent/BOL", "sample_name": "BOL", "params": {...}}, ...]
    """
    import json
    import zipfile as zipmod
    from scripts import SCRIPT_REGISTRY

    if script not in SCRIPT_REGISTRY:
        raise HTTPException(400, f"Unknown script: {script}")

    try:
        samples = json.loads(manifest)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid manifest JSON")

    if not samples:
        raise HTTPException(400, "No samples defined")

    # Read all file contents into memory, grouped by sample folder
    file_data = {}  # folder_prefix → [(relative_path, bytes)]

    if zipfile and zipfile.filename:
        # ── Zip upload: extract and group ──
        import io
        zip_bytes = await zipfile.read()
        with zipmod.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                fname = info.filename
                content = zf.read(info)
                for sample in samples:
                    folder = sample["folder"]
                    if fname == folder or fname.startswith(folder + "/"):
                        file_data.setdefault(folder, []).append((fname, content))
                        break
    elif files:
        # ── Individual file upload ──
        for f in files:
            content = await f.read()
            if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"{f.filename} exceeds {MAX_UPLOAD_MB}MB limit")
            for sample in samples:
                folder = sample["folder"]
                if f.filename == folder or f.filename.startswith(folder + "/"):
                    file_data.setdefault(folder, []).append((f.filename, content))
                    break
    else:
        raise HTTPException(400, "No files uploaded")

    # Create one job per sample
    job_results = []
    for sample in samples:
        folder = sample["folder"]
        sample_name = sample.get("sample_name", folder.split("/")[-1][:18])
        params = sample.get("params", {})
        params["sample_name"] = sample_name

        job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        input_dir = JOBS_DIR / job_id / "input"
        output_dir = JOBS_DIR / job_id / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        # Write files for this sample
        sample_files = file_data.get(folder, [])
        filenames = []
        for filepath, content in sample_files:
            safe_path = Path(filepath)
            if '..' in safe_path.parts:
                continue
            dest = input_dir / safe_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            filenames.append(filepath)

        if not filenames:
            continue  # skip empty samples

        # Register job
        with jobs_lock:
            jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "message": f"Running {script} on {sample_name}...",
                "script": script,
                "input_files": filenames,
                "submitted_at": datetime.now().isoformat(),
            }

        # Submit to process pool
        future = executor.submit(
            _run_job, job_id, script, str(input_dir), str(output_dir), params
        )
        future.add_done_callback(lambda f, jid=job_id: _on_job_done(jid, f))

        job_results.append({"job_id": job_id, "sample_name": sample_name,
                            "files": len(filenames)})

    if not job_results:
        raise HTTPException(400, "No files matched any sample folders")

    return {"jobs": job_results}
