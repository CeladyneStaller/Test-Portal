# Analysis Portal (Lite)

A lightweight web portal for your team to upload CSV data, run Python analysis
scripts, and download results. No database, no cloud storage — just FastAPI
with temporary local file storage.

## Architecture

```
Browser  →  FastAPI  →  ProcessPoolExecutor  →  Your Python Scripts
               ↕                                   (CSVs in → PNGs + Excel out)
          /tmp/jobs/
        (auto-cleaned)
```

- Jobs run in a background process pool so multiple team members can submit concurrently
- Results are stored locally and auto-cleaned after 24 hours (configurable)
- No Redis, no database, no cloud accounts required

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or with auto-reload during development
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000`

### Docker (optional)

```bash
docker build -t analysis-portal .
docker run -p 8000:8000 analysis-portal
```

## Adding Your Analysis Scripts

1. Place your script in `scripts/`
2. Add a `run(input_dir, output_dir) -> dict` wrapper function
3. Register it in `scripts/__init__.py`

### Script Contract

```python
def run(input_dir: str, output_dir: str) -> dict:
    """
    Args:
        input_dir:  Path to folder containing uploaded CSV files
        output_dir: Path to folder where you write PNGs and Excel files

    Returns:
        dict with metadata, e.g. {"files": ["plot.png"], "status": "success"}
    """
```

Your scripts can import from each other and from `scripts.helpers` freely.

### Example: registering a script

```python
# scripts/__init__.py
from scripts.polarization import run as polarization_run
from scripts.durability import run as durability_run

SCRIPT_REGISTRY = {
    "Polarization Curves": polarization_run,
    "Durability Report": durability_run,
}
```

## Configuration

Environment variables (all optional):

| Variable       | Default | Description                          |
|----------------|---------|--------------------------------------|
| `PORT`         | `8000`  | Server port                          |
| `MAX_WORKERS`  | `3`     | Concurrent analysis jobs             |
| `MAX_UPLOAD_MB`| `100`   | Per-file upload size limit           |
| `JOB_TTL_HOURS`| `24`    | Hours before results are cleaned up  |

## Project Structure

```
analysis-portal-lite/
├── app/
│   ├── main.py           # FastAPI routes + job management
│   └── templates/
│       └── index.html    # Frontend UI
├── scripts/
│   ├── __init__.py       # Script registry
│   ├── example.py        # Example script (replace with yours)
│   └── helpers/          # Shared utilities
│       └── __init__.py
├── requirements.txt
├── Dockerfile
└── README.md
```
