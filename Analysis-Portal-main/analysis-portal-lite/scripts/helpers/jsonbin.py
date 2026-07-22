"""
JSONBin transport — index bin + per-run detail bins.
=====================================================

Record assembly lives in `record.py` and is pure. This module is transport only:
creating detail bins, appending to the index, retry and error handling.

Environment variables
---------------------
  JSONBIN_API_KEY         Master Key from the jsonbin.io account
  JSONBIN_COLLECTION_ID   Collection that groups the detail bins
  JSONBIN_INDEX_BIN_ID    Bin holding {"schema": 2, "runs": [...]}

If any is missing, push_job_metrics() returns a skip result and does nothing,
so environments without JSONBin configured behave normally.

  JSONBIN_METRICS_BIN_ID  Legacy single-bin store. No longer read. The old bin
                          is left untouched as the schema-1 archive.

Architecture notes
------------------
Detail bins are created with POST, which is atomic — no coordination needed.
Only the index append is read-modify-write, and it is serialized through
_index_lock. That covers a single Railway instance, which is the current
deployment; horizontal scaling would need optimistic locking on JSONBin's
version counter.

Two JSONBin quirks are load-bearing here:
  * X-Bin-Private on a PUT creates a new bin instead of updating the target,
    so update requests carry only Content-Type and the key.
  * Cloudflare rejects the default Python-urllib User-Agent with error 1010,
    so every request sends browser-like headers.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import urllib.request as _ureq
    import urllib.error as _uerr
except ImportError:  # pragma: no cover
    _ureq = None
    _uerr = None

from scripts.helpers.record import (
    build_detail_record,
    build_index_entry,
    detail_bin_name,
    is_comparison_script,
)

_JSONBIN_BASE = "https://api.jsonbin.io/v3/b"
_TIMEOUT_S = 20
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 1.5  # exponential: 1.5, 3.0, 6.0

# Serializes the index read-modify-write across concurrent job completions.
_index_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────

def _api_key() -> Optional[str]:
    return os.environ.get('JSONBIN_API_KEY') or None


def _collection_id() -> Optional[str]:
    return os.environ.get('JSONBIN_COLLECTION_ID') or None


def _index_bin_id() -> Optional[str]:
    return os.environ.get('JSONBIN_INDEX_BIN_ID') or None


def is_configured() -> bool:
    """True when every environment variable needed to push is present."""
    return bool(_api_key() and _collection_id() and _index_bin_id())


def _missing_config() -> List[str]:
    missing = []
    if not _api_key():
        missing.append('JSONBIN_API_KEY')
    if not _collection_id():
        missing.append('JSONBIN_COLLECTION_ID')
    if not _index_bin_id():
        missing.append('JSONBIN_INDEX_BIN_ID')
    return missing


# ─────────────────────────────────────────────────────────────────────
#  HTTP
# ─────────────────────────────────────────────────────────────────────

def _base_headers() -> Dict[str, str]:
    return {
        'Content-Type': 'application/json',
        'X-Master-Key': _api_key() or '',
        # Cloudflare blocks the default Python-urllib UA with error 1010.
        'User-Agent': ('Mozilla/5.0 (compatible; AnalysisPortal/1.0; '
                       '+https://anthropic.com/claude)'),
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
    }


def _request(url: str, method: str = 'GET',
             body: Optional[Dict[str, Any]] = None,
             extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """One HTTP call with retry and exponential backoff.

    Retries on 5xx, 429 and network errors. Does not retry other 4xx responses,
    which indicate a request that will not succeed on repetition.
    """
    if _ureq is None:  # pragma: no cover
        raise RuntimeError("urllib not available")
    if not _api_key():
        raise RuntimeError("JSONBIN_API_KEY env var is not set")

    headers = _base_headers()
    if extra_headers:
        headers.update(extra_headers)
    data = json.dumps(body).encode('utf-8') if body is not None else None

    last_err: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            req = _ureq.Request(url, data=data, headers=headers, method=method)
            with _ureq.urlopen(req, timeout=_TIMEOUT_S) as resp:
                raw = resp.read().decode('utf-8')
            return json.loads(raw) if raw else {}
        except _uerr.HTTPError as e:
            detail = ''
            try:
                detail = e.read().decode('utf-8', errors='replace')[:300]
            except Exception:
                pass
            if 400 <= e.code < 500 and e.code != 429:
                raise RuntimeError(
                    f"JSONBin HTTP {e.code} on {method} {url}: {detail}") from e
            last_err = e
        except Exception as e:
            last_err = e
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
    raise RuntimeError(
        f"JSONBin call failed after {_MAX_RETRIES} attempts: {last_err}")


# ─────────────────────────────────────────────────────────────────────
#  Detail bins
# ─────────────────────────────────────────────────────────────────────

def create_detail_bin(record: Dict[str, Any], *, name: str = '') -> str:
    """Create a private bin inside the collection. Returns the new bin ID.

    POST is atomic, so concurrent job completions cannot collide here.
    """
    headers = {'X-Collection-Id': _collection_id() or '',
               'X-Bin-Private': 'true'}
    if name:
        headers['X-Bin-Name'] = name[:128]

    payload = _request(_JSONBIN_BASE, method='POST', body=record,
                       extra_headers=headers)

    bin_id = (payload or {}).get('metadata', {}).get('id')
    if not bin_id:
        raise RuntimeError(
            f"JSONBin create returned no bin id: {str(payload)[:200]}")
    return bin_id


# ─────────────────────────────────────────────────────────────────────
#  Index bin
# ─────────────────────────────────────────────────────────────────────

def fetch_index() -> Dict[str, Any]:
    """Read the current index. Returns a well-formed record even if empty."""
    payload = _request(f"{_JSONBIN_BASE}/{_index_bin_id()}/latest", method='GET')
    record = payload.get('record') if isinstance(payload, dict) else None
    if record is None:
        record = payload
    if not isinstance(record, dict):
        record = {}
    if not isinstance(record.get('runs'), list):
        record = {'schema': 2, 'runs': []}
    record.setdefault('schema', 2)
    return record


def _write_index(record: Dict[str, Any]) -> None:
    """Overwrite the index bin.

    X-Bin-Private is deliberately omitted: on a PUT it causes JSONBin to create
    a new bin rather than update the target.
    """
    _request(f"{_JSONBIN_BASE}/{_index_bin_id()}", method='PUT', body=record)


def append_index_entry(entry: Dict[str, Any]) -> int:
    """Append one entry to the index. Returns the new run count.

    Caller must hold _index_lock.
    """
    index = fetch_index()
    index['runs'].append(entry)
    _write_index(index)
    return len(index['runs'])


# ─────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────

def push_job_metrics(*,
                     job_id: str,
                     sample_name: Optional[str],
                     script: str,
                     output_dir: Path,
                     input_files: Optional[List[str]] = None,
                     summary: Optional[Any] = None,
                     script_short: str = '',
                     extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist one completed analysis as a detail bin plus an index entry.

    Returns
    -------
    dict with keys:
        pushed     bool
        reason     str | None   why it did not push, or what went wrong
        bin_id     str | None   the detail bin, set even if the index append failed
        n_runs     int | None   index length after append

    Never raises. Failures are reported through the return value so a metrics
    problem can never fail an analysis that otherwise succeeded.
    """
    out: Dict[str, Any] = {'pushed': False, 'reason': None,
                           'bin_id': None, 'n_runs': None}

    # Comparison output is derived from runs already stored. Excluded here as
    # well as in record.py so the rule holds regardless of caller.
    if is_comparison_script(script):
        out['reason'] = f'skipped: {script} produces comparison output'
        return out

    if not is_configured():
        out['reason'] = ('JSONBin not configured (missing '
                         + ', '.join(_missing_config()) + ')')
        return out

    timestamp = (datetime.now(timezone.utc)
                 .isoformat(timespec='seconds').replace('+00:00', 'Z'))

    try:
        record = build_detail_record(
            job_id=job_id, sample_name=sample_name, script=script,
            timestamp=timestamp, input_files=input_files,
            output_dir=Path(output_dir), summary=summary)
    except Exception as e:
        out['reason'] = f'record assembly failed: {e}'
        return out

    if extra:
        record['extra'] = extra

    if not record.get('metrics'):
        out['reason'] = 'skipped: no sidecar data found in output'
        return out

    try:
        bin_id = create_detail_bin(
            record, name=detail_bin_name(record, script_short))
    except Exception as e:
        out['reason'] = f'detail bin creation failed: {e}'
        return out
    out['bin_id'] = bin_id

    try:
        entry = build_index_entry(record, bin_id)
    except Exception as e:
        # Fork D: the detail bin survives as an orphan rather than being deleted.
        out['reason'] = (f'index entry assembly failed: {e} — '
                         f'orphan detail bin {bin_id} retained')
        return out

    try:
        with _index_lock:
            out['n_runs'] = append_index_entry(entry)
    except Exception as e:
        out['reason'] = (f'index append failed: {e} — '
                         f'orphan detail bin {bin_id} retained')
        return out

    out['pushed'] = True
    return out