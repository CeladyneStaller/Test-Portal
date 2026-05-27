"""
JSONBin metrics persistence
============================
Pushes a flat metric summary for each completed analysis job to a shared
JSONBin collection ({"runs": [...]}).

Environment variables required:
  JSONBIN_API_KEY        — Master Key from jsonbin.io account
  JSONBIN_METRICS_BIN_ID — ID of the bin holding the collection

If either is missing, push_job_metrics() returns False silently. This
keeps dev environments that haven't configured JSONBin from failing.

Concurrency: all push operations are serialized through _push_lock so
the read-modify-write cycle doesn't lose data when two jobs finish at
nearly the same time. (Within a single Railway instance only — multi-
instance deployments would need optimistic locking via X-Bin-Versioning.)
"""

import os
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import urllib.request as _ureq
    import urllib.error as _uerr
except ImportError:
    _ureq = None
    _uerr = None


_JSONBIN_BASE = "https://api.jsonbin.io/v3/b"
_TIMEOUT_S = 20
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 1.5  # exponential: 1.5, 3.0, 6.0

# Process-level lock to serialize read-modify-write across concurrent jobs.
_push_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────
#  Metric extraction
# ─────────────────────────────────────────────────────────────────────

def _parse_metric_kv(text_str: str) -> Dict[str, Any]:
    """Parse 'KEY = VALUE' or 'KEY: VALUE' lines. Numeric values become floats."""
    import re
    result = {}
    if not text_str:
        return result
    for line in text_str.split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(.+?)\s*=\s*(.+)$', line) or re.match(r'^(.+?):\s*(.+)$', line)
        if not m:
            continue
        key = m.group(1).strip()
        raw_val = m.group(2).strip()
        if not key or not raw_val:
            continue
        # Try to convert to a number (strip units like " V", " mA/cm²", "kPa")
        num_match = re.match(r'^([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\S*)', raw_val)
        if num_match:
            try:
                value = float(num_match.group(1))
                # Keep the unit as separate key for traceability
                unit = num_match.group(2).strip()
                if unit:
                    result[key] = {'value': value, 'unit': unit}
                else:
                    result[key] = value
                continue
            except ValueError:
                pass
        result[key] = raw_val
    return result


def _plot_type_bucket(plot_type: str) -> str:
    """Map plot_type to a top-level characterization bucket name."""
    pt = (plot_type or 'other').lower()
    if pt.startswith('activation'):
        return 'activation'
    if pt.startswith('cleaning'):
        return 'cleaning'
    if pt.startswith('crossover') or 'h2x' in pt:
        return 'crossover'
    if pt.startswith('eis') or pt.startswith('nyquist'):
        return 'eis'
    if pt.startswith('ecsa'):
        return 'ecsa'
    if pt.startswith('ocv'):
        return 'ocv'
    if pt.startswith('polcurve') or pt in ('losses_vs_cycle', 'j_vs_cycle',
                                             'model_fit', 'ir_correction'):
        return 'polcurve'
    if pt.startswith('durability'):
        return 'durability'
    if pt.startswith('clr') or pt.startswith('coth'):
        return 'CLR'
    return pt


def _parse_conditions_from_filename(filename: str) -> Dict[str, Any]:
    """Reuse the comparison script's condition parser if available; else
    do a minimal parse for T/RH/P."""
    import re
    if not filename:
        return {}
    name = os.path.splitext(os.path.basename(filename))[0]
    out: Dict[str, Any] = {}
    m = re.search(r'(\d+(?:o\d+)?)C(?![A-Za-z])', name)
    if m:
        try:
            out['T_C'] = float(m.group(1).replace('o', '.'))
        except ValueError:
            pass
    m = re.search(r'(\d+(?:o\d+)?)RH', name, re.IGNORECASE)
    if m:
        try:
            out['RH_pct'] = float(m.group(1).replace('o', '.'))
        except ValueError:
            pass
    m = re.search(r'(\d+(?:o\d+)?)\s*(kPa|barg|psi|bar)', name, re.IGNORECASE)
    if m:
        try:
            out['P_value'] = float(m.group(1).replace('o', '.'))
            out['P_unit'] = m.group(2)
        except ValueError:
            pass
    return out


def extract_metrics_from_output(output_dir: Path) -> Dict[str, Any]:
    """
    Walk all sidecar JSONs under output_dir, parse text annotations + ref
    line labels into metric dicts, organize by characterization bucket.

    Returns
    -------
    {
      'polcurve': {'<filename>': {'OCV': 0.95, ...}, ...},
      'eis':      {'<filename>': {'HFR': {'value': 0.045, 'unit': 'Ω·cm²'}}, ...},
      'conditions': {'T_C': 80.0, 'RH_pct': 100.0, ...},  # union of all parsed
      ...
    }
    """
    metrics: Dict[str, Any] = {}
    union_conditions: Dict[str, Any] = {}

    output_dir = Path(output_dir)
    if not output_dir.exists():
        return metrics

    for sidecar_path in output_dir.rglob('*.json'):
        if '_plot_data' not in sidecar_path.parts:
            continue
        try:
            with open(sidecar_path, 'r') as f:
                sidecar = json.load(f)
        except Exception:
            continue

        plot_type = sidecar.get('plot_type', 'unknown')
        bucket = _plot_type_bucket(plot_type)
        plot_name = sidecar_path.stem

        # Parse text annotations + axhline/axvline labels containing '='
        plot_metrics: Dict[str, Any] = {}
        axes = sidecar.get('data', {}).get('axes', [])
        for ax in axes:
            if ax.get('is_twin'):
                continue
            for txt in ax.get('texts', []):
                plot_metrics.update(_parse_metric_kv(txt.get('text', '')))
            for ref_line in ax.get('axhlines', []) + ax.get('axvlines', []):
                lbl = (ref_line.get('label') or '').strip()
                if lbl and '=' in lbl:
                    plot_metrics.update(_parse_metric_kv(lbl))

        if plot_metrics:
            metrics.setdefault(bucket, {})[plot_name] = plot_metrics

        # Pull conditions out of the filename too
        cond = _parse_conditions_from_filename(plot_name)
        for k, v in cond.items():
            union_conditions.setdefault(k, v)

    if union_conditions:
        metrics['conditions'] = union_conditions
    return metrics


# ─────────────────────────────────────────────────────────────────────
#  JSONBin I/O
# ─────────────────────────────────────────────────────────────────────

def _api_key() -> Optional[str]:
    return os.environ.get('JSONBIN_API_KEY') or None


def _bin_id() -> Optional[str]:
    return os.environ.get('JSONBIN_METRICS_BIN_ID') or None


def _bin_url() -> Optional[str]:
    bid = _bin_id()
    if not bid:
        return None
    return f"{_JSONBIN_BASE}/{bid}"


def _request(url: str, method: str = 'GET',
             body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One HTTP call with retries + exponential backoff."""
    if _ureq is None:
        raise RuntimeError("urllib not available")
    api_key = _api_key()
    if not api_key:
        raise RuntimeError("JSONBIN_API_KEY env var is not set")

    headers = {
        'Content-Type': 'application/json',
        'X-Master-Key': api_key,
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode('utf-8')

    last_err: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            req = _ureq.Request(url, data=data, headers=headers, method=method)
            with _ureq.urlopen(req, timeout=_TIMEOUT_S) as resp:
                raw = resp.read().decode('utf-8')
            return json.loads(raw) if raw else {}
        except _uerr.HTTPError as e:
            # Don't retry on 4xx (client error) — those won't fix themselves
            body_txt = ''
            try:
                body_txt = e.read().decode('utf-8', errors='replace')
            except Exception:
                pass
            if 400 <= e.code < 500 and e.code != 429:
                raise RuntimeError(
                    f"JSONBin HTTP {e.code} on {method} {url}: {body_txt}") from e
            last_err = e
        except Exception as e:
            last_err = e
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
    raise RuntimeError(f"JSONBin call failed after {_MAX_RETRIES} attempts: {last_err}")


def _fetch_collection() -> Dict[str, Any]:
    """Read the current state of the metrics collection bin."""
    url = _bin_url() + '/latest'
    payload = _request(url, method='GET')
    # JSONBin v3 wraps data in {"record": ..., "metadata": ...}
    record = payload.get('record') if isinstance(payload, dict) else None
    if record is None:
        record = payload
    if not isinstance(record, dict):
        record = {}
    if 'runs' not in record or not isinstance(record.get('runs'), list):
        record = {'runs': []}
    return record


def _write_collection(record: Dict[str, Any]) -> None:
    """Overwrite the metrics collection bin (PUT)."""
    # NOTE: JSONBin PUT must NOT include X-Bin-Private header (it would
    # treat the call as create-new). We only pass Content-Type + Master Key.
    _request(_bin_url(), method='PUT', body=record)


# ─────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────

def is_configured() -> bool:
    """True if the env vars needed to push are set."""
    return bool(_api_key() and _bin_id())


def push_job_metrics(*,
                     job_id: str,
                     sample_name: Optional[str],
                     script: str,
                     output_dir: Path,
                     input_files: Optional[List[str]] = None,
                     extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract metrics from job output and append a record to the JSONBin
    collection.

    Returns a dict with:
        {'pushed': bool, 'reason': str | None, 'metrics': dict}

    Never raises — failures are surfaced via the return dict so callers
    can decide what to do with them (typically: log a warning, keep going).
    """
    out: Dict[str, Any] = {'pushed': False, 'reason': None, 'metrics': {}}

    if not is_configured():
        out['reason'] = 'JSONBin not configured (missing JSONBIN_API_KEY or JSONBIN_METRICS_BIN_ID)'
        return out

    try:
        metrics = extract_metrics_from_output(Path(output_dir))
    except Exception as e:
        out['reason'] = f'metric extraction failed: {e}'
        return out
    out['metrics'] = metrics

    record_entry: Dict[str, Any] = {
        'job_id': job_id,
        'sample_name': sample_name or '',
        'script': script,
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'input_files': list(input_files or []),
        'metrics': metrics,
    }
    if extra:
        record_entry['extra'] = extra

    try:
        with _push_lock:
            collection = _fetch_collection()
            collection.setdefault('runs', []).append(record_entry)
            _write_collection(collection)
    except Exception as e:
        out['reason'] = f'JSONBin push failed: {e}'
        return out

    out['pushed'] = True
    return out
