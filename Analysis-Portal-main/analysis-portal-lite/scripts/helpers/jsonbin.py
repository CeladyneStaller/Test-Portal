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
from typing import Any, Dict, List, Optional, Tuple

try:
    import urllib.request as _ureq
    import urllib.error as _uerr
except ImportError:  # pragma: no cover
    _ureq = None
    _uerr = None

from scripts.helpers.record import (
    attach_sidecars,
    build_detail_record,
    build_index_entry,
    decode_sidecars,
    detail_bin_name,
    is_comparison_script,
    load_sidecars,
    merge_detail_record,
    merge_index_entry,
    select_sidecars,
    sidecar_bucket_sizes,
    sidecar_sizes,
    strip_sidecars,
)

_JSONBIN_BASE = "https://api.jsonbin.io/v3/b"

# JSONBin fronts its API with nginx, which rejects bodies over its
# client_max_body_size with a 413 before the request reaches their application.
# That proxy cap is far below the documented 10 MB bin size — nginx defaults to
# 1 MB — so the effective transport limit is the proxy's, not the plan's.
# Sidecars are compressed to stay under it; this is the backstop for runs that
# still exceed it, and the threshold leaves headroom for headers and framing.
MAX_BODY_BYTES = int(os.environ.get('JSONBIN_MAX_BODY_BYTES', 900_000))

# Characterizations whose sidecars are not stored. Cleaning CVs carry every
# point of every cycle across three panels — a single 50-cycle file measured
# ~242 KB raw — and a qualification run can contain a dozen of them, which on
# its own exceeds the transport limit. Their metrics and summary are unaffected;
# only plot re-rendering for that bucket is given up.
# Override with a comma-separated list, or set empty to store everything.
_exclude_raw = os.environ.get('JSONBIN_SIDECAR_EXCLUDE', 'cleaning')
SIDECAR_EXCLUDE_BUCKETS = {b.strip() for b in _exclude_raw.split(',') if b.strip()}

_TIMEOUT_S = 20
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 1.5  # exponential: 1.5, 3.0, 6.0

# Merge a run into the sample's existing bin rather than creating a new one.
# An env kill-switch rather than a constant: this changes the write path and
# has a data-displacement mode, so reverting to append-only should not need a
# code change.
MERGE_BY_SAMPLE = os.environ.get(
    'JSONBIN_MERGE_BY_SAMPLE', 'true').strip().lower() not in ('0', 'false', 'no')

# Serializes the index read-modify-write across concurrent job completions.
_index_lock = threading.Lock()

# One lock per sample. Merging is read-modify-write on a detail bin, so two
# jobs finishing for the same sample would otherwise lose one another's work.
# Locks are acquired sample-first then index, always in that order, so the two
# cannot deadlock.
_sample_locks: Dict[str, threading.Lock] = {}
_sample_locks_guard = threading.Lock()


def _sample_lock(sample_name: str) -> threading.Lock:
    key = sample_name or '__unnamed__'
    with _sample_locks_guard:
        lock = _sample_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _sample_locks[key] = lock
        return lock


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


def fetch_detail_bin(bin_id: str) -> Dict[str, Any]:
    """Read one detail bin's record."""
    payload = _request(f"{_JSONBIN_BASE}/{bin_id}/latest", method='GET')
    record = payload.get('record') if isinstance(payload, dict) else None
    if not isinstance(record, dict):
        raise RuntimeError(f'detail bin {bin_id} returned no usable record')
    return record


def update_detail_bin(bin_id: str, record: Dict[str, Any]) -> None:
    """Overwrite an existing detail bin.

    As with the index, X-Bin-Private is omitted: on a PUT it makes JSONBin
    create a new bin instead of updating the target.
    """
    _request(f"{_JSONBIN_BASE}/{bin_id}", method='PUT', body=record)


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


def find_sample_entry(index: Dict[str, Any],
                      sample_name: str) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """Most recent index entry for a sample, with its position.

    Returns (None, None) when the sample has no entry yet.

    A sample may still have several entries — anything written before merging
    was switched on, including the degraded push and its retry. The most recent
    is the one merged into; the others are left alone rather than guessed at,
    because consolidating them properly means fetching and merging their detail
    bins, which is phase 5's job.
    """
    matches = [(i, e) for i, e in enumerate(index.get('runs', []))
               if e.get('sample_name') == sample_name]
    if not matches:
        return None, None
    matches.sort(key=lambda t: str(t[1].get('run_date')
                                   or t[1].get('timestamp', '')))
    return matches[-1]


def write_index_entry(index: Dict[str, Any], entry: Dict[str, Any],
                      position: Optional[int]) -> int:
    """Replace the entry at `position`, or append when it is None.

    Caller must hold _index_lock and must have read `index` inside that lock.
    """
    runs = index.setdefault('runs', [])
    if position is None:
        runs.append(entry)
    else:
        runs[position] = entry
    _write_index(index)
    return len(runs)


# ─────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────

def _plot_bucket_of(sidecar: Dict[str, Any]) -> str:
    from scripts.helpers.record import plot_bucket
    return plot_bucket(sidecar.get('plot_type', 'unknown'))


def _plot_step_of(plot_name: str) -> str:
    from scripts.helpers.record import parse_conditions
    return str(parse_conditions(plot_name).get('step') or '')


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
                           'bin_id': None, 'n_runs': None,
                           'body_bytes': None, 'sidecars_omitted': False,
                           'sidecars_kept': 0, 'sidecars_dropped': 0,
                           'sidecar_bytes_by_bucket': {},
                           'merged': False, 'sidecars_displaced': 0,
                           'n_jobs': None}

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

    # Base record first, sidecars attached afterwards. Sidecar selection needs
    # to know how much of the wire budget the rest of the record consumes.
    try:
        record = build_detail_record(
            job_id=job_id, sample_name=sample_name, script=script,
            timestamp=timestamp, input_files=input_files,
            output_dir=Path(output_dir), summary=summary,
            include_sidecars=False)
        all_sidecars = load_sidecars(Path(output_dir))
    except Exception as e:
        out['reason'] = f'record assembly failed: {e}'
        return out

    # Report the full breakdown regardless of what is ultimately stored, so the
    # space consumers are visible from the job record without guesswork.
    out['sidecar_bytes_by_bucket'] = sidecar_bucket_sizes(all_sidecars)

    if extra:
        record['extra'] = extra

    if not record.get('metrics'):
        out['reason'] = 'skipped: no sidecar data found in output'
        return out

    # ── locate the sample's existing bin, if merging is on ──
    #
    # The index is read once here and reused for the write, both inside the
    # index lock, so a concurrent push cannot slip between the lookup and the
    # update. The sample lock spans the whole fetch-merge-write sequence for
    # this sample's detail bin.
    existing_entry: Optional[Dict[str, Any]] = None
    existing_record: Optional[Dict[str, Any]] = None
    entry_pos: Optional[int] = None
    bin_id: Optional[str] = None
    index: Optional[Dict[str, Any]] = None

    sample_lk = _sample_lock(sample_name or '')
    with sample_lk:
        if MERGE_BY_SAMPLE:
            try:
                with _index_lock:
                    index = fetch_index()
                entry_pos, existing_entry = find_sample_entry(
                    index, sample_name or '')
                if existing_entry and existing_entry.get('bin_id'):
                    bin_id = existing_entry['bin_id']
                    existing_record = fetch_detail_bin(bin_id)
            except Exception as e:
                # A lookup failure must not lose the run. Fall back to creating
                # a new bin, which is the pre-merge behaviour.
                out['reason'] = f'sample lookup failed, creating a new bin: {e}'
                existing_entry = existing_record = index = None
                entry_pos = bin_id = None

        # ── merge, then fit the *merged* result to the wire budget ──
        #
        # The limit applies to the merged whole, not the increment, so a run
        # that would have fit on its own can still overflow once combined.
        # Fork D: the incoming run's sidecars are protected and previously
        # stored ones are dropped first — a fresh measurement must never
        # silently fail to store its plots.
        incoming_names = set(all_sidecars)
        merged_sidecars = dict(all_sidecars)
        if existing_record:
            prior = decode_sidecars(existing_record)
            touched = {(b, str((e.get('conditions') or {}).get('step') or ''))
                       for b, plots in (record.get('metrics') or {}).items()
                       for e in plots.values()}
            for name, sc in prior.items():
                b = _plot_bucket_of(sc)
                st = _plot_step_of(name)
                if (b, st) not in touched:
                    merged_sidecars.setdefault(name, sc)

        base = merge_detail_record(existing_record, record) if existing_record \
            else record
        attach_sidecars(base, {})
        base_bytes = len(json.dumps(base, ensure_ascii=False).encode())

        kept, dropped = select_sidecars(
            merged_sidecars,
            budget_bytes=max(0, MAX_BODY_BYTES - base_bytes) * 4,
            exclude_buckets=SIDECAR_EXCLUDE_BUCKETS)

        attach_sidecars(base, kept)
        body_bytes = len(json.dumps(base, ensure_ascii=False).encode())

        # Trim largest-first, but sacrifice previously-stored sidecars before
        # any belonging to the run being written.
        while body_bytes > MAX_BODY_BYTES and kept:
            ranked = list(sidecar_sizes(kept))
            victim = next((n for n in ranked if n not in incoming_names), None)
            if victim is None:
                victim = ranked[0]
            kept.pop(victim)
            dropped.append(victim)
            attach_sidecars(base, kept)
            body_bytes = len(json.dumps(base, ensure_ascii=False).encode())

        if body_bytes > MAX_BODY_BYTES:
            base = strip_sidecars(
                base, f'record was {body_bytes:,} B, over the '
                      f'{MAX_BODY_BYTES:,} B transport limit')
            body_bytes = len(json.dumps(base, ensure_ascii=False).encode())

        record = base
        out['body_bytes'] = body_bytes
        out['sidecars_kept'] = len(kept)
        out['sidecars_dropped'] = len(dropped)
        out['sidecars_omitted'] = bool(dropped)
        out['merged'] = bool(existing_record)
        # Visible per fork D: how many already-stored sidecars this merge cost.
        displaced = [n for n in dropped if n not in incoming_names]
        out['sidecars_displaced'] = len(displaced)

        # ── write the detail bin ──
        was_update = bool(bin_id)
        try:
            if bin_id:
                update_detail_bin(bin_id, record)
            else:
                bin_id = create_detail_bin(
                    record, name=detail_bin_name(record, script_short))
        except Exception as e:
            out['reason'] = (f'detail bin {"update" if was_update else "creation"} '
                             f'failed: {e}')
            return out
        out['bin_id'] = bin_id

        # A bin written but not referenced by the index fails two different
        # ways depending on how it got there, and the distinction matters when
        # cleaning up: a freshly created bin is unreachable, whereas an updated
        # one is still indexed and merely newer than the index describes.
        def _stranded(stage: str, err: Exception) -> str:
            if was_update:
                return (f'{stage}: {err} — detail bin {bin_id} was updated but '
                        f'the index still describes its previous state')
            return (f'{stage}: {err} — orphan detail bin {bin_id} retained')

        # ── write the index entry ──
        try:
            entry = build_index_entry(record, bin_id)
            if existing_entry:
                entry = merge_index_entry(existing_entry, entry)
            out['n_jobs'] = entry.get('n_jobs', 1)
        except Exception as e:
            out['reason'] = _stranded('index entry assembly failed', e)
            return out

        try:
            with _index_lock:
                if index is None:
                    # Reached only when merging is off, or the lookup failed.
                    # Either way append rather than replace: with merging off
                    # that is the intended behaviour, and after a failed lookup
                    # a duplicate entry is a far better outcome than
                    # overwriting one whose contents were never read.
                    index = fetch_index()
                    entry_pos = None
                out['n_runs'] = write_index_entry(index, entry, entry_pos)
        except Exception as e:
            out['reason'] = _stranded('index write failed', e)
            return out

    out['pushed'] = True
    if out['sidecars_dropped']:
        note = (f"pushed with {out['sidecars_kept']} of "
                f"{out['sidecars_kept'] + out['sidecars_dropped']} sidecars — "
                f"the rest were excluded by bucket or did not fit the "
                f"transport limit; metrics and summary are intact")
        if out['sidecars_displaced']:
            note += (f". {out['sidecars_displaced']} previously stored "
                     f"sidecar(s) were displaced by this merge")
        out['reason'] = note
    return out