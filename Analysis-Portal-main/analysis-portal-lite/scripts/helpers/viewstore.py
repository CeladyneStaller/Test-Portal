"""
View tab data layer.
=====================

Read-only access to the JSONBin store for the viewer. Transport lives in
`jsonbin.py`; this module adds the read paths the viewer needs — filtering,
caching, and materialising stored sidecars back onto disk.

Why materialisation matters: `compare_polcurves.run()` locates each source with
`find_sidecar(output_dir, filename)`, reading from disk. Writing decoded
sidecars into the same `{dir}/_plot_data/{plot}.json` layout the analysis
scripts produce means the existing comparison script runs unchanged against
historical data — grouping modes, clean labels, condition subtitles, readout
boxes and the Excel Metrics sheet all come along for free.

The viewer is read-only (fork F). Nothing here writes to JSONBin.
"""

import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.helpers import jsonbin
from scripts.helpers.record import decode_sidecars

# Detail bins used to be immutable, which is why this cache has no expiry.
# Sample-keyed merging makes them mutable: a later run rewrites the same bin.
# The cache key therefore carries the index entry's timestamp, which advances
# on every merge — a stale record simply stops being addressable and ages out
# by LRU, so nothing needs to invalidate it explicitly.
# 32 x ~360 KB is roughly 11 MB.
DETAIL_CACHE_SIZE = 32

# The index is appended to as runs complete, so it gets a short TTL rather than
# being cached indefinitely.
INDEX_TTL_S = 30

_detail_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_cache_lock = threading.Lock()
_index_cache: Dict[str, Any] = {'at': 0.0, 'data': None}


# ─────────────────────────────────────────────────────────────────────
#  Index
# ─────────────────────────────────────────────────────────────────────

def fetch_index(force: bool = False) -> Dict[str, Any]:
    """Read the index, cached for INDEX_TTL_S."""
    with _cache_lock:
        fresh = (not force
                 and _index_cache['data'] is not None
                 and (time.time() - _index_cache['at']) < INDEX_TTL_S)
        if fresh:
            return _index_cache['data']

    index = jsonbin.fetch_index()

    with _cache_lock:
        _index_cache['data'] = index
        _index_cache['at'] = time.time()
    return index


def _entry_analyses(entry: Dict[str, Any]) -> List[str]:
    return sorted({d.get('Analysis', '') for d in entry.get('Data', [])
                   if d.get('Analysis')})


def list_runs(*, sample: Optional[str] = None,
              script: Optional[str] = None,
              analysis: Optional[str] = None,
              since: Optional[str] = None,
              until: Optional[str] = None,
              limit: Optional[int] = None,
              force: bool = False) -> Dict[str, Any]:
    """Return index entries, newest first, optionally filtered.

    Filters are the initial set agreed in scope: sample name substring, script,
    analysis type, and an ISO date range. Condition filtering is deferred.

    Timestamps are ISO-8601 Z strings, so lexicographic comparison is also
    chronological and no parsing is needed.
    """
    index = fetch_index(force=force)
    runs = list(index.get('runs', []))

    if sample:
        needle = sample.lower()
        runs = [r for r in runs
                if needle in str(r.get('sample_name', '')).lower()]
    if script:
        runs = [r for r in runs if r.get('script') == script]
    if analysis:
        runs = [r for r in runs if analysis in _entry_analyses(r)]
    if since:
        runs = [r for r in runs if str(r.get('timestamp', '')) >= since]
    if until:
        runs = [r for r in runs if str(r.get('timestamp', '')) <= until]

    # Order by experiment date where it is known, falling back to the analysis
    # date. Both are ISO, so the leading YYYY-MM-DD compares lexicographically
    # even though one is a date and the other a datetime.
    runs.sort(key=_sort_key, reverse=True)
    total = len(runs)
    if limit:
        runs = runs[:limit]

    return {
        'runs': runs,
        'total': total,
        'returned': len(runs),
        'facets': index_facets(index),
    }


def _sort_key(entry: Dict[str, Any]) -> str:
    """Experiment date when recoverable, else the analysis date."""
    return str(entry.get('run_date') or entry.get('timestamp', ''))[:10]


def index_facets(index: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """Distinct values available for filtering, for populating the UI controls."""
    if index is None:
        index = fetch_index()
    runs = index.get('runs', [])
    samples, scripts, analyses = set(), set(), set()
    for r in runs:
        if r.get('sample_name'):
            samples.add(r['sample_name'])
        if r.get('script'):
            scripts.add(r['script'])
        analyses.update(_entry_analyses(r))
    return {
        'samples': sorted(samples),
        'scripts': sorted(scripts),
        'analyses': sorted(analyses),
    }


# ─────────────────────────────────────────────────────────────────────
#  Detail bins
# ─────────────────────────────────────────────────────────────────────

def _entry_for(key: str, index: Optional[Dict[str, Any]] = None
               ) -> Optional[Dict[str, Any]]:
    """Resolve an index entry from a bin id, sample name, or job id.

    `bin_id` is tried first because it is the only one guaranteed unambiguous.
    A sample may still have more than one entry — anything written before
    merging was enabled — and resolving those by sample name would silently
    hand back whichever is newest, so a caller holding a specific entry gets
    the bin it actually asked for.

    Resolving through the index also validates the key: an arbitrary bin id
    cannot be used to read a bin the index does not reference.
    """
    runs = (index or fetch_index()).get('runs', [])
    for field in ('bin_id', 'sample_name', 'job_id'):
        matches = [e for e in runs if e.get(field) == key]
        if len(matches) == 1:
            return matches[0]
        if matches:
            # Ambiguous only for sample_name pre-migration; take the newest.
            matches.sort(key=lambda e: str(e.get('timestamp', '')))
            return matches[-1]
    return None


def fetch_detail(key: str) -> Dict[str, Any]:
    """Fetch a run's detail bin by bin id, sample name, or job id.

    Cached against the index entry's timestamp, so a merged bin is re-read
    rather than served stale.
    """
    entry = _entry_for(key)
    if entry is None:
        # Could have been written moments ago. Retry once against a fresh index.
        entry = _entry_for(key, fetch_index(force=True))
    if entry is None or not entry.get('bin_id'):
        raise KeyError(f'no indexed run matching {key!r}')

    bin_id = entry['bin_id']
    cache_key = f"{bin_id}@{entry.get('timestamp', '')}"

    with _cache_lock:
        if cache_key in _detail_cache:
            _detail_cache.move_to_end(cache_key)
            return _detail_cache[cache_key]

    payload = jsonbin._request(
        f'{jsonbin._JSONBIN_BASE}/{bin_id}/latest', method='GET')
    record = payload.get('record') if isinstance(payload, dict) else None
    if not isinstance(record, dict):
        raise RuntimeError(f'detail bin {bin_id} returned no usable record')

    with _cache_lock:
        _detail_cache[cache_key] = record
        _detail_cache.move_to_end(cache_key)
        while len(_detail_cache) > DETAIL_CACHE_SIZE:
            _detail_cache.popitem(last=False)
    return record


def cache_stats() -> Dict[str, Any]:
    """Cache state, for the diagnostics endpoint."""
    with _cache_lock:
        return {
            'detail_cached': len(_detail_cache),
            'detail_capacity': DETAIL_CACHE_SIZE,
            'detail_keys': list(_detail_cache.keys()),
            'index_age_s': (round(time.time() - _index_cache['at'], 1)
                            if _index_cache['data'] is not None else None),
            'index_ttl_s': INDEX_TTL_S,
        }


def clear_cache() -> None:
    with _cache_lock:
        _detail_cache.clear()
        _index_cache['data'] = None
        _index_cache['at'] = 0.0


# ─────────────────────────────────────────────────────────────────────
#  Plot inventory
# ─────────────────────────────────────────────────────────────────────

def run_plots(key: str) -> List[Dict[str, Any]]:
    """List a run's plots, flagging which can be re-rendered.

    A plot is renderable only if its sidecar was stored. Cleaning sidecars are
    excluded at write time because they dominate the transport budget, so
    cleaning plots appear here with `renderable: False` — their metrics are
    intact, only the plot data is absent.
    """
    detail = fetch_detail(key)
    stored = set(decode_sidecars(detail).keys())

    out: List[Dict[str, Any]] = []
    for bucket, plots in (detail.get('metrics') or {}).items():
        for plot_name, entry in plots.items():
            out.append({
                'plot': plot_name,
                'analysis': bucket,
                'conditions': entry.get('conditions') or {},
                'values': entry.get('values') or {},
                'renderable': plot_name in stored,
            })
    out.sort(key=lambda p: (p['analysis'], p['plot']))
    return out


# ─────────────────────────────────────────────────────────────────────
#  Materialisation
# ─────────────────────────────────────────────────────────────────────

def materialize_sidecars(key: str, dest_dir: Path,
                         plots: Optional[List[str]] = None) -> List[str]:
    """Write a run's stored sidecars to disk in analysis-output layout.

    Produces `{dest_dir}/_plot_data/{plot}.json`, which is exactly what
    `find_sidecar()` expects — so anything that reads sidecars off disk works
    against historical runs without modification.

    Returns the plot names actually written. Requested plots whose sidecars were
    not stored are skipped silently; callers should compare against the returned
    list rather than assume.
    """
    detail = fetch_detail(key)
    sidecars = decode_sidecars(detail)
    if plots is not None:
        wanted = set(plots)
        sidecars = {k: v for k, v in sidecars.items() if k in wanted}

    dest = Path(dest_dir) / '_plot_data'
    dest.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for name, sidecar in sidecars.items():
        (dest / f'{name}.json').write_text(
            json.dumps(sidecar, ensure_ascii=False))
        written.append(name)
    return sorted(written)


def materialize_for_compare(selections: List[Dict[str, str]],
                            root: Path) -> List[Dict[str, str]]:
    """Stage historical plots for the comparison script.

    `selections` is [{key, plot, label?}, ...] where `key` identifies a run —
    a bin id, sample name or job id. `job_id` is still accepted for callers
    that have not been updated.

    Each run gets its own directory under `root`, mirroring how live jobs are
    laid out, so the returned list can be passed straight through as the
    comparison script's `sources` parameter.

    Selections whose sidecar was not stored are omitted.
    """
    def _key(sel: Dict[str, str]) -> str:
        return str(sel.get('key') or sel.get('job_id') or '')

    by_run: Dict[str, List[str]] = {}
    for sel in selections:
        by_run.setdefault(_key(sel), []).append(sel['plot'])

    available: Dict[str, set] = {}
    dirs: Dict[str, Path] = {}
    for i, (key, plots) in enumerate(by_run.items()):
        # Directory names come from the position rather than the key: a key may
        # be a sample name, and those contain characters that are awkward on a
        # filesystem.
        run_dir = Path(root) / f'run{i}'
        dirs[key] = run_dir
        available[key] = set(materialize_sidecars(key, run_dir, plots))

    sources: List[Dict[str, str]] = []
    for sel in selections:
        key, plot = _key(sel), sel['plot']
        if plot not in available.get(key, set()):
            continue
        detail = fetch_detail(key)
        sources.append({
            'job_id': detail.get('job_id', key),
            'label': sel.get('label', ''),
            # The comparison script strips the extension to find the sidecar,
            # so the suffix here is nominal.
            'filename': f'{plot}.png',
            'output_dir': str(dirs[key]),
            'sample_name': detail.get('sample_name', ''),
        })
    return sources