"""
Record assembly for the JSONBin index + detail architecture.
=============================================================

Pure functions. No network, no environment variables, no side effects — so this
module is fully testable offline. Transport lives in `jsonbin.py`.

Produces two artefacts per completed analysis job:

  build_detail_record(...)  → the per-run detail bin body (schema 2)
  build_index_entry(...)    → the lightweight entry appended to the index bin

Content tiers (design doc §3.4, fork F):
  Tier 1 — `summary`  : caller-supplied summary scalars from the results dict
  Tier 3 — `sidecars` : full plot data for every panel
  Tier 2 (Excel data tables) is deliberately omitted as redundant against tier 3.
"""

import base64
import gzip
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 2

# ─────────────────────────────────────────────────────────────────────
#  Sidecar encoding
# ─────────────────────────────────────────────────────────────────────
#
# Sidecars are stored compressed. JSONBin's reverse proxy (nginx) rejects
# request bodies over roughly 1 MB with a 413 — a limit well below the
# documented 10 MB bin size, and enforced before the request reaches their
# application. Sidecar arrays are highly repetitive numeric JSON and compress
# around 6.5x raw, ~4.9x after base64 framing, which brings all but pathological
# runs comfortably under the cap.
#
# Everything a human would inspect in the JSONBin dashboard — metrics, summary,
# conditions — stays plain JSON. Only the machine-only sidecar block is opaque.

SIDECAR_ENCODING = 'gzip+base64'


def encode_sidecars(sidecars: Dict[str, Any]) -> Dict[str, Any]:
    """Compress the sidecar block for transport."""
    raw = json.dumps(sidecars, ensure_ascii=False, separators=(',', ':')).encode()
    packed = base64.b64encode(gzip.compress(raw, 9)).decode('ascii')
    return {'encoding': SIDECAR_ENCODING,
            'n_plots': len(sidecars),
            'raw_bytes': len(raw),
            'data': packed}


def decode_sidecars(record: Dict[str, Any]) -> Dict[str, Any]:
    """Recover the sidecar dict from a detail record.

    Handles both the compressed form and the plain form, so a consumer does not
    need to know which was written. Returns {} when a record carries none.
    """
    block = record.get('sidecars')
    if not block:
        return {}
    if not isinstance(block, dict) or 'encoding' not in block:
        return block  # plain, pre-compression form
    if block.get('encoding') != SIDECAR_ENCODING:
        raise ValueError(f"unknown sidecar encoding {block.get('encoding')!r}")
    raw = gzip.decompress(base64.b64decode(block['data']))
    return json.loads(raw.decode('utf-8'))


def strip_sidecars(record: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Return a copy without sidecars, marked so the loss is visible.

    Used when even the compressed record exceeds the transport limit. Metrics,
    summary and conditions survive — only plot re-rendering is lost, and the
    marker records why so it is not mistaken for a run that never had any.
    """
    out = {k: v for k, v in record.items() if k != 'sidecars'}
    out['sidecars_omitted'] = reason
    return out

# ─────────────────────────────────────────────────────────────────────
#  Comparison exclusion
# ─────────────────────────────────────────────────────────────────────
#
# Comparison output never enters the store. Its numbers are derived from runs
# that are already persisted, so recording them would double-count and would
# attribute one sample's metrics to a synthetic multi-sample "run".
#
# Two barriers already exist upstream — main.py skips comparison jobs, and the
# comparison renderer writes no sidecars — but both are incidental. The first
# depends on the caller staying correct, the second on the renderer never
# gaining sidecar support. The guards here make the exclusion structural.

COMPARISON_SCRIPTS = {'Plot Comparison', 'Compare Polcurves'}


def is_comparison_script(script: Optional[str]) -> bool:
    """True if this script produces comparison output, which is never stored."""
    if not script:
        return False
    if script in COMPARISON_SCRIPTS:
        return True
    s = script.lower()
    return 'comparison' in s or s.startswith('compare ')


def _is_comparison_sidecar(plot_name: str, sidecar: Dict[str, Any]) -> bool:
    """True if a sidecar came from comparison output.

    Defensive: comparison plots carry no sidecars today. This keeps the
    exclusion holding if that ever changes.

    The plot_type test is deliberately narrow. A substring match on 'compare'
    would swallow 'polcurve_hfrcompare' — the EIS-versus-current-interrupt
    analysis — which is a legitimate measurement, not comparison output.
    Losing a real analysis silently is far worse than the defensive check
    missing a case that cannot currently occur, so only an exact type or a
    '_comparison' suffix counts.
    """
    if sidecar.get('comparison') is True:
        return True
    meta = sidecar.get('metadata') or {}
    if isinstance(meta, dict) and meta.get('comparison') is True:
        return True
    pt = str(sidecar.get('plot_type', '')).lower()
    return pt == 'comparison' or pt.endswith('_comparison')


# ─────────────────────────────────────────────────────────────────────
#  Metric text parsing
# ─────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r'^([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*(\S*)')


def parse_metric_kv(text_str: str) -> Dict[str, Any]:
    """Parse 'KEY = VALUE' / 'KEY: VALUE' lines into a dict.

    Values become {'value': float, 'unit': str} when a unit follows the number,
    a bare float when it does not, and the raw string when non-numeric.
    """
    result: Dict[str, Any] = {}
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
        num_match = _NUM_RE.match(raw_val)
        if num_match:
            try:
                value = float(num_match.group(1))
            except ValueError:
                result[key] = raw_val
                continue
            unit = num_match.group(2).strip()
            result[key] = {'value': value, 'unit': unit} if unit else value
            continue
        result[key] = raw_val
    return result


# ─────────────────────────────────────────────────────────────────────
#  Conditions
# ─────────────────────────────────────────────────────────────────────

def _num(s: Optional[str]) -> Optional[float]:
    """Decode filename numerics. 'o' stands in for a decimal point: 0o2 -> 0.2."""
    if s is None:
        return None
    try:
        return float(s.replace('o', '.'))
    except (ValueError, TypeError):
        return None


# Step must be a standalone underscore-delimited token (b14, c10, a2).
#
# The permissive form used elsewhere — ([A-Za-z])(\d+) with alpha lookarounds —
# matches 'H2' inside a gas spec when the filename has no real step, silently
# producing step='h2'. Anchoring to token boundaries removes that: every gas
# token starts with a digit, so it can never satisfy this pattern.
_STEP_RE = re.compile(r'(?:^|_)([A-Za-z]\d+)(?=_|$)')

_GAS_PATTERNS = (
    ('H2_slpm', re.compile(r'(\d+(?:o\d+)?)H2(?![A-Za-z])', re.IGNORECASE)),
    ('Air_slpm', re.compile(r'(\d+(?:o\d+)?)Air', re.IGNORECASE)),
    ('N2_slpm', re.compile(r'(\d+(?:o\d+)?)N2(?![A-Za-z])', re.IGNORECASE)),
    ('O2_slpm', re.compile(r'(\d+(?:o\d+)?)O2(?![A-Za-z])', re.IGNORECASE)),
)


def parse_conditions(filename: str) -> Dict[str, Any]:
    """Extract experimental conditions encoded in a plot filename.

    Key names are normalised for the data contract — unit-suffixed and
    self-documenting, rather than the bare forms used internally by
    compare_polcurves.

    Possible keys: step, T_C, RH_pct, P_value, P_unit,
                   H2_slpm, Air_slpm, N2_slpm, O2_slpm, V_setpoint
    All are optional; sparse results are expected and correct.
    """
    if not filename:
        return {}
    name = os.path.splitext(os.path.basename(filename))[0]
    cond: Dict[str, Any] = {}

    m = _STEP_RE.search(name)
    if m:
        cond['step'] = m.group(1).lower()

    m = re.search(r'(\d+(?:o\d+)?)C(?![A-Za-z])', name)
    if m and (v := _num(m.group(1))) is not None:
        cond['T_C'] = v

    m = re.search(r'(\d+(?:o\d+)?)RH', name, re.IGNORECASE)
    if m and (v := _num(m.group(1))) is not None:
        cond['RH_pct'] = v

    m = re.search(r'(\d+(?:o\d+)?)\s*(kPa|barg|psi|bar)', name, re.IGNORECASE)
    if m and (v := _num(m.group(1))) is not None:
        cond['P_value'] = v
        cond['P_unit'] = m.group(2)

    for key, pat in _GAS_PATTERNS:
        m = pat.search(name)
        if m and (v := _num(m.group(1))) is not None:
            cond[key] = v

    # Voltage setpoint: digits + V, not adjacent to letters (excludes 'mV', 'kV')
    m = re.search(r'(?<![A-Za-z])(\d+(?:o\d+)?)V(?![A-Za-z])', name)
    if m and (v := _num(m.group(1))) is not None:
        cond['V_setpoint'] = v

    return cond


# ─────────────────────────────────────────────────────────────────────
#  Buckets
# ─────────────────────────────────────────────────────────────────────

def plot_bucket(plot_type: str) -> str:
    """Map a sidecar plot_type to its characterization bucket."""
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
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in pt) or 'other'


# ─────────────────────────────────────────────────────────────────────
#  key_values whitelist
# ─────────────────────────────────────────────────────────────────────

# canonical name -> candidate source keys, checked in order.
# Summary scalars (tier 1) are consulted before parsed plot annotations, so a
# value taken from the results dict always wins over one scraped from display
# text. This is what makes 'Average ECSA' correct without depending on the
# malformed H_UPD readout box.
KEY_VALUES: Dict[str, List[Tuple[str, List[str]]]] = {
    'polcurve': [
        ('OCV', ['OCV', 'OCV (V)']),
        ('V @ 1 A/cm²', ['V @ 1 A/cm²', 'V@1A (V)', 'V_at_1Acm2', 'V at 1 A/cm2']),
    ],
    'eis': [
        ('HFR', ['HFR', 'Mean HFR', 'HFR_mean', 'Mean HFR (mOhm·cm2)']),
    ],
    'crossover': [
        ('|j_xover|', ['|j_xover|', 'j_xover', 'j_xover_mA_cm2',
                       'j_xover (mA/cm2)']),
    ],
    'ecsa': [
        ('Average ECSA', ['Average ECSA', 'average_ECSA_m2_per_g',
                          'Avg ECSA (m2/g)']),
    ],
}

# Units are implied per canonical key; the {value, unit} form lives in the
# detail bin. Documented here so consumers are not guessing.
KEY_VALUE_UNITS = {
    'OCV': 'V',
    'V @ 1 A/cm²': 'V',
    'HFR': 'Ω·cm²',
    '|j_xover|': 'mA/cm²',
    'Average ECSA': 'm²/g',
}


def _scalar(v: Any) -> Optional[float]:
    """Reduce a metric value to a bare number, or None if it isn't numeric."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict) and isinstance(v.get('value'), (int, float)):
        return float(v['value'])
    return None


def _summary_lookup(summary: Optional[Any]) -> Dict[str, Any]:
    """Flatten caller-supplied summary rows into one key -> value map.

    Accepts a list of row dicts (the shape run_batch already builds) or a single
    dict. Later rows do not overwrite earlier ones, so with multiple files the
    first file's values win — the index is a routing aid, and drill-down to the
    detail bin gives per-file precision.
    """
    flat: Dict[str, Any] = {}
    if not summary:
        return flat
    rows = summary if isinstance(summary, list) else [summary]
    for row in rows:
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            if k not in flat:
                flat[k] = v
    return flat


def _round_sig(v: float, sig: int = 6) -> float:
    """Round to significant figures, not decimal places.

    Index entries are size-sensitive: an unrounded 0.8992422222222223 costs 18
    characters against 0.899242 for eight. Rounding by decimal places would
    destroy small magnitudes — j0 at 1.2e-06 would collapse to 1e-06 — so this
    works in significant figures instead.
    """
    if v == 0 or not math.isfinite(v):
        return v
    return round(v, -int(math.floor(math.log10(abs(v)))) + (sig - 1))


def build_key_values(bucket: str, values: Dict[str, Any],
                     summary_flat: Dict[str, Any]) -> Dict[str, float]:
    """Promote whitelisted metrics for one analysis unit into the index.

    Summary scalars (tier 1) take precedence over metrics parsed from plot
    annotations — every candidate name is tried against the summary before any
    is tried against the parsed values. Checking source-by-source rather than
    name-by-name matters: the parsed values often carry the display-form name
    ('V @ 1 A/cm²') while the summary carries the results-dict name
    ('V_at_1Acm2'), so a name-ordered search would silently prefer the
    lower-precision annotation.
    """
    out: Dict[str, float] = {}
    for canonical, candidates in KEY_VALUES.get(bucket, []):
        chosen: Optional[float] = None
        for source in (summary_flat, values):
            for src in candidates:
                if src in source:
                    n = _scalar(source[src])
                    if n is not None:
                        chosen = n
                        break
            if chosen is not None:
                break
        if chosen is not None:
            out[canonical] = _round_sig(chosen)
    return out


# ─────────────────────────────────────────────────────────────────────
#  Sidecar loading and metric extraction
# ─────────────────────────────────────────────────────────────────────

def load_sidecars(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load every sidecar JSON under output_dir, keyed by plot name.

    Sidecars originating from comparison output are skipped — see the
    comparison-exclusion note at the top of this module.
    """
    out: Dict[str, Dict[str, Any]] = {}
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return out
    for path in sorted(output_dir.rglob('*.json')):
        if '_plot_data' not in path.parts:
            continue
        try:
            with open(path, 'r') as f:
                sidecar = json.load(f)
        except Exception:
            continue
        if not isinstance(sidecar, dict):
            continue
        if _is_comparison_sidecar(path.stem, sidecar):
            continue
        out[path.stem] = sidecar
    return out


def extract_values(sidecar: Dict[str, Any]) -> Dict[str, Any]:
    """Pull metrics from a sidecar's text annotations and labelled ref lines."""
    values: Dict[str, Any] = {}
    for ax in sidecar.get('data', {}).get('axes', []):
        if ax.get('is_twin'):
            continue
        for txt in ax.get('texts', []):
            values.update(parse_metric_kv(txt.get('text', '')))
        for ref in ax.get('axhlines', []) + ax.get('axvlines', []):
            label = (ref.get('label') or '').strip()
            if label and '=' in label:
                values.update(parse_metric_kv(label))
    return values


# ─────────────────────────────────────────────────────────────────────
#  Detail record
# ─────────────────────────────────────────────────────────────────────

def build_detail_record(*, job_id: str, sample_name: Optional[str], script: str,
                        timestamp: str, input_files: Optional[List[str]],
                        output_dir: Path,
                        summary: Optional[Any] = None,
                        include_sidecars: bool = True,
                        compress_sidecars: bool = True) -> Dict[str, Any]:
    """Assemble the schema-2 detail bin body for one completed job.

    metrics is nested bucket -> plot -> {conditions, values}. Each plot carries
    its own conditions, so no blending occurs across differing setpoints.

    Top-level `conditions` is a convenience for the single-condition case: it is
    populated only when every plot resolves to an identical condition set, and is
    None otherwise. Consumers must fall through to the per-plot blocks when it is
    None rather than assume a run-wide condition.

    Raises
    ------
    ValueError
        If `script` is a comparison script. Callers are expected to filter these
        out first; reaching here means a caller-side bug, so this fails loudly
        rather than silently writing derived data into the store.
    """
    if is_comparison_script(script):
        raise ValueError(
            f"refusing to build a record for comparison script {script!r} — "
            "comparison output is derived from runs already persisted"
        )

    sidecars = load_sidecars(output_dir)

    metrics: Dict[str, Dict[str, Any]] = {}
    condition_sets: List[str] = []

    for plot_name, sidecar in sidecars.items():
        bucket = plot_bucket(sidecar.get('plot_type', 'unknown'))
        conditions = parse_conditions(plot_name)
        values = extract_values(sidecar)
        metrics.setdefault(bucket, {})[plot_name] = {
            'conditions': conditions,
            'values': values,
        }
        # Only plots that actually encode conditions participate in the
        # uniformity test. Aggregate views — batch overlays, stitched
        # timelines — carry no conditions in their filename, and absence of
        # information is not a differing condition.
        if conditions:
            condition_sets.append(json.dumps(conditions, sort_keys=True))

    uniform = (len(set(condition_sets)) == 1) if condition_sets else False
    top_conditions = json.loads(condition_sets[0]) if uniform else None

    record: Dict[str, Any] = {
        'schema': SCHEMA_VERSION,
        'job_id': job_id,
        'sample_name': sample_name or '',
        'script': script,
        'timestamp': timestamp,
        'input_files': list(input_files or []),
        'metrics': metrics,
        'conditions': top_conditions,
    }
    if summary:
        record['summary'] = summary
    if include_sidecars and sidecars:
        record['sidecars'] = (encode_sidecars(sidecars) if compress_sidecars
                              else sidecars)
    return record


# ─────────────────────────────────────────────────────────────────────
#  Index entry
# ─────────────────────────────────────────────────────────────────────

def _analysis_units(metrics: Dict[str, Dict[str, Any]],
                    summary_flat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collapse plots into distinct (Analysis, step, Conditions) units.

    Several plots from one characterization at one setpoint — say a polcurve and
    its Tafel panel — become a single unit with their values merged.
    """
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    order: List[Tuple[str, str, str]] = []

    for bucket, plots in metrics.items():
        for _, entry in plots.items():
            conditions = dict(entry.get('conditions') or {})
            step = conditions.pop('step', '')
            key = (bucket, step, json.dumps(conditions, sort_keys=True))
            if key not in grouped:
                grouped[key] = {'Analysis': bucket, 'step': step,
                                'Conditions': conditions, '_values': {}}
                order.append(key)
            grouped[key]['_values'].update(entry.get('values') or {})

    built: List[Dict[str, Any]] = []
    for key in order:
        g = grouped[key]
        unit: Dict[str, Any] = {
            'Analysis': g['Analysis'],
            'step': g['step'],
            'Conditions': g['Conditions'],
        }
        kv = build_key_values(g['Analysis'], g['_values'], summary_flat)
        if kv:
            unit['key_values'] = kv
        built.append(unit)

    # Aggregate plots — batch overlays, comparison views — have no step, no
    # conditions and no promoted values, so they add an empty unit that is pure
    # index noise. Drop those, but only when the bucket already has a real unit:
    # a characterization whose filenames simply encode nothing (a bare cleaning
    # run, say) must still be represented.
    informative = {u['Analysis'] for u in built
                   if u['step'] or u['Conditions'] or u.get('key_values')}
    return [u for u in built
            if u['step'] or u['Conditions'] or u.get('key_values')
            or u['Analysis'] not in informative]


def build_index_entry(detail_record: Dict[str, Any], bin_id: str) -> Dict[str, Any]:
    """Build the lightweight index entry pointing at a detail bin.

    One entry per job. `Data` holds one element per analysis unit, so a Full
    Analysis run spanning several characterizations and setpoints is represented
    without collapsing them to a single arbitrary value.
    """
    summary_flat = _summary_lookup(detail_record.get('summary'))
    return {
        'job_id': detail_record.get('job_id', ''),
        'sample_name': detail_record.get('sample_name', ''),
        'script': detail_record.get('script', ''),
        'timestamp': detail_record.get('timestamp', ''),
        'bin_id': bin_id,
        'Data': _analysis_units(detail_record.get('metrics', {}), summary_flat),
    }


def detail_bin_name(detail_record: Dict[str, Any], script_short: str = '') -> str:
    """Human-legible bin name for the JSONBin dashboard. Capped at 128 chars."""
    parts = [detail_record.get('sample_name') or 'run',
             script_short or detail_record.get('script', ''),
             detail_record.get('timestamp', '')]
    name = '-'.join(p for p in parts if p)
    return name[:128]