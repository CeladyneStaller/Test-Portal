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
import datetime
import gzip
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


def attach_sidecars(record: Dict[str, Any], sidecars: Dict[str, Any],
                    compress: bool = True) -> Dict[str, Any]:
    """Attach a chosen sidecar set to an already-built record, in place.

    Kept separate from build_detail_record so the transport can decide which
    sidecars fit the wire budget before committing them — the base record has
    to be measured first to know how much room is left.
    """
    if not sidecars:
        record.pop('sidecars', None)
        record.pop('sidecar_bytes_by_bucket', None)
        return record
    record['sidecars'] = encode_sidecars(sidecars) if compress else sidecars
    record['sidecar_bytes_by_bucket'] = sidecar_bucket_sizes(sidecars)
    return record


def strip_sidecars(record: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Return a copy without sidecars, marked so the loss is visible.

    Used when even the compressed record exceeds the transport limit. Metrics,
    summary and conditions survive — only plot re-rendering is lost, and the
    marker records why so it is not mistaken for a run that never had any.
    """
    out = {k: v for k, v in record.items() if k != 'sidecars'}
    out['sidecars_omitted'] = reason
    return out


def sidecar_sizes(sidecars: Dict[str, Any]) -> Dict[str, int]:
    """Serialized byte cost of each sidecar, largest first."""
    sizes = {
        name: len(json.dumps(sc, ensure_ascii=False,
                             separators=(',', ':')).encode())
        for name, sc in sidecars.items()
    }
    return dict(sorted(sizes.items(), key=lambda kv: kv[1], reverse=True))


def sidecar_bucket_sizes(sidecars: Dict[str, Any]) -> Dict[str, int]:
    """Serialized byte cost per characterization bucket, largest first.

    Reported on every push so the space consumers are visible from the job
    record rather than inferred.
    """
    per_bucket: Dict[str, int] = {}
    for name, sc in sidecars.items():
        bucket = plot_bucket(sc.get('plot_type', 'unknown'))
        n = len(json.dumps(sc, ensure_ascii=False,
                           separators=(',', ':')).encode())
        per_bucket[bucket] = per_bucket.get(bucket, 0) + n
    return dict(sorted(per_bucket.items(), key=lambda kv: kv[1], reverse=True))


def select_sidecars(sidecars: Dict[str, Any], budget_bytes: int,
                    exclude_buckets: Optional[Set[str]] = None
                    ) -> Tuple[Dict[str, Any], List[str]]:
    """Choose which sidecars to keep within a raw-byte budget.

    Two filters, applied in order:

    1. Bucket exclusion — whole characterizations dropped by configuration,
       for cases where their plots are known to be large and not worth storing.
    2. Size-ranked retention — of what remains, the smallest are kept first
       until the budget is exhausted. Keeping several light analyses is more
       useful than one heavy one, and it degrades smoothly instead of the
       all-or-nothing choice of dropping every sidecar.

    Returns (kept, dropped_names).
    """
    exclude = exclude_buckets or set()
    dropped: List[str] = []
    candidates: Dict[str, Any] = {}

    for name, sc in sidecars.items():
        if plot_bucket(sc.get('plot_type', 'unknown')) in exclude:
            dropped.append(name)
        else:
            candidates[name] = sc

    # Smallest first, so a budget buys the most plots.
    ordered = sorted(sidecar_sizes(candidates).items(), key=lambda kv: kv[1])
    kept: Dict[str, Any] = {}
    used = 0
    for name, size in ordered:
        if used + size <= budget_bytes:
            kept[name] = candidates[name]
            used += size
        else:
            dropped.append(name)
    return kept, dropped

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
        # 'o' is the legacy decimal marker; '.' passes through untouched.
        return float(s.replace('o', '.'))
    except (ValueError, TypeError):
        return None


# Step must be a standalone underscore-delimited token (b14, c10, a2).
#
# The permissive form used elsewhere — ([A-Za-z])(\d+) with alpha lookarounds —
# matches 'H2' inside a gas spec when the filename has no real step, silently
# producing step='h2'. Anchoring to token boundaries removes that: every gas
# token starts with a digit, so it can never satisfy this pattern.
# Two filename conventions are in active use and both must parse:
#
#   0o4H2 / 80C / 100kPa            'o' stands in for a decimal point
#   0.4WH2 / 80.3Cell / 0.2WA       literal decimal, W/D wet-dry marker
#
# A number is therefore digits with an optional fractional part introduced by
# either '.' or 'o'.
_NUM = r'(\d+(?:[.o]\d+)?)'

# Step is an underscore-delimited token: one letter, digits, and an optional
# trailing letter. That last part matters — b17b, b21b and b45b are real steps,
# and without it they parse as no step at all, which collapses distinct
# measurements into one index unit.
#
# Deliberately not matched: multi-letter prefixes such as Ramp0 or shutdown1.
# Allowing [A-Za-z]+ there would also match sample-name fragments like Gen2.
# Those files carry no key_values, so the units they produce hold nothing the
# index needs to tell apart.
_STEP_RE = re.compile(r'(?:^|_)([A-Za-z]\d+[A-Za-z]?)(?=_|$)')

# A gas token may be followed by a meaningless 'B<n>' suffix (0.05WN2B3). It
# carries no information but must not block the match, so the boundary check
# admits it alongside end-of-string and ordinary delimiters.
_GAS_END = r'(?=$|[^A-Za-z0-9]|B\d)'

# Gas flows: number, optional wet/dry marker, species. 'Air' is tried before
# the bare 'A' abbreviation so 0.4WAir does not match as air with a stray 'ir'.
# The W/D marker is recognised but not recorded — it would add bytes to every
# gas-bearing unit, and the units it would disambiguate carry no key_values.
_GAS_PATTERNS = (
    ('H2_slpm', re.compile(_NUM + r'[WD]?H2' + _GAS_END)),
    ('N2_slpm', re.compile(_NUM + r'[WD]?N2' + _GAS_END)),
    ('O2_slpm', re.compile(_NUM + r'[WD]?O2' + _GAS_END)),
    ('Air_slpm', re.compile(_NUM + r'[WD]?(?:Air|A)' + _GAS_END)),
)


# Only these are treated as file extensions. os.path.splitext splits on the
# last dot, which in this domain is usually a decimal point: it turns
# '..._0.2WH2-0.2WA' into a stem ending '-0' and an "extension" of '.2WA',
# silently discarding a real gas flow. Matching known suffixes instead leaves
# the decimals alone.
_EXT_RE = re.compile(
    r'\.(png|jpg|jpeg|svg|pdf|json|csv|txt|tsv|fcd|xlsx|xls)$', re.IGNORECASE)


# Sample names carry the experiment date as a YYMMDD prefix. Anchored at the
# start so an embedded second date cannot win — 260511_Gen2-260506 is the 11th
# of May, not the 6th. The trailing boundary stops a longer digit run from
# being read as a date.
_RUN_DATE_RE = re.compile(r'^(\d{2})(\d{2})(\d{2})(?=[_\-]|$)')


def parse_run_date(sample_name: Optional[str]) -> Optional[str]:
    """Experiment date from a sample-name prefix, as an ISO date string.

    Returns None when there is no prefix or it is not a real calendar date, so
    absence is meaningful: the consumer knows the date was not recoverable and
    can fall back to the analysis timestamp itself rather than being handed an
    analysis date dressed up as an experiment date.

    Why this matters: the index timestamp records when a job was *processed*.
    All six live entries are stamped within two days of each other because they
    were analysed in a batch, while the cells were tested across two months.
    Trending against that axis plots queue order.
    """
    if not sample_name:
        return None
    m = _RUN_DATE_RE.match(str(sample_name))
    if not m:
        return None
    yy, mm, dd = (int(g) for g in m.groups())
    try:
        return datetime.date(2000 + yy, mm, dd).isoformat()
    except ValueError:
        return None          # e.g. month 34 — a number, but not a date


# Test stand, derived from the data format rather than plumbed through every
# analysis script. This mirrors the auto-detection in fuelcell_analysis exactly:
# an .fcd anywhere means Scribner, otherwise delimited text means FCTS. It
# reflects the file format, so a run whose stand parameter was overridden by
# hand could disagree — in practice the format and the stand go together.
_SCRIBNER_EXT = {'.fcd'}
_FCTS_EXT = {'.csv', '.txt', '.tsv'}


def parse_stand(input_files: Optional[List[str]]) -> Optional[str]:
    """'Scribner', 'FCTS', or None when the files say nothing useful."""
    exts = {os.path.splitext(str(f))[1].lower() for f in (input_files or [])}
    exts = {e for e in exts if e}
    if not exts:
        return None
    if exts & _SCRIBNER_EXT:
        return 'Scribner'
    if exts <= _FCTS_EXT:
        return 'FCTS'
    return None


def _strip_extension(basename: str) -> str:
    return _EXT_RE.sub('', basename)


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
    name = _strip_extension(os.path.basename(filename))
    cond: Dict[str, Any] = {}

    m = _STEP_RE.search(name)
    if m:
        cond['step'] = m.group(1).lower()

    # '80C' and '80.3Cell' are the same field in two conventions.
    m = re.search(_NUM + r'C(?:ell)?(?![A-Za-z])', name)
    if m and (v := _num(m.group(1))) is not None:
        cond['T_C'] = v

    m = re.search(_NUM + r'RH', name, re.IGNORECASE)
    if m and (v := _num(m.group(1))) is not None:
        cond['RH_pct'] = v

    m = re.search(_NUM + r'\s*(kPa|barg|psi|bar)', name, re.IGNORECASE)
    if m and (v := _num(m.group(1))) is not None:
        cond['P_value'] = v
        cond['P_unit'] = m.group(2)

    for key, pat in _GAS_PATTERNS:
        m = pat.search(name)
        if m and (v := _num(m.group(1))) is not None:
            cond[key] = v

    # Voltage setpoint: digits + V, not adjacent to letters (excludes 'mV', 'kV')
    m = re.search(r'(?<![A-Za-z])' + _NUM + r'V(?![A-Za-z])', name)
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
        # Current density at fixed operating voltages. Sparse by design — a key
        # is absent when the curve did not reach that voltage, never estimated.
        ('j @ 0.7 V', ['j_at_0.7V', 'j @ 0.7 V']),
        ('j @ 0.65 V', ['j_at_0.65V', 'j @ 0.65 V']),
        ('j @ 0.6 V', ['j_at_0.6V', 'j @ 0.6 V']),
        ('j @ 0.5 V', ['j_at_0.5V', 'j @ 0.5 V']),
        ('j @ 0.4 V', ['j_at_0.4V', 'j @ 0.4 V']),
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
    'j @ 0.7 V': 'A/cm²',
    'j @ 0.65 V': 'A/cm²',
    'j @ 0.6 V': 'A/cm²',
    'j @ 0.5 V': 'A/cm²',
    'j @ 0.4 V': 'A/cm²',
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


def _rows_for_unit(summary: Optional[Any], bucket: str,
                   plot_names: List[str]) -> List[Dict[str, Any]]:
    """Summary rows belonging to one analysis unit.

    A batch run produces one summary row per input file, and an index entry has
    one unit per (analysis, step, conditions). Without this matching, flattening
    every row together makes the first file's values stand in for all of them —
    so a five-polcurve Full Analysis would report the same OCV and j@V under
    every step. Rows are matched by label appearing in one of the unit's plot
    names, which is how the analysis scripts construct those names.
    """
    rows = summary if isinstance(summary, list) else ([summary] if summary else [])
    matched: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_bucket = row.get('Analysis')
        if row_bucket and row_bucket != bucket:
            continue
        label = str(row.get('Label') or row.get('label') or '')
        if label and any(label in pn for pn in plot_names):
            matched.append(row)
    if matched:
        return matched

    # No label matched. A single candidate row for this bucket is unambiguous —
    # a one-file run whose label simply does not appear in the plot name — so
    # use it. With several candidates there is no way to tell which belongs
    # here, and guessing would reintroduce the misattribution this exists to
    # prevent, so report nothing.
    candidates = [r for r in rows if isinstance(r, dict)
                  and (not r.get('Analysis') or r.get('Analysis') == bucket)]
    return candidates if len(candidates) == 1 else []


def _summary_lookup(summary: Optional[Any]) -> Dict[str, Any]:
    """Flatten summary rows into one key -> value map.

    Accepts a list of row dicts (the shape run_batch already builds) or a single
    dict. Earlier rows win, so this is only unambiguous once rows have been
    narrowed to a single analysis unit by _rows_for_unit.
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
        # Per-bucket accounting travels with the record so the space consumers
        # are visible without having to decompress and re-measure.
        record['sidecar_bytes_by_bucket'] = sidecar_bucket_sizes(sidecars)
    return record


# ─────────────────────────────────────────────────────────────────────
#  Index entry
# ─────────────────────────────────────────────────────────────────────

def _analysis_units(metrics: Dict[str, Dict[str, Any]],
                    summary: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Collapse plots into distinct (Analysis, step, Conditions) units.

    Several plots from one characterization at one setpoint — say a polcurve and
    its Tafel panel — become a single unit with their values merged.
    """
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    order: List[Tuple[str, str, str]] = []

    for bucket, plots in metrics.items():
        for plot_name, entry in plots.items():
            conditions = dict(entry.get('conditions') or {})
            step = conditions.pop('step', '')
            key = (bucket, step, json.dumps(conditions, sort_keys=True))
            if key not in grouped:
                grouped[key] = {'Analysis': bucket, 'step': step,
                                'Conditions': conditions, '_values': {},
                                '_plots': []}
                order.append(key)
            grouped[key]['_values'].update(entry.get('values') or {})
            grouped[key]['_plots'].append(plot_name)

    built: List[Dict[str, Any]] = []
    for key in order:
        g = grouped[key]
        unit: Dict[str, Any] = {
            'Analysis': g['Analysis'],
            'step': g['step'],
            'Conditions': g['Conditions'],
        }
        # Narrow the summary to this unit before flattening, so each step
        # reports its own file's values rather than the first file's.
        rows = _rows_for_unit(summary, g['Analysis'], g['_plots'])
        kv = build_key_values(g['Analysis'], g['_values'],
                              _summary_lookup(rows))
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
    built = [u for u in built
             if u['step'] or u['Conditions'] or u.get('key_values')
             or u['Analysis'] not in informative]

    return _trim_conditions(built)


def _trim_conditions(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop Conditions from units that promote no key_values.

    Those units exist to say "this run included an OCV analysis at step b18";
    the conditions behind them are already in the detail bin, per plot, which is
    what run detail reads. Carrying them in the index costs ~110 B each on the
    populated naming convention for something nothing queries.

    Safe only because steps now parse. Conditions previously did part of the
    work of telling units apart — b17b and b20b both failed to yield a step and
    were separated solely by their RH values differing. With steps parsing,
    (Analysis, step) is sufficient identity, so the conditions are redundant
    rather than load-bearing.

    Units that lose their Conditions are then re-merged on (Analysis, step),
    since two units differing only by a field that is no longer emitted would
    otherwise appear as indistinguishable duplicates.
    """
    out: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for u in units:
        if u.get('key_values'):
            out.append(u)
            continue
        key = (u['Analysis'], u['step'])
        if key in seen:
            continue                      # already represented
        slim = {'Analysis': u['Analysis'], 'step': u['step']}
        seen[key] = slim
        out.append(slim)
    return out


def build_index_entry(detail_record: Dict[str, Any], bin_id: str) -> Dict[str, Any]:
    """Build the lightweight index entry pointing at a detail bin.

    One entry per job. `Data` holds one element per analysis unit, so a Full
    Analysis run spanning several characterizations and setpoints is represented
    without collapsing them to a single arbitrary value.
    """
    entry: Dict[str, Any] = {
        'job_id': detail_record.get('job_id', ''),
        'sample_name': detail_record.get('sample_name', ''),
        'script': detail_record.get('script', ''),
        'timestamp': detail_record.get('timestamp', ''),
        'bin_id': bin_id,
        'Data': _analysis_units(detail_record.get('metrics', {}),
                                detail_record.get('summary')),
    }
    # Emitted only when recoverable. Consumers fall back to `timestamp` and can
    # tell the two apart, rather than an analysis date silently standing in for
    # an experiment date.
    run_date = parse_run_date(detail_record.get('sample_name'))
    if run_date:
        entry['run_date'] = run_date
    stand = parse_stand(detail_record.get('input_files'))
    if stand:
        entry['stand'] = stand
    return entry


def detail_bin_name(detail_record: Dict[str, Any], script_short: str = '') -> str:
    """Human-legible bin name for the JSONBin dashboard. Capped at 128 chars."""
    parts = [detail_record.get('sample_name') or 'run',
             script_short or detail_record.get('script', ''),
             detail_record.get('timestamp', '')]
    name = '-'.join(p for p in parts if p)
    return name[:128]


# ═══════════════════════════════════════════════════════════════════
#  Merging — sample-keyed overwrite
# ═══════════════════════════════════════════════════════════════════
#
# Pure functions. Nothing here talks to JSONBin; transport is phase 3.
#
# The rule, from OVERWRITE_SCOPE.md §1: replace at (analysis, step)
# granularity, preserve anything untouched. Both cases follow from it —
# crossover-then-Full-Analysis and Full-Analysis-then-crossover — because in
# each the incoming record declares which (analysis, step) pairs it covers, and
# only those are displaced.
#
# Steps are the identity. This is safe only because steps now parse reliably
# for both naming conventions; before that fix, b21b through b24b all yielded
# no step and would have merged into one another.


def _plot_step(entry: Dict[str, Any]) -> str:
    return str((entry.get('conditions') or {}).get('step') or '')


def touched_units(record: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """The (analysis, step) pairs a record covers.

    This is what an incoming record displaces when merged into an existing one.
    """
    out: Set[Tuple[str, str]] = set()
    for bucket, plots in (record.get('metrics') or {}).items():
        for entry in plots.values():
            out.add((bucket, _plot_step(entry)))
    return out


def _annotate_summary_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Tag each summary row with the (Analysis, step) it belongs to.

    Rows carry a Label but not always an Analysis — the Full Analysis
    orchestrator stamps one, the standalone scripts do not. The bucket is
    recovered the same way `_rows_for_unit` does it: by finding the plot whose
    name contains the row's label.
    """
    metrics = record.get('metrics') or {}
    rows = record.get('summary') or []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get('Label') or row.get('label') or '')
        bucket = row.get('Analysis') or ''
        step = ''
        for b, plots in metrics.items():
            for plot_name, entry in plots.items():
                if label and label in plot_name:
                    bucket = bucket or b
                    step = _plot_step(entry)
                    break
            if step:
                break
        if not bucket and len(metrics) == 1:
            bucket = next(iter(metrics))       # unambiguous single-bucket run
        if not step and label:
            step = str(parse_conditions(label).get('step') or '')
        out.append({'_bucket': bucket, '_step': step, 'row': row})
    return out


def merge_detail_record(existing: Optional[Dict[str, Any]],
                        incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a new run into a sample's accumulated detail record.

    Everything the incoming record covers is replaced; everything else is kept.
    Applied identically to metrics, summary and sidecars so the three cannot
    drift apart.

    `input_files` accumulates as a deduplicated union — worth watching, since a
    sample analysed repeatedly will grow it, and it competes with sidecars for
    the transport budget.
    """
    if not existing:
        merged = json.loads(json.dumps(incoming))
        merged['jobs'] = [{
            'job_id': incoming.get('job_id', ''),
            'timestamp': incoming.get('timestamp', ''),
            'buckets': sorted((incoming.get('metrics') or {}).keys()),
        }]
        return merged

    touched = touched_units(incoming)
    merged = json.loads(json.dumps(existing))

    # ── metrics ──
    old_metrics = merged.get('metrics') or {}
    kept: Dict[str, Dict[str, Any]] = {}
    for bucket, plots in old_metrics.items():
        survivors = {name: e for name, e in plots.items()
                     if (bucket, _plot_step(e)) not in touched}
        if survivors:
            kept[bucket] = survivors
    for bucket, plots in (incoming.get('metrics') or {}).items():
        kept.setdefault(bucket, {}).update(json.loads(json.dumps(plots)))
    merged['metrics'] = kept

    # ── summary ──
    surviving_rows = [a['row'] for a in _annotate_summary_rows(existing)
                      if (a['_bucket'], a['_step']) not in touched]
    merged['summary'] = surviving_rows + list(incoming.get('summary') or [])
    if not merged['summary']:
        merged.pop('summary', None)

    # ── sidecars ──
    old_sc = decode_sidecars(existing)
    new_sc = decode_sidecars(incoming)
    sc_kept: Dict[str, Any] = {}
    for name, sc in old_sc.items():
        bucket = plot_bucket(sc.get('plot_type', 'unknown'))
        step = str(parse_conditions(name).get('step') or '')
        if (bucket, step) not in touched:
            sc_kept[name] = sc
    sc_kept.update(new_sc)
    attach_sidecars(merged, sc_kept)
    # A size-driven omission on a previous write says nothing about the merged
    # result; transport re-evaluates it.
    merged.pop('sidecars_omitted', None)

    # ── top-level conditions, recomputed over the merged set ──
    sets = []
    for plots in merged['metrics'].values():
        for e in plots.values():
            c = e.get('conditions') or {}
            if c:
                sets.append(json.dumps(c, sort_keys=True))
    merged['conditions'] = (json.loads(sets[0])
                            if sets and len(set(sets)) == 1 else None)

    # ── provenance ──
    merged['schema'] = SCHEMA_VERSION
    merged['job_id'] = incoming.get('job_id', '')     # last writer
    merged['timestamp'] = incoming.get('timestamp', '')
    merged['script'] = incoming.get('script', merged.get('script', ''))
    merged['sample_name'] = (incoming.get('sample_name')
                             or merged.get('sample_name', ''))
    files = list(merged.get('input_files') or [])
    for f in (incoming.get('input_files') or []):
        if f not in files:
            files.append(f)
    merged['input_files'] = files
    # An existing record always represents at least one job, but records
    # written before this field existed — which is all of them — carry no
    # `jobs` list. Seed it from the record's own identity so merging does not
    # erase the provenance of whatever was already there.
    jobs = list(merged.get('jobs') or [])
    if not jobs:
        jobs = [{'job_id': existing.get('job_id', ''),
                 'timestamp': existing.get('timestamp', ''),
                 'buckets': sorted((existing.get('metrics') or {}).keys())}]
    jobs.append({'job_id': incoming.get('job_id', ''),
                 'timestamp': incoming.get('timestamp', ''),
                 'buckets': sorted((incoming.get('metrics') or {}).keys())})
    merged['jobs'] = jobs
    return merged


def merge_index_entry(existing: Optional[Dict[str, Any]],
                      incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a new run's index entry into a sample's accumulated entry.

    Units are replaced on (Analysis, step) — the same rule the detail record
    uses, so the index cannot disagree with the bin it points at.

    `job_id` becomes the last writer rather than the only one, with `n_jobs`
    alongside (fork B). `script` likewise reflects the last writer: for a
    sample-keyed entry it is no longer a property of the sample, and a run that
    began as a Full Analysis may later be topped up by a standalone crossover.
    """
    if not existing:
        out = json.loads(json.dumps(incoming))
        out['n_jobs'] = 1
        return out

    merged = json.loads(json.dumps(existing))
    incoming_units = {(u.get('Analysis', ''), u.get('step', ''))
                      for u in (incoming.get('Data') or [])}
    survivors = [u for u in (merged.get('Data') or [])
                 if (u.get('Analysis', ''), u.get('step', '')) not in incoming_units]
    merged['Data'] = survivors + list(incoming.get('Data') or [])

    merged['job_id'] = incoming.get('job_id', '')
    merged['n_jobs'] = int(merged.get('n_jobs') or 1) + 1
    merged['timestamp'] = incoming.get('timestamp', '')
    merged['script'] = incoming.get('script', merged.get('script', ''))
    merged['sample_name'] = (incoming.get('sample_name')
                             or merged.get('sample_name', ''))
    merged['bin_id'] = incoming.get('bin_id') or merged.get('bin_id', '')
    run_date = incoming.get('run_date') or merged.get('run_date')
    if run_date:
        merged['run_date'] = run_date
    else:
        merged.pop('run_date', None)
    stand = incoming.get('stand') or merged.get('stand')
    if stand:
        merged['stand'] = stand
    else:
        merged.pop('stand', None)
    return merged