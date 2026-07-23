"""
Integration sweep: every fuel-cell script from raw data through to a built
index entry. HTTP is mocked; nothing touches JSONBin.

Run:  python3 scripts/helpers/sweep_integration.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.helpers import jsonbin  # noqa: E402

_passed = 0
_failed = 0
ROOT = tempfile.mkdtemp(prefix='sweep_')


def check(name, cond, detail=''):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"    FAIL {name} {detail}")


# ─────────────────────────────────────────────────────────────────────
#  Synthetic instrument files
# ─────────────────────────────────────────────────────────────────────

def polcurve_fcd(path, hfr_base=45.0, ci=False):
    """Wide Scribner layout: named columns at the indices the preset expects."""
    N = 30
    cols = [f'c{i}' for i in range(N)]
    cols[0] = 'Time (Sec)'; cols[1] = 'I (A)'; cols[5] = 'E_Stack (V)'
    cols[13] = 'T_cell (C)'; cols[14] = 'T_anode_dp (C)'
    cols[15] = 'H2_flow (slpm)'; cols[17] = 'T_cathode_dp (C)'
    cols[18] = 'Air_flow (slpm)'; cols[20] = 'HFR (mOhm)'
    cols[28] = 'Ctrl_Mode'
    if ci:
        cols[21] = 'E_iR_Stack (mOhm)'
    rows, t = [], 0.0
    for cyc in range(3):
        sp = (np.arange(0.90, 0.39, -0.05) if cyc % 2 == 0
              else np.arange(0.40, 0.91, 0.05))
        for v in sp:
            j = max(0.0, (0.92 - v) * 3.2)
            hfr = hfr_base + j * 2.0
            for _ in range(12):
                r = ['0'] * N
                r[0] = f'{t:.2f}'
                r[1] = f'{j * 5.0 + np.random.normal(0, .01):.5f}'
                r[5] = f'{v + np.random.normal(0, .002):.5f}'
                r[13] = '80.0'; r[14] = '80.0'; r[15] = '0.200'
                r[17] = '80.0'; r[18] = '0.200'
                r[20] = f'{hfr + np.random.normal(0, .5):.4f}'
                if ci:
                    r[21] = f'{hfr * 1.1 + 2:.4f}'
                r[28] = '1'
                rows.append(r); t += 1.0
    _write(path, cols, rows)


def cleaning_fcd(path):
    cols = ['Time (Sec)', 'I (A)', 'Pt2', 'Pt3', 'E_Stack (V)',
            'HFR (mOhm)', 'x', 'Ctrl_Mode']
    Vs, Js = [], []
    for cyc in range(15):
        Va = np.linspace(0.05, 1.0, 100)
        Vc = np.linspace(1.0, 0.05, 100)[1:]
        ha = 25 * np.exp(-cyc * .15) + 6
        dl = 4 * np.exp(-cyc * .12) + 1
        Vs += [Va, Vc]
        Js += [ha * np.exp(-((Va - .2) / .08) ** 2) + dl
               + np.random.normal(0, .15, 100),
               (-23 * np.exp(-cyc * .15) - 5)
               * np.exp(-((Vc - .2) / .08) ** 2) - dl
               + np.random.normal(0, .15, 99)]
    V = np.concatenate(Vs); J = np.concatenate(Js) * 0.001 * 5.0
    rows = [[f'{i * 0.001:.4f}', f'{jj:.5f}', '0', '0', f'{vv:.5f}',
             '0', '0', '2'] for i, (vv, jj) in enumerate(zip(V, J))]
    _write(path, cols, rows)


def activation_fcd(path):
    cols = ['Time (Sec)', 'I (A)', 'E_Stack (V)', 'Ctrl_Mode']
    n = 3600
    v = np.zeros(n)
    for i in range(5):
        v[i * 720:(i + 1) * 720] = 0.85 - i * 0.04
    I = np.clip(5 * (0.95 - v - 0.1), 0, None) + np.random.normal(0, .01, n)
    rows = [[str(i), f'{ii:.5f}', f'{vv:.5f}', '1']
            for i, (vv, ii) in enumerate(zip(v, I))]
    _write(path, cols, rows)


def _write(path, cols, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        f.write('# FCD\n# generated\n')
        f.write('\t'.join(cols) + '\n')
        f.write('End Comments\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')


# ─────────────────────────────────────────────────────────────────────
#  Mock transport
# ─────────────────────────────────────────────────────────────────────

class Capture:
    def __init__(self):
        self.calls = []

    def __call__(self, url, method='GET', body=None, extra_headers=None):
        self.calls.append((method, body, extra_headers or {}))
        if method == 'POST':
            return {'metadata': {'id': 'BIN'}}
        if method == 'GET':
            return {'record': {'schema': 2, 'runs': []}}
        return {'record': body}


# ─────────────────────────────────────────────────────────────────────
#  Cases
# ─────────────────────────────────────────────────────────────────────

CASES = [
    ('FC Polarization Curve', 'polcurve_analysis', 'PolCurve',
     lambda d: polcurve_fcd(f'{d}/260126_FCS6_b14_IV_80C_100RH_0o2H2_0o2Air_0kPa.fcd'),
     'polcurve', ['OCV', 'V @ 1 A/cm²']),
    ('FC Polarization Curve (Downswing)', 'polcurve_analysis_down', 'PolCurveDown',
     lambda d: polcurve_fcd(f'{d}/260126_FCS6_b14_IV_80C_100RH_0o2H2_0o2Air_0kPa.fcd'),
     'polcurve', ['OCV', 'V @ 1 A/cm²']),
    ('FC Polarization Curve (HFR Compare)', 'polcurve_analysis_hfr_compare',
     'PolCurveHFRcmp',
     lambda d: polcurve_fcd(f'{d}/260126_FCS6_b14_IV_80C_100RH_0o2H2_0o2Air_0kPa.fcd',
                            ci=True),
     'polcurve', ['OCV', 'V @ 1 A/cm²']),
    ('FC Electrode Cleaning', 'electrode_cleaning_analysis', 'Cleaning',
     lambda d: cleaning_fcd(f'{d}/260126_FCS6_a2_CV-500mVs.fcd'),
     'cleaning', []),
    ('FC Activation', 'activation_analysis', 'Activation',
     lambda d: activation_fcd(f'{d}/260126_FCS6_b2_activation.fcd'),
     'activation', []),
]

os.environ.update(JSONBIN_API_KEY='k', JSONBIN_COLLECTION_ID='C',
                  JSONBIN_INDEX_BIN_ID='I')

print(f"{'script':38} {'sum':>4} {'idx B':>6} {'detail B':>9}  key_values")
print('-' * 96)

for script_name, module, short, make_input, bucket, want_kv in CASES:
    work = Path(ROOT) / module
    (work / 'input').mkdir(parents=True, exist_ok=True)
    make_input(str(work / 'input'))

    mod = __import__(f'scripts.{module}', fromlist=['run'])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = mod.run(str(work / 'input'), str(work / 'output'),
                          {'geo_area': '5.0', 'image_format': 'png'})
    except Exception as e:
        check(f'{script_name} runs', False, f'({e})')
        continue

    check(f'{script_name} succeeded', res.get('status') == 'success')
    summary = res.get('summary')
    check(f'{script_name} emits summary', isinstance(summary, list) and summary)

    cap = Capture()
    jsonbin._request = cap
    push = jsonbin.push_job_metrics(
        job_id='j', sample_name='260126_FCS6', script=script_name,
        output_dir=work / 'output', input_files=['in.fcd'],
        script_short=short, summary=summary)

    check(f'{script_name} pushed', push['pushed'], f"({push['reason']})")
    if not push['pushed']:
        continue

    # The index is read first for the sample lookup, so locate calls by method
    # rather than position.
    detail = next(b for m_, b, _ in cap.calls if m_ == 'POST')
    entry = next(b for m_, b, _ in cap.calls if m_ == 'PUT')['runs'][0]
    idx_b = len(json.dumps(entry, ensure_ascii=False).encode())
    det_b = len(json.dumps(detail, ensure_ascii=False).encode())

    check(f'{script_name} schema 2', detail.get('schema') == 2)
    # Sidecars are stored unless the bucket is excluded by configuration.
    # Cleaning is excluded by default — its CV plots dominate the wire budget.
    excluded = bucket in jsonbin.SIDECAR_EXCLUDE_BUCKETS
    if excluded:
        check(f'{script_name} sidecars deliberately excluded',
              'sidecars' not in detail)
        check(f'{script_name} exclusion still reports bucket bytes',
              bucket in push.get('sidecar_bytes_by_bucket', {}))
    else:
        check(f'{script_name} sidecars embedded', 'sidecars' in detail)
    check(f'{script_name} summary embedded', 'summary' in detail)
    check(f'{script_name} bucket {bucket}', bucket in detail['metrics'],
          f"(got {list(detail['metrics'])})")
    check(f'{script_name} entry has Data', len(entry['Data']) >= 1)
    post_hdrs = next(h for m_, _, h in cap.calls if m_ == 'POST')
    check(f'{script_name} bin name', post_hdrs.get('X-Bin-Name', '')
          .startswith(f'260126_FCS6-{short}-'))
    check(f'{script_name} detail under 10MB', det_b < 10_000_000)

    kv = {}
    for d in entry['Data']:
        kv.update(d.get('key_values') or {})
    for k in want_kv:
        check(f'{script_name} key_values has {k}', k in kv,
              f"(got {sorted(kv)})")

    # Polcurve variants report current density at fixed voltages. The synthetic
    # curves span 0.90-0.40 V, so all five targets should be present, and every
    # reported value must be a real number rather than a placeholder.
    if bucket == 'polcurve':
        j_keys = [k for k in kv if k.startswith('j @ ')]
        check(f'{script_name} reports j@V', len(j_keys) == 5,
              f'(got {sorted(j_keys)})')
        check(f'{script_name} j@V values numeric',
              all(isinstance(kv[k], (int, float)) for k in j_keys))
        # Monotonic: lower voltage must draw more current.
        ordered = ['j @ 0.7 V', 'j @ 0.65 V', 'j @ 0.6 V', 'j @ 0.5 V', 'j @ 0.4 V']
        present = [kv[k] for k in ordered if k in kv]
        check(f'{script_name} j@V rises as V falls',
              all(a < b for a, b in zip(present, present[1:])),
              f'(got {present})')

    print(f"{script_name:38} {len(summary):>4} {idx_b:>6} {det_b:>9,}  "
          f"{', '.join(f'{k}={v}' for k, v in kv.items()) or '—'}")

shutil.rmtree(ROOT, ignore_errors=True)
print('-' * 96)
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)