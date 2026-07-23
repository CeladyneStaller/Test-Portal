#!/usr/bin/env python3
"""
JSONBin diagnostic.

Read-only by default: reports what is actually in the index bin, and whether
each entry's detail bin can be fetched.

    python3 scripts/helpers/diagnose_jsonbin.py

With --write-test it also performs a full index round trip — appends a probe
entry, verifies it landed, then removes it and verifies the removal. That is
the only way to prove the append path end to end, and it is opt-in because it
writes to the live index.

    python3 scripts/helpers/diagnose_jsonbin.py --write-test

On Railway, run it in the same environment as the app so it sees the same
variables:  railway run python3 scripts/helpers/diagnose_jsonbin.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OK, BAD, INFO = "  ok  ", " FAIL ", "      "
WRITE_TEST = '--write-test' in sys.argv


def head(t):
    print(f"\n{t}\n" + "-" * len(t))


def die(msg, hint=''):
    print(f"{BAD}{msg}")
    if hint:
        print(f"{INFO}{hint}")
    sys.exit(1)


# ── 1. Environment ────────────────────────────────────────────────
head("1. Environment")

env = {k: os.environ.get(k) for k in
       ('JSONBIN_API_KEY', 'JSONBIN_COLLECTION_ID', 'JSONBIN_INDEX_BIN_ID')}
for k, v in env.items():
    if v:
        print(f"{OK}{k} = {(v[:6] + '…' + v[-4:]) if k.endswith('KEY') else v}")
    else:
        print(f"{BAD}{k} is not set")
missing = [k for k, v in env.items() if not v]
if missing:
    die(f"missing: {', '.join(missing)}",
        "Set these on the Railway service and redeploy.")

try:
    from scripts.helpers import jsonbin, record
except Exception as e:
    die(f"cannot import helpers: {e}")

if not hasattr(jsonbin, 'create_detail_bin'):
    die("jsonbin.py is the old single-bin version",
        "Deploy the index+detail version to scripts/helpers/jsonbin.py")
print(f"{OK}helpers import, index+detail version")


# ── 2. What is actually in the index ──────────────────────────────
head("2. Index contents")

try:
    index = jsonbin.fetch_index()
except Exception as e:
    msg = str(e)
    hint = ''
    if '401' in msg or '403' in msg:
        hint = "Key rejected. Confirm JSONBIN_API_KEY is the Master Key."
    elif '404' in msg:
        hint = (f"Bin {env['JSONBIN_INDEX_BIN_ID']} not found — wrong ID, or it "
                f"belongs to a different JSONBin account.")
    die(f"cannot read the index: {e}", hint)

runs = index.get('runs', [])
print(f"{OK}index readable — schema {index.get('schema')}, "
      f"{len(runs)} run(s)")

if not runs:
    print(f"{INFO}The index is EMPTY. If analyses reported pushed=true, they")
    print(f"{INFO}wrote to a different bin than {env['JSONBIN_INDEX_BIN_ID']}.")
else:
    print(f"\n{'#':>3}  {'timestamp':20} {'sample':26} {'script':30} bin_id")
    print("     " + "-" * 92)
    for i, r in enumerate(runs[-15:], start=max(1, len(runs) - 14)):
        print(f"{i:>3}  {str(r.get('timestamp',''))[:20]:20} "
              f"{str(r.get('sample_name',''))[:26]:26} "
              f"{str(r.get('script',''))[:30]:30} {r.get('bin_id','—')}")
    if len(runs) > 15:
        print(f"{INFO}(showing the most recent 15 of {len(runs)})")

    n_kv = sum(1 for r in runs
               for d in r.get('Data', []) if d.get('key_values'))
    print(f"\n{INFO}{n_kv} analysis unit(s) across the index carry key_values")


# ── 3. Are the detail bins reachable? ─────────────────────────────
head("3. Detail bin reachability")

if not runs:
    print(f"{INFO}nothing to check")
else:
    checked = runs[-5:]
    for r in checked:
        bid = r.get('bin_id')
        if not bid:
            print(f"{BAD}{r.get('job_id')} has no bin_id")
            continue
        try:
            payload = jsonbin._request(
                f"{jsonbin._JSONBIN_BASE}/{bid}/latest", method='GET')
            rec = payload.get('record') or {}
            n_plots = sum(len(v) for v in (rec.get('metrics') or {}).values())
            has_sc = 'sidecars' in rec
            print(f"{OK}{bid}  schema={rec.get('schema')}  plots={n_plots}  "
                  f"sidecars={'yes' if has_sc else 'no'}")
        except Exception as e:
            print(f"{BAD}{bid} unreachable: {str(e)[:90]}")
    if len(runs) > 5:
        print(f"{INFO}(checked the most recent 5)")


# ── 4. Index round trip ───────────────────────────────────────────
head("4. Index append round trip" + ("" if WRITE_TEST else " (skipped)"))

if not WRITE_TEST:
    print(f"{INFO}Read-only by default. Re-run with --write-test to append a")
    print(f"{INFO}probe entry, confirm it lands, and remove it again. That is")
    print(f"{INFO}the only check that exercises the append path end to end.")
else:
    stamp = (datetime.now(timezone.utc)
             .isoformat(timespec='seconds').replace('+00:00', 'Z'))
    probe_id = f'diagnostic-probe-{stamp}'
    probe = {'job_id': probe_id, 'sample_name': '__diagnostic__',
             'script': 'Diagnostic Probe', 'timestamp': stamp,
             'bin_id': 'none', 'Data': []}

    before = len(runs)
    print(f"{INFO}index has {before} run(s) before the test")

    # Append.
    try:
        after = jsonbin.append_index_entry(probe)
        print(f"{OK}append reported {after} run(s)")
    except Exception as e:
        die(f"append failed: {e}",
            "This is the write path the analyses use. The error above is the "
            "real cause of missing index entries.")

    # Verify by re-reading, not by trusting the return value.
    try:
        check = jsonbin.fetch_index()
    except Exception as e:
        die(f"re-read after append failed: {e}",
            f"Probe {probe_id} may still be in the index.")

    ids = [r.get('job_id') for r in check.get('runs', [])]
    if probe_id in ids:
        print(f"{OK}probe found in the index on re-read — the append PERSISTED")
    else:
        die("probe is NOT in the index on re-read",
            "The PUT reported success but did not persist. Check that the bin "
            "ID is a bin you own and is not read-only, and that no proxy is "
            "caching the GET.")

    if len(check.get('runs', [])) != before + 1:
        print(f"{BAD}run count went {before} → {len(check.get('runs', []))}, "
              f"expected {before + 1}")
        print(f"{INFO}A concurrent analysis may have written during the test.")

    # Remove the probe and confirm.
    try:
        cleaned = {k: v for k, v in check.items() if k != 'runs'}
        cleaned['runs'] = [r for r in check['runs']
                           if r.get('job_id') != probe_id]
        jsonbin._write_index(cleaned)
        final = jsonbin.fetch_index()
        if any(r.get('job_id') == probe_id for r in final.get('runs', [])):
            print(f"{BAD}probe removal did not persist — remove {probe_id} "
                  f"by hand")
        else:
            print(f"{OK}probe removed; index restored to {len(final.get('runs', []))} run(s)")
    except Exception as e:
        print(f"{BAD}cleanup failed: {e}")
        print(f"{INFO}Remove the entry with job_id {probe_id} manually.")


# ── Summary ───────────────────────────────────────────────────────
head("Result")
if runs:
    print(f"The index bin {env['JSONBIN_INDEX_BIN_ID']} contains {len(runs)} run(s).")
    print("If a run you expected is missing, check that job's metrics_push")
    print("object via GET /api/jobs — 'pushed' and 'reason' say what happened.")
else:
    print("The index bin is empty.")
if not WRITE_TEST:
    print("\nRe-run with --write-test to exercise the append path itself.")