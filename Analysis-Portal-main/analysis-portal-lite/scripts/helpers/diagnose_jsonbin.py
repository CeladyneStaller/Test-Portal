#!/usr/bin/env python3
"""
JSONBin write diagnostic.

Tests each layer in order and stops at the first failure, reporting exactly
what broke. Safe to run against production: it creates one throwaway detail bin
and reads the index, but never modifies the index.

Run from the app root (the directory containing scripts/):

    python3 scripts/helpers/diagnose_jsonbin.py

On Railway:  railway run python3 scripts/helpers/diagnose_jsonbin.py
or open a shell on the service and run it there — it must run in the same
environment as the app so it sees the same variables.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OK = "  ok  "
BAD = " FAIL "
INFO = "      "


def head(t):
    print(f"\n{t}\n" + "-" * len(t))


def die(msg, hint=''):
    print(f"{BAD}{msg}")
    if hint:
        print(f"{INFO}{hint}")
    print("\nStopped at first failure.")
    sys.exit(1)


# ── 1. Environment ────────────────────────────────────────────────
head("1. Environment variables")

env = {k: os.environ.get(k) for k in
       ('JSONBIN_API_KEY', 'JSONBIN_COLLECTION_ID', 'JSONBIN_INDEX_BIN_ID')}
for k, v in env.items():
    if v:
        shown = (v[:6] + '…' + v[-4:]) if k.endswith('KEY') else v
        print(f"{OK}{k} = {shown}")
    else:
        print(f"{BAD}{k} is not set")

missing = [k for k, v in env.items() if not v]
if missing:
    die(f"missing: {', '.join(missing)}",
        "Set these on the Railway service, then redeploy. Variables set on the "
        "project but not applied to the service will not appear here.")

legacy = os.environ.get('JSONBIN_METRICS_BIN_ID')
if legacy:
    print(f"{INFO}JSONBIN_METRICS_BIN_ID is still set ({legacy}) — harmless, "
          f"no longer read.")


# ── 2. Imports ────────────────────────────────────────────────────
head("2. Module imports")

try:
    from scripts.helpers import record
    print(f"{OK}scripts.helpers.record")
except Exception as e:
    die(f"cannot import record.py: {e}",
        "record.py must be at scripts/helpers/record.py")

try:
    from scripts.helpers import jsonbin
    print(f"{OK}scripts.helpers.jsonbin")
except Exception as e:
    die(f"cannot import jsonbin.py: {e}")

if not hasattr(jsonbin, 'create_detail_bin'):
    die("jsonbin.py has no create_detail_bin()",
        "The old single-bin version is still deployed. Replace "
        "scripts/helpers/jsonbin.py with the index+detail version.")
print(f"{OK}jsonbin.py is the index+detail version")

try:
    from scripts import SCRIPT_SHORT
    print(f"{OK}scripts/__init__.py ({len(SCRIPT_SHORT)} scripts registered)")
except Exception as e:
    print(f"{BAD}cannot import scripts/__init__.py: {e}")
    print(f"{INFO}Not fatal for the push itself — bin names lose their script "
          f"short label — but it points at a broken analysis module.")

print(f"{OK}is_configured() = {jsonbin.is_configured()}")


# ── 3. Reach JSONBin, read the index ──────────────────────────────
head("3. Read the index bin")

try:
    index = jsonbin.fetch_index()
except Exception as e:
    msg = str(e)
    hint = ''
    if '401' in msg or '403' in msg:
        hint = ("Authentication rejected. Check JSONBIN_API_KEY is the Master "
                "Key, not an Access Key, and that it has not been rotated.")
    elif '404' in msg:
        hint = (f"Bin {env['JSONBIN_INDEX_BIN_ID']} not found. Check the ID, "
                f"and that it belongs to this JSONBin account.")
    elif '1010' in msg:
        hint = ("Cloudflare bot block. The browser-like User-Agent should "
                "prevent this — confirm the deployed jsonbin.py sets it.")
    die(f"cannot read the index: {e}", hint)

runs = index.get('runs', [])
print(f"{OK}index readable — schema {index.get('schema')}, {len(runs)} run(s)")
if runs:
    last = runs[-1]
    print(f"{INFO}most recent: {last.get('timestamp')} "
          f"{last.get('sample_name')} / {last.get('script')} "
          f"-> bin {last.get('bin_id')}")
else:
    print(f"{INFO}index is empty — nothing has been written yet")


# ── 4. Create a throwaway detail bin ──────────────────────────────
head("4. Create a detail bin in the collection")

probe = {'schema': 2, 'job_id': 'diagnostic-probe',
         'note': 'connectivity check — safe to delete'}
try:
    bin_id = jsonbin.create_detail_bin(probe, name='diagnostic-probe')
except Exception as e:
    msg = str(e)
    hint = ''
    if '401' in msg or '403' in msg:
        hint = "Key rejected for writes. Confirm it is the Master Key."
    elif '404' in msg or '400' in msg:
        hint = (f"Collection {env['JSONBIN_COLLECTION_ID']} may be wrong or "
                f"may belong to another account.")
    die(f"cannot create a bin: {e}", hint)

print(f"{OK}created bin {bin_id}")
print(f"{INFO}delete it from the JSONBin dashboard when you are done")


# ── 5. Record assembly against a real job ─────────────────────────
head("5. Record assembly from an existing job output")

jobs_dir = Path(os.environ.get('JOBS_DIR', 'jobs'))
candidates = []
if jobs_dir.exists():
    for d in sorted(jobs_dir.iterdir(), reverse=True):
        out = d / 'output'
        if out.is_dir() and any(out.rglob('_plot_data/*.json')):
            candidates.append(out)
        if len(candidates) >= 1:
            break

if not candidates:
    print(f"{INFO}no job output with sidecars found under {jobs_dir}/")
    print(f"{INFO}skipping — run an analysis first, then re-run this check")
else:
    out = candidates[0]
    print(f"{INFO}using {out}")
    try:
        rec = record.build_detail_record(
            job_id='diagnostic', sample_name='diagnostic',
            script='FC Polarization Curve', timestamp='2026-01-01T00:00:00Z',
            input_files=[], output_dir=out)
    except Exception as e:
        die(f"record assembly failed: {e}")

    n_plots = sum(len(v) for v in rec['metrics'].values())
    print(f"{OK}assembled — buckets {sorted(rec['metrics'])}, {n_plots} plot(s)")
    if not rec['metrics']:
        print(f"{BAD}no metrics extracted — the push would skip with "
              f"'no sidecar data found in output'")
        print(f"{INFO}check that _plot_data/*.json exist under the job output")
    entry = record.build_index_entry(rec, bin_id)
    size = len(json.dumps(entry, ensure_ascii=False).encode())
    print(f"{OK}index entry {size} B, {len(entry['Data'])} Data element(s)")


# ── Summary ───────────────────────────────────────────────────────
head("Result")
print("All layers reachable. JSONBin credentials, collection and index bin")
print("are working, and a bin was successfully created.")
print()
print("If analyses still are not appearing in the index, the cause is upstream")
print("of the transport. Check, in order:")
print("  * the job's message field for 'Metrics push warning: ...'")
print("  * GET /api/jobs/<job_id> for the 'metrics_push' object")
print("  * the service logs for lines starting '[metrics]'")
print()
print(f"Remember to delete the probe bin {bin_id}.")
