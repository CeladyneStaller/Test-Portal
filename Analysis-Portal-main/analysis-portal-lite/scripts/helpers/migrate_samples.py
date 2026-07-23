#!/usr/bin/env python3
"""
Consolidate run-keyed index entries into sample-keyed ones.

Entries written before merging was enabled are one-per-job, so a sample
analysed more than once has several entries pointing at several bins. This
merges each sample's entries and bins into one, using the same (analysis, step)
rule the live write path uses.

    python3 scripts/helpers/migrate_samples.py            # dry run, writes nothing
    python3 scripts/helpers/migrate_samples.py --apply    # performs the merge
    python3 scripts/helpers/migrate_samples.py --restore index-backup-....json

Dry run is the default and reports exactly what would change. `--apply` writes
a backup of the whole index to a timestamped file first, and also prints it, so
a restore is possible from the file or from the deploy logs.

Superseded detail bins are left in place rather than deleted — the same choice
fork D makes for orphans. They stop being referenced by the index; delete them
from the JSONBin dashboard once the result looks right.

On Railway:  railway ssh "python3 /app/scripts/helpers/migrate_samples.py"
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.helpers import jsonbin                                  # noqa: E402
from scripts.helpers.record import (                                 # noqa: E402
    decode_sidecars, merge_detail_record, merge_index_entry,
)

OK, BAD, INFO = "  ok  ", " FAIL ", "      "
APPLY = '--apply' in sys.argv
RESTORE = next((a for a in sys.argv[1:] if a.endswith('.json')), None) \
    if '--restore' in sys.argv else None


def head(t):
    print(f"\n{t}\n" + "-" * len(t))


def die(msg, hint=''):
    print(f"{BAD}{msg}")
    if hint:
        print(f"{INFO}{hint}")
    sys.exit(1)


def _bucket_of(sidecar):
    from scripts.helpers.record import plot_bucket
    return plot_bucket((sidecar or {}).get('plot_type', 'unknown'))


def _entry_sort_key(e):
    return str(e.get('run_date') or e.get('timestamp', ''))


# ── restore mode ──────────────────────────────────────────────────
if RESTORE:
    head("Restore")
    try:
        backup = json.loads(Path(RESTORE).read_text())
    except Exception as e:
        die(f"cannot read {RESTORE}: {e}")
    runs = backup.get('runs')
    if not isinstance(runs, list):
        die(f"{RESTORE} does not look like an index backup (no 'runs' list)")
    print(f"{INFO}{RESTORE} holds {len(runs)} entr(ies)")
    if not APPLY:
        print(f"{INFO}Dry run — add --apply to write it back.")
        sys.exit(0)
    try:
        jsonbin._write_index(backup)
    except Exception as e:
        die(f"restore failed: {e}")
    print(f"{OK}index restored from {RESTORE}")
    sys.exit(0)


# ── 1. read ───────────────────────────────────────────────────────
head("1. Current index")

if not jsonbin.is_configured():
    die("JSONBin is not configured",
        "Run this in the same environment as the app, e.g. railway ssh.")

try:
    index = jsonbin.fetch_index()
except Exception as e:
    die(f"cannot read the index: {e}")

runs = index.get('runs', [])
by_sample = {}
for e in runs:
    by_sample.setdefault(e.get('sample_name', ''), []).append(e)

dupes = {k: v for k, v in by_sample.items() if len(v) > 1}
print(f"{OK}{len(runs)} entr(ies) across {len(by_sample)} sample(s)")
if not dupes:
    print(f"{INFO}Every sample already has a single entry — nothing to do.")
    sys.exit(0)
print(f"{INFO}{len(dupes)} sample(s) have more than one entry:")
for name, group in dupes.items():
    print(f"{INFO}  {name}  ({len(group)} entries)")


# ── 2. plan ───────────────────────────────────────────────────────
head("2. Plan")

plans = []
for name, group in dupes.items():
    group = sorted(group, key=_entry_sort_key)
    survivor = group[-1]                     # newest keeps its bin
    superseded = group[:-1]

    merged_entry = None
    merged_detail = None
    all_sidecars = {}
    protected = set()
    failed = None

    for e in group:
        bid = e.get('bin_id')
        try:
            rec = jsonbin.fetch_detail_bin(bid) if bid else {}
        except Exception as exc:
            failed = f"{bid}: {exc}"
            break
        sc = decode_sidecars(rec)
        # Later entries win on collision, matching the merge rule.
        all_sidecars.update(sc)
        if e is survivor:
            protected = set(sc)
        merged_detail = merge_detail_record(merged_detail, rec)
        merged_entry = merge_index_entry(merged_entry, e)

    if failed:
        print(f"{BAD}{name}: cannot read a detail bin — skipping ({failed})")
        continue

    merged_entry['bin_id'] = survivor.get('bin_id')
    fitted, kept, dropped, body_bytes = jsonbin.fit_record_to_budget(
        merged_detail, all_sidecars, protected=protected)

    units_before = sum(len(e.get('Data') or []) for e in group)
    plans.append({'sample': name, 'survivor': survivor,
                  'superseded': superseded, 'entry': merged_entry,
                  'detail': fitted, 'body_bytes': body_bytes,
                  'kept': len(kept), 'dropped': len(dropped),
                  'units_before': units_before,
                  'units_after': len(merged_entry.get('Data') or [])})

    print(f"\n{INFO}{name}")
    print(f"{INFO}  entries {len(group)} → 1        "
          f"units {units_before} → {len(merged_entry.get('Data') or [])}")
    print(f"{INFO}  surviving bin  {survivor.get('bin_id')}")
    print(f"{INFO}  superseded     {', '.join(e.get('bin_id','?') for e in superseded)}")
    print(f"{INFO}  merged body    {body_bytes:,} B  "
          f"({kept and len(kept) or 0} sidecars kept"
          + (f", {len(dropped)} dropped" if dropped else "") + ")")
    if dropped:
        # Two very different reasons hide behind a dropped sidecar. Buckets in
        # SIDECAR_EXCLUDE_BUCKETS — cleaning by default — are never stored at
        # all, so losing them here costs nothing that was ever kept. Only a
        # size-driven drop actually destroys something.
        excluded = [n for n in dropped
                    if _bucket_of(all_sidecars.get(n, {}))
                    in jsonbin.SIDECAR_EXCLUDE_BUCKETS]
        lost = [n for n in dropped if n not in excluded]
        if excluded:
            print(f"{INFO}  {len(excluded)} excluded by bucket "
                  f"({', '.join(sorted(jsonbin.SIDECAR_EXCLUDE_BUCKETS))}) — "
                  f"never stored, nothing lost")
        if lost:
            print(f"{BAD}  {len(lost)} previously stored sidecar(s) would be "
                  f"dropped to fit the transport limit")

if not plans:
    die("nothing can be migrated")

kept_entries = [e for e in runs
                if not any(e is s for p in plans for s in p['superseded'])]
new_runs = []
for e in kept_entries:
    match = next((p for p in plans if e is p['survivor']), None)
    new_runs.append(match['entry'] if match else e)

print(f"\n{INFO}index would go from {len(runs)} to {len(new_runs)} entr(ies)")


# ── 3. apply ──────────────────────────────────────────────────────
head("3. Apply" + ("" if APPLY else " (dry run)"))

if not APPLY:
    print(f"{INFO}Nothing was written. Re-run with --apply to perform the merge.")
    print(f"{INFO}Superseded bins are kept either way; remove them by hand once")
    print(f"{INFO}the result looks right.")
    sys.exit(0)

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup_path = Path(f'index-backup-{stamp}.json')
try:
    backup_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(f"{OK}index backed up to {backup_path.resolve()}")
except Exception as e:
    print(f"{BAD}could not write a backup file: {e}")
print(f"{INFO}Backup also printed below, so it survives an ephemeral filesystem.")
print(f"{INFO}Restore with: --restore {backup_path} --apply")
print("----- BEGIN INDEX BACKUP -----")
print(json.dumps(index, ensure_ascii=False))
print("----- END INDEX BACKUP -----")

# Detail bins first: if the index write then fails, the bins hold merged data
# and the index still describes the old layout, which is recoverable. The
# reverse order would leave the index pointing at unmerged bins.
for p in plans:
    bid = p['entry']['bin_id']
    try:
        jsonbin.update_detail_bin(bid, p['detail'])
        print(f"{OK}{p['sample']}: merged into bin {bid}")
    except Exception as e:
        die(f"{p['sample']}: writing bin {bid} failed: {e}",
            "The index has not been touched. Nothing is lost; re-run.")

try:
    index['runs'] = new_runs
    jsonbin._write_index(index)
    print(f"{OK}index rewritten — {len(new_runs)} entr(ies)")
except Exception as e:
    die(f"index write failed: {e}",
        f"Detail bins were merged. Re-run to retry, or restore the index with "
        f"--restore {backup_path} --apply")

head("Done")
print(f"{len(plans)} sample(s) consolidated. Superseded bins are still present:")
for p in plans:
    for e in p['superseded']:
        print(f"{INFO}  {e.get('bin_id')}  ({p['sample']})")
print("\nThey are no longer referenced by the index. Delete them from the")
print("JSONBin dashboard once you are satisfied with the result.")
