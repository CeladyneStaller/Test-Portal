"""
Offline harness for jsonbin.py. _request is mocked, so no network is touched.

Run:  python3 scripts/helpers/test_jsonbin.py
Exit code is non-zero if any assertion fails.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.helpers import jsonbin  # noqa: E402

_passed = 0
_failed = 0


def check(name, got, want):
    global _passed, _failed
    if got == want:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL {name}\n       got:  {got!r}\n       want: {want!r}")


def check_true(name, cond, detail=''):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL {name} {detail}")


# ─────────────────────────────────────────────────────────────────────
#  Mock transport
# ─────────────────────────────────────────────────────────────────────

class Mock:
    """Records every call and returns scripted responses."""

    def __init__(self, index_runs=None, fail_on=None, new_bin_id='NEWBIN'):
        self.calls = []
        self.index_runs = list(index_runs or [])
        self.fail_on = fail_on or set()   # e.g. {'POST', 'PUT', 'GET'}
        self.new_bin_id = new_bin_id
        self.written_index = None

    def __call__(self, url, method='GET', body=None, extra_headers=None):
        self.calls.append({'url': url, 'method': method, 'body': body,
                           'headers': extra_headers or {}})
        if method in self.fail_on:
            raise RuntimeError(f"simulated {method} failure")
        if method == 'POST':
            return {'record': body, 'metadata': {'id': self.new_bin_id}}
        if method == 'GET':
            return {'record': {'schema': 2, 'runs': self.index_runs},
                    'metadata': {}}
        if method == 'PUT':
            self.written_index = body
            return {'record': body, 'metadata': {}}
        raise AssertionError(f'unexpected method {method}')


def install(mock):
    jsonbin._request = mock


def configure(on=True):
    if on:
        os.environ['JSONBIN_API_KEY'] = 'testkey'
        os.environ['JSONBIN_COLLECTION_ID'] = 'COLL123'
        os.environ['JSONBIN_INDEX_BIN_ID'] = 'INDEX456'
    else:
        for k in ('JSONBIN_API_KEY', 'JSONBIN_COLLECTION_ID',
                  'JSONBIN_INDEX_BIN_ID'):
            os.environ.pop(k, None)


# ─────────────────────────────────────────────────────────────────────
#  Fixture output dir
# ─────────────────────────────────────────────────────────────────────

SIDECAR = {
    'plot_type': 'polcurve',
    'data': {'axes': [{
        'texts': [{'text': 'OCV = 0.950 V\nV @ 1 A/cm² = 0.559 V'}],
        'axhlines': [], 'axvlines': [], 'lines': [],
    }]},
}


def make_output(root):
    d = Path(root) / 'output' / '_plot_data'
    d.mkdir(parents=True, exist_ok=True)
    name = 'polcurve_b14_IV_80C_100RH_0o2H2_0o2Air_0kPa'
    (d / f'{name}.json').write_text(json.dumps(SIDECAR))
    return Path(root) / 'output'


def push(out, **kw):
    args = dict(job_id='20260430-220515-a3f1c2', sample_name='260126_FCS6',
                script='FC Polarization Curve', output_dir=out,
                input_files=['cellA.fcd'], script_short='PolCurve')
    args.update(kw)
    return jsonbin.push_job_metrics(**args)


# ─────────────────────────────────────────────────────────────────────

print("configuration")
configure(False)
check_true('unconfigured', not jsonbin.is_configured())
tmp = tempfile.mkdtemp()
try:
    out = make_output(tmp)
    r = push(out)
    check_true('skips when unconfigured', r['pushed'] is False)
    check_true('names the missing vars',
               'JSONBIN_API_KEY' in r['reason']
               and 'JSONBIN_COLLECTION_ID' in r['reason']
               and 'JSONBIN_INDEX_BIN_ID' in r['reason'], f"({r['reason']})")
finally:
    shutil.rmtree(tmp)

configure(True)
check_true('configured', jsonbin.is_configured())

print("happy path")
tmp = tempfile.mkdtemp()
try:
    out = make_output(tmp)
    m = Mock(); install(m)
    r = push(out)

    check_true('pushed', r['pushed'], f"({r['reason']})")
    check('bin id returned', r['bin_id'], 'NEWBIN')
    check('index length', r['n_runs'], 1)
    check('three calls', [c['method'] for c in m.calls], ['POST', 'GET', 'PUT'])

    post = m.calls[0]
    check('POST to bins root', post['url'], jsonbin._JSONBIN_BASE)
    check('collection header', post['headers'].get('X-Collection-Id'), 'COLL123')
    check('private on create', post['headers'].get('X-Bin-Private'), 'true')
    check_true('bin name set', post['headers'].get('X-Bin-Name', '').startswith(
        '260126_FCS6-PolCurve-'))
    check('detail body is schema 2', post['body']['schema'], 2)
    check_true('detail carries sidecars (tier 3)', 'sidecars' in post['body'])

    get = m.calls[1]
    check('GET latest index', get['url'],
          f"{jsonbin._JSONBIN_BASE}/INDEX456/latest")

    put = m.calls[2]
    check('PUT index', put['url'], f"{jsonbin._JSONBIN_BASE}/INDEX456")
    check_true('PUT sends no X-Bin-Private',
               'X-Bin-Private' not in put['headers'])
    check('one run appended', len(put['body']['runs']), 1)
    entry = put['body']['runs'][0]
    check('entry points at detail bin', entry['bin_id'], 'NEWBIN')
    check('entry Data', len(entry['Data']), 1)
    check('entry key_values', entry['Data'][0]['key_values'],
          {'OCV': 0.95, 'V @ 1 A/cm²': 0.559})
    size = len(json.dumps(entry, ensure_ascii=False).encode())
    check_true('entry compact', size < 600, f'({size} B)')
finally:
    shutil.rmtree(tmp)

print("append preserves existing runs")
tmp = tempfile.mkdtemp()
try:
    out = make_output(tmp)
    m = Mock(index_runs=[{'job_id': 'older'}]); install(m)
    r = push(out)
    check('index length', r['n_runs'], 2)
    check('existing run kept', m.written_index['runs'][0]['job_id'], 'older')
    check('new run appended last',
          m.written_index['runs'][1]['job_id'], '20260430-220515-a3f1c2')
    check('schema preserved', m.written_index['schema'], 2)
finally:
    shutil.rmtree(tmp)

print("comparison exclusion")
tmp = tempfile.mkdtemp()
try:
    out = make_output(tmp)
    m = Mock(); install(m)
    r = push(out, script='Plot Comparison')
    check_true('not pushed', r['pushed'] is False)
    check_true('reason names comparison', 'comparison' in r['reason'])
    check('no HTTP calls at all', len(m.calls), 0)
finally:
    shutil.rmtree(tmp)

print("failure handling")
tmp = tempfile.mkdtemp()
try:
    out = make_output(tmp)

    # Detail bin creation fails: nothing written, no orphan.
    m = Mock(fail_on={'POST'}); install(m)
    r = push(out)
    check_true('create failure reported', r['pushed'] is False)
    check_true('no bin id on create failure', r['bin_id'] is None)
    check_true('reason mentions detail bin',
               'detail bin creation failed' in r['reason'], f"({r['reason']})")

    # Index append fails after the bin exists: fork D says keep the orphan.
    m = Mock(fail_on={'PUT'}); install(m)
    r = push(out)
    check_true('append failure reported', r['pushed'] is False)
    check('orphan bin id surfaced', r['bin_id'], 'NEWBIN')
    check_true('reason names the orphan',
               'orphan detail bin NEWBIN retained' in r['reason'],
               f"({r['reason']})")

    # Index read fails: same orphan treatment.
    m = Mock(fail_on={'GET'}); install(m)
    r = push(out)
    check_true('read failure keeps orphan',
               r['bin_id'] == 'NEWBIN' and 'orphan' in r['reason'])
finally:
    shutil.rmtree(tmp)

print("nothing to push")
tmp = tempfile.mkdtemp()
try:
    empty = Path(tmp) / 'output'
    empty.mkdir(parents=True)
    m = Mock(); install(m)
    r = push(empty)
    check_true('no sidecars -> skip', r['pushed'] is False)
    check_true('reason explains', 'no sidecar data' in r['reason'],
               f"({r['reason']})")
    check('no HTTP calls', len(m.calls), 0)
finally:
    shutil.rmtree(tmp)

print("summary pass-through (tier 1)")
tmp = tempfile.mkdtemp()
try:
    out = make_output(tmp)
    m = Mock(); install(m)
    r = push(out, summary=[{'Label': 'cellA', 'OCV': 0.88}])
    check_true('pushed', r['pushed'])
    check_true('summary embedded in detail', 'summary' in m.calls[0]['body'])
    entry = m.written_index['runs'][0]
    check('summary wins over parsed in key_values',
          entry['Data'][0]['key_values']['OCV'], 0.88)
finally:
    shutil.rmtree(tmp)

print("fetch_index tolerates malformed bins")
m = Mock(); install(m)
m.index_runs = []
check('empty index normalised', jsonbin.fetch_index(),
      {'schema': 2, 'runs': []})


def bad_get(url, method='GET', body=None, extra_headers=None):
    return {'record': ['not', 'a', 'dict'], 'metadata': {}}


jsonbin._request = bad_get
check('non-dict record normalised', jsonbin.fetch_index(),
      {'schema': 2, 'runs': []})


def missing_runs(url, method='GET', body=None, extra_headers=None):
    return {'record': {'schema': 2}, 'metadata': {}}


jsonbin._request = missing_runs
check('missing runs normalised', jsonbin.fetch_index(),
      {'schema': 2, 'runs': []})

configure(False)
print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
