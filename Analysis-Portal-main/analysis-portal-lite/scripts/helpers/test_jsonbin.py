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


def _body(mock, method):
    """Body of the first call made with a given method."""
    return next(c['body'] for c in mock.calls if c['method'] == method)


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
    # The index is read first, to look the sample up, then reused for the
    # write — so a new sample costs GET, POST, PUT.
    check('three calls', [c['method'] for c in m.calls], ['GET', 'POST', 'PUT'])
    check_true('a new sample is not reported as merged', not r['merged'])

    post = next(c for c in m.calls if c['method'] == 'POST')
    check('POST to bins root', post['url'], jsonbin._JSONBIN_BASE)
    check('collection header', post['headers'].get('X-Collection-Id'), 'COLL123')
    check('private on create', post['headers'].get('X-Bin-Private'), 'true')
    check_true('bin name set', post['headers'].get('X-Bin-Name', '').startswith(
        '260126_FCS6-PolCurve-'))
    check('detail body is schema 2', post['body']['schema'], 2)
    check_true('detail carries sidecars (tier 3)', 'sidecars' in post['body'])

    get = next(c for c in m.calls if c['method'] == 'GET')
    check('GET latest index', get['url'],
          f"{jsonbin._JSONBIN_BASE}/INDEX456/latest")

    put = next(c for c in m.calls if c['method'] == 'PUT')
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
    check_true('a created bin is described as an orphan',
               'orphan' in r['reason'])

    # Index read fails: same orphan treatment.
    m = Mock(fail_on={'GET'}); install(m)
    r = push(out)
    # A failing GET breaks the sample lookup first; the run still lands in a
    # new bin, and the index write then fails on the same GET.
    check_true('read failure still creates the bin', r['bin_id'] == 'NEWBIN')
    check_true('read failure is reported', not r['pushed'] and bool(r['reason']))
finally:
    shutil.rmtree(tmp)

print("sidecar selection and size guard")
tmp = tempfile.mkdtemp()
try:
    out = make_output(tmp)
    d = Path(out) / '_plot_data'
    # A heavy cleaning sidecar alongside the light polcurve one.
    (d / 'cleaning_cycles_a4_EClean50x.json').write_text(json.dumps(
        {'plot_type': 'cleaning_cycles',
         'data': {'axes': [{'texts': [{'text': 'Cycles = 50'}],
                            'axhlines': [], 'axvlines': [],
                            'lines': [{'y': list(range(2000))}]}]}}))

    original_excl = jsonbin.SIDECAR_EXCLUDE_BUCKETS
    original_max = jsonbin.MAX_BODY_BYTES

    # Default excludes cleaning by bucket, regardless of available room.
    jsonbin.SIDECAR_EXCLUDE_BUCKETS = {'cleaning'}
    m = Mock(); install(m)
    r = push(out)
    check_true('pushed with cleaning excluded', r['pushed'], f"({r['reason']})")
    check('one sidecar kept', r['sidecars_kept'], 1)
    check('cleaning dropped', r['sidecars_dropped'], 1)
    check_true('flagged omitted', r['sidecars_omitted'])
    check_true('reason counts them', '1 of 2 sidecars' in (r['reason'] or ''),
               f"({r['reason']})")
    # Full breakdown is reported even for buckets that were not stored.
    check_true('bucket breakdown reported',
               'cleaning' in r['sidecar_bytes_by_bucket']
               and 'polcurve' in r['sidecar_bytes_by_bucket'])
    sent = _body(m, 'POST')
    check_true('sidecars present', 'sidecars' in sent)

    # With nothing excluded, both are stored.
    jsonbin.SIDECAR_EXCLUDE_BUCKETS = set()
    m = Mock(); install(m)
    r = push(out)
    check('both kept when nothing excluded', r['sidecars_kept'], 2)
    check('none dropped', r['sidecars_dropped'], 0)
    check_true('not flagged omitted', not r['sidecars_omitted'])
    check_true('no reason on clean push', r['reason'] is None)

    # A budget too small for everything keeps the smallest and still pushes.
    jsonbin.MAX_BODY_BYTES = 4000
    m = Mock(); install(m)
    r = push(out)
    check_true('still pushes under a tight budget', r['pushed'],
               f"({r['reason']})")
    check_true('body within limit', r['body_bytes'] <= 4000,
               f"({r['body_bytes']} B)")
    check_true('something was dropped', r['sidecars_dropped'] >= 1)
    check('index still appended', len(m.written_index['runs']), 1)
    # Data holds one unit per analysis; find the polcurve one rather than
    # assuming ordering.
    pol = [d for d in m.written_index['runs'][0]['Data']
           if d['Analysis'] == 'polcurve']
    check('polcurve unit present', len(pol), 1)
    check_true('index entry unaffected by sidecar trimming',
               pol[0]['key_values']['OCV'] == 0.95)

    # A budget too small even for the base record degrades to no sidecars.
    jsonbin.MAX_BODY_BYTES = 200
    m = Mock(); install(m)
    r = push(out)
    check_true('still pushes when base alone is over', r['pushed'],
               f"({r['reason']})")
    sent = _body(m, 'POST')
    check_true('sidecars gone entirely', 'sidecars' not in sent)
    check_true('omission marked in record', 'sidecars_omitted' in sent)
    check_true('metrics survive', bool(sent.get('metrics')))

    jsonbin.SIDECAR_EXCLUDE_BUCKETS = original_excl
    jsonbin.MAX_BODY_BYTES = original_max
finally:
    shutil.rmtree(tmp)

print("merge by sample")

class MergeMock:
    """Index and detail bins that persist across pushes, like the real store."""

    def __init__(self):
        self.calls = []
        self.index = {'schema': 2, 'runs': []}
        self.bins = {}
        self._n = 0

    def __call__(self, url, method='GET', body=None, extra_headers=None):
        self.calls.append({'url': url, 'method': method, 'body': body,
                           'headers': extra_headers or {}})
        if method == 'POST':
            self._n += 1
            bid = f'BIN{self._n}'
            self.bins[bid] = json.loads(json.dumps(body))
            return {'record': body, 'metadata': {'id': bid}}
        if method == 'GET':
            for bid in self.bins:
                if f'/{bid}/' in url:
                    return {'record': json.loads(json.dumps(self.bins[bid]))}
            return {'record': json.loads(json.dumps(self.index))}
        if method == 'PUT':
            for bid in self.bins:
                if url.endswith(f'/{bid}'):
                    self.bins[bid] = json.loads(json.dumps(body))
                    return {'record': body}
            self.index = json.loads(json.dumps(body))
            return {'record': body}
        raise AssertionError(method)


def make_merge_output(root, plots):
    d = Path(root) / 'output' / '_plot_data'
    d.mkdir(parents=True, exist_ok=True)
    for name, pt, txt in plots:
        d_sc = {'plot_type': pt, 'data': {'axes': [{
            'texts': [{'text': txt}] if txt else [],
            'axhlines': [], 'axvlines': [], 'lines': []}]}}
        (d / f'{name}.json').write_text(json.dumps(d_sc))
    return Path(root) / 'output'


XO = lambda st: (f'crossover_{st}_G_H2X_80.3Cell-100RH_0.4WH2', 'crossover',
                 '|j_xover| = 1.2 mA/cm2')
PC = lambda st: (f'polcurve_{st}_G_PolarizationCurve_80.3Cell-95RH_0.2WH2',
                 'polcurve', 'OCV = 0.95 V')

tmp = tempfile.mkdtemp()
try:
    m = MergeMock(); install(m)
    SAMPLE = '260421_GSMA-Qual-1'

    # First push: a standalone crossover at two steps.
    out1 = make_merge_output(Path(tmp) / 'r1', [XO('a6'), XO('b28b')])
    r1 = jsonbin.push_job_metrics(
        job_id='job-1', sample_name=SAMPLE, script='H2 Crossover',
        output_dir=out1, input_files=['a.csv'], script_short='XO')
    check_true('first push succeeds', r1['pushed'], f"({r1['reason']})")
    check_true('first push is not a merge', not r1['merged'])
    check('index has one entry', len(m.index['runs']), 1)
    first_bin = r1['bin_id']

    # Second push: Full Analysis covering one of those steps plus new ones.
    out2 = make_merge_output(Path(tmp) / 'r2', [PC('b17b'), PC('c6'), XO('a6')])
    r2 = jsonbin.push_job_metrics(
        job_id='job-2', sample_name=SAMPLE, script='Fuel Cell Full Analysis',
        output_dir=out2, input_files=['b.csv'], script_short='FCA')
    check_true('second push succeeds', r2['pushed'], f"({r2['reason']})")
    check_true('second push reports a merge', r2['merged'])
    check('same bin reused', r2['bin_id'], first_bin)
    check('index still has one entry', len(m.index['runs']), 1)
    check('n_jobs tracks contributors', r2['n_jobs'], 2)

    # The merged bin holds both runs' data, with the covered step replaced.
    binrec = m.bins[first_bin]
    check('merged bin has both buckets',
          sorted(binrec['metrics'].keys()), ['crossover', 'polcurve'])
    check('untouched crossover step survives',
          len(binrec['metrics']['crossover']), 2)
    check('polcurve added', len(binrec['metrics']['polcurve']), 2)
    check('provenance records both jobs',
          [j['job_id'] for j in binrec['jobs']], ['job-1', 'job-2'])

    entry = m.index['runs'][0]
    units = sorted((u['Analysis'], u['step']) for u in entry['Data'])
    check('index units match the bin', units,
          [('crossover', 'a6'), ('crossover', 'b28b'),
           ('polcurve', 'b17b'), ('polcurve', 'c6')])
    check('entry job_id is the last writer', entry['job_id'], 'job-2')

    # A different sample must not merge into this one.
    out3 = make_merge_output(Path(tmp) / 'r3', [PC('b17b')])
    r3 = jsonbin.push_job_metrics(
        job_id='job-3', sample_name='260429_GSMA-Qual-3',
        script='Fuel Cell Full Analysis', output_dir=out3,
        input_files=['c.csv'], script_short='FCA')
    check_true('different sample is not merged', not r3['merged'])
    check('different sample gets its own bin', r3['bin_id'] != first_bin, True)
    check('index now has two entries', len(m.index['runs']), 2)

    # The kill switch restores append-only behaviour.
    jsonbin.MERGE_BY_SAMPLE = False
    out4 = make_merge_output(Path(tmp) / 'r4', [XO('a6')])
    r4 = jsonbin.push_job_metrics(
        job_id='job-4', sample_name=SAMPLE, script='H2 Crossover',
        output_dir=out4, input_files=['d.csv'], script_short='XO')
    check_true('kill switch disables merging', not r4['merged'])
    check('kill switch appends instead', len(m.index['runs']), 3)
    jsonbin.MERGE_BY_SAMPLE = True
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
    check_true('summary embedded in detail', 'summary' in _body(m, 'POST'))
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