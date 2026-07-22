"""
Offline harness for record.py. No network, no fixtures on disk beyond a tmpdir.

Run:  python3 scripts/helpers/test_record.py
Exit code is non-zero if any assertion fails.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.helpers.record import (  # noqa: E402
    build_detail_record, build_index_entry, detail_bin_name,
    parse_conditions, parse_metric_kv, plot_bucket, build_key_values,
    is_comparison_script, load_sidecars,
    SCHEMA_VERSION,
)

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


def entry_size(entry):
    return len(json.dumps(entry, ensure_ascii=False).encode()) + 2


# ─────────────────────────────────────────────────────────────────────
#  Sidecar fixtures
# ─────────────────────────────────────────────────────────────────────

def sidecar(plot_type, texts=(), axvlines=(), lines=()):
    return {
        'plot_type': plot_type,
        'data': {'axes': [{
            'title': '', 'texts': [{'text': t} for t in texts],
            'axhlines': [], 'axvlines': [{'label': l} for l in axvlines],
            'lines': [{'label': l} for l in lines],
        }]},
    }


FIXTURES = {
    'polcurve_b14_IV_80C_100RH_0o3V_0o2H2_0o2Air_0kPa': sidecar(
        'polcurve',
        texts=['OCV = 0.950 V\nPeak P = 560 mW/cm²\nV @ 1 A/cm² = 0.559 V']),
    'polcurve_c10_IV_60C_50RH_0o2H2_0o5Air_150kPa': sidecar(
        'polcurve', texts=['OCV = 0.940 V\nPeak P = 512 mW/cm²']),
    'eis_b14_80C_100RH': sidecar('eis', axvlines=['HFR = 0.0450 Ω·cm²']),
    'crossover_a2_80C_100RH_0o2H2_0o2N2': sidecar(
        'crossover', texts=['|j_xover| = 1.200 mA/cm2\nH2 flux = 8.6 nmol/cm2/s']),
    'cleaning_cycles_s1_CV-500mVs': sidecar(
        'cleaning_cycles',
        texts=['Cycles = 15\nConvergence = cycle 12\nDL_reduction = 60.2%']),
    'ecsa_hupd_a2_80C_100RH': sidecar('ecsa_hupd', texts=['RF = 12.1']),
    # Aggregate view: no conditions in the filename, no promoted values.
    'polcurve_batch_overlay': sidecar('polcurve_overlay'),
}


def write_fixtures(root, names):
    d = Path(root) / 'output' / '_plot_data'
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / f'{n}.json').write_text(json.dumps(FIXTURES[n]))
    return Path(root) / 'output'


# ─────────────────────────────────────────────────────────────────────

print("parse_metric_kv")
check('unit parsed', parse_metric_kv('OCV = 0.95 V'),
      {'OCV': {'value': 0.95, 'unit': 'V'}})
check('bare number', parse_metric_kv('Cycles = 15'), {'Cycles': 15.0})
check('non-numeric', parse_metric_kv('Convergence = cycle 12'),
      {'Convergence': 'cycle 12'})
check('scientific', parse_metric_kv('j₀ = 1.20e-06 A/cm²'),
      {'j₀': {'value': 1.2e-06, 'unit': 'A/cm²'}})
check('percent unit', parse_metric_kv('DL_reduction = 60.2%'),
      {'DL_reduction': {'value': 60.2, 'unit': '%'}})

print("parse_conditions")
c = parse_conditions('polcurve_b14_IV_80C_100RH_0o3V_0o2H2_0o2Air_0kPa')
check('step', c.get('step'), 'b14')
check('T normalised', c.get('T_C'), 80.0)
check('RH normalised', c.get('RH_pct'), 100.0)
check('gas decoded', c.get('H2_slpm'), 0.2)
check('air decoded', c.get('Air_slpm'), 0.2)
check('pressure', (c.get('P_value'), c.get('P_unit')), (0.0, 'kPa'))
check('V setpoint', c.get('V_setpoint'), 0.3)

# Regression: the permissive step pattern used elsewhere matches 'H2' in a gas
# spec when no real step is present. Token anchoring must prevent that.
check('no false step from gas token',
      parse_conditions('eis_80C_100RH_0o2H2').get('step'), None)
check('no false step from sample name',
      parse_conditions('260126_FCS6_polcurve').get('step'), None)
check('sparse ok', parse_conditions('eis_b14_80C_100RH'),
      {'step': 'b14', 'T_C': 80.0, 'RH_pct': 100.0})
check('empty ok', parse_conditions('cleaning_cycles_CV-500mVs'), {})
# Multi-letter sample prefixes must not match: 'FCS6' is 3 letters + digit.
check('multi-letter token is not a step',
      parse_conditions('260126_FCS6-PolCurve-polcurve_80C').get('step'), None)
# Known limitation: a single-letter+digits token is a valid step shape, so a
# sample label of the same shape is indistinguishable. Pinned deliberately.
check('single-letter sample label reads as step (known)',
      parse_conditions('cleaning_cycles_s1_CV-500mVs').get('step'), 's1')

print("plot_bucket")
for pt, want in (('polcurve_down', 'polcurve'), ('ir_correction', 'polcurve'),
                 ('nyquist', 'eis'), ('ecsa_hupd', 'ecsa'),
                 ('cleaning_diagnostics', 'cleaning'),
                 ('durability_hfr', 'durability'), ('CLR_analysis', 'CLR')):
    check(f'{pt}', plot_bucket(pt), want)

print("key_values sourcing")
check('from parsed values',
      build_key_values('polcurve', {'OCV': {'value': 0.95, 'unit': 'V'}}, {}),
      {'OCV': 0.95})
check('summary wins over parsed',
      build_key_values('polcurve', {'OCV': {'value': 0.95, 'unit': 'V'}},
                       {'OCV': 0.88}),
      {'OCV': 0.88})
# Regression: the summary carries the results-dict name while the parsed values
# carry the display name. A name-ordered search would prefer the annotation and
# silently lose precision; source-ordered search must prefer the summary.
check('summary wins under a different candidate name',
      build_key_values('polcurve',
                       {'V @ 1 A/cm²': {'value': 0.607, 'unit': 'V'}},
                       {'V_at_1Acm2': 0.6074496081374499}),
      {'V @ 1 A/cm²': 0.60745})
check('parsed used when summary lacks every candidate',
      build_key_values('polcurve',
                       {'V @ 1 A/cm²': {'value': 0.607, 'unit': 'V'}},
                       {'Label': 'x'}),
      {'V @ 1 A/cm²': 0.607})
# Rounding is by significant figures so small magnitudes survive.
check('long float rounded', build_key_values(
    'polcurve', {}, {'OCV': 0.8992422222222223}), {'OCV': 0.899242})
check('small magnitude preserved', build_key_values(
    'crossover', {}, {'j_xover_mA_cm2': 0.00000123456789}),
    {'|j_xover|': 1.23457e-06})
check('crossover raw results key accepted',
      build_key_values('crossover', {}, {'j_xover_mA_cm2': 1.2}),
      {'|j_xover|': 1.2})
check('ecsa from summary only',
      build_key_values('ecsa', {'RF': 12.1}, {'average_ECSA_m2_per_g': 44.4}),
      {'Average ECSA': 44.4})
check('no ecsa without summary', build_key_values('ecsa', {'RF': 12.1}, {}), {})
check('non-numeric rejected',
      build_key_values('crossover', {'|j_xover|': 'n/a'}, {}), {})

print("detail record — single analysis")
tmp = tempfile.mkdtemp()
try:
    out = write_fixtures(tmp, ['polcurve_b14_IV_80C_100RH_0o3V_0o2H2_0o2Air_0kPa'])
    rec = build_detail_record(
        job_id='20260430-220515-a3f1c2', sample_name='260126_FCS6',
        script='FC Polarization Curve', timestamp='2026-04-30T22:05:15Z',
        input_files=['cellA.fcd'], output_dir=out)
    check('schema', rec['schema'], SCHEMA_VERSION)
    check('bucket present', list(rec['metrics'].keys()), ['polcurve'])
    plot = list(rec['metrics']['polcurve'].values())[0]
    check('per-plot conditions', plot['conditions']['T_C'], 80.0)
    check('per-plot values', plot['values']['OCV'], {'value': 0.95, 'unit': 'V'})
    check_true('top conditions populated when uniform',
               rec['conditions'] is not None and rec['conditions']['T_C'] == 80.0)
    check_true('sidecars embedded (tier 3)', 'sidecars' in rec)
    check_true('no summary key when absent', 'summary' not in rec)

    idx = build_index_entry(rec, '6a17582221f9ee59d292d410')
    check('one Data element', len(idx['Data']), 1)
    check('Analysis', idx['Data'][0]['Analysis'], 'polcurve')
    check('step hoisted out of Conditions', idx['Data'][0]['step'], 'b14')
    check_true('step removed from Conditions',
               'step' not in idx['Data'][0]['Conditions'])
    check('key_values', idx['Data'][0]['key_values'],
          {'OCV': 0.95, 'V @ 1 A/cm²': 0.559})
    sz = entry_size(idx)
    check_true('single-run entry ~393 B', 340 <= sz <= 450, f'(got {sz} B)')
    print(f"       single polcurve index entry: {sz} B")
finally:
    shutil.rmtree(tmp)

print("detail record — multi-condition (top-level conditions must be null)")
tmp = tempfile.mkdtemp()
try:
    out = write_fixtures(tmp, [
        'polcurve_b14_IV_80C_100RH_0o3V_0o2H2_0o2Air_0kPa',
        'polcurve_c10_IV_60C_50RH_0o2H2_0o5Air_150kPa'])
    rec = build_detail_record(
        job_id='j2', sample_name='260126_FCS6', script='FC Polarization Curve',
        timestamp='2026-04-30T22:05:15Z', input_files=[], output_dir=out)
    check_true('top conditions null when mixed', rec['conditions'] is None)
    idx = build_index_entry(rec, 'bin2')
    check('two Data elements', len(idx['Data']), 2)
    steps = sorted(d['step'] for d in idx['Data'])
    check('both steps present', steps, ['b14', 'c10'])
    temps = sorted(d['Conditions']['T_C'] for d in idx['Data'])
    check('conditions not blended', temps, [60.0, 80.0])
finally:
    shutil.rmtree(tmp)

print("detail record — full analysis fan-out")
tmp = tempfile.mkdtemp()
try:
    out = write_fixtures(tmp, list(FIXTURES.keys()))
    rec = build_detail_record(
        job_id='j3', sample_name='260126_FCS6',
        script='Fuel Cell Full Analysis', timestamp='2026-04-30T22:05:15Z',
        input_files=['a.fcd', 'b.fcd'], output_dir=out,
        summary=[{'Label': 'cellA', 'average_ECSA_m2_per_g': 44.4}])
    check_true('summary embedded (tier 1)', 'summary' in rec)
    buckets = sorted(rec['metrics'].keys())
    check('all buckets', buckets,
          ['cleaning', 'crossover', 'ecsa', 'eis', 'polcurve'])

    idx = build_index_entry(rec, 'bin3')
    check('six analysis units', len(idx['Data']), 6)
    by = {(d['Analysis'], d['step']): d for d in idx['Data']}
    check('eis key_values', by[('eis', 'b14')]['key_values'], {'HFR': 0.045})
    check('crossover key_values', by[('crossover', 'a2')]['key_values'],
          {'|j_xover|': 1.2})
    check('ecsa promoted from summary',
          by[('ecsa', 'a2')]['key_values'], {'Average ECSA': 44.4})
    # Known ambiguity: 's1' in 'cleaning_cycles_s1_CV-500mVs' is a sample label,
    # not a protocol step, but is indistinguishable from one by filename alone —
    # both are a single letter followed by digits. Recorded as current behaviour
    # rather than guessed away; see delivery notes.
    check_true('cleaning has no key_values',
               'key_values' not in by[('cleaning', 's1')])
    sz = entry_size(idx)
    check_true('full-analysis entry ~1000 B', 700 <= sz <= 1400, f'(got {sz} B)')
    print(f"       full-analysis index entry:  {sz} B")

    detail_sz = len(json.dumps(rec, ensure_ascii=False).encode())
    print(f"       detail record with sidecars: {detail_sz:,} B")
    check_true('detail record well under 10 MB', detail_sz < 10_000_000)

    check_true('bin name capped',
               len(detail_bin_name(rec, 'FCAnalysis')) <= 128)
finally:
    shutil.rmtree(tmp)

print("aggregate plots (batch overlays)")
tmp = tempfile.mkdtemp()
try:
    # A real measurement plus its batch overlay. The overlay must not add a
    # degenerate unit, and must not defeat the uniform-conditions test.
    out = write_fixtures(tmp, [
        'polcurve_b14_IV_80C_100RH_0o3V_0o2H2_0o2Air_0kPa',
        'polcurve_batch_overlay'])
    rec = build_detail_record(
        job_id='j6', sample_name='260126_FCS6', script='FC Polarization Curve',
        timestamp='2026-04-30T22:05:15Z', input_files=[], output_dir=out)
    check_true('overlay does not null top conditions',
               rec['conditions'] is not None
               and rec['conditions'].get('T_C') == 80.0)
    check_true('overlay still recorded in metrics',
               'polcurve_batch_overlay' in rec['metrics']['polcurve'])
    idx = build_index_entry(rec, 'bin6')
    check('overlay suppressed from index', len(idx['Data']), 1)
    check('surviving unit is the measurement', idx['Data'][0]['step'], 'b14')

    # A bucket whose only plot has no parseable conditions must survive, or the
    # run vanishes from the index entirely.
    out2 = write_fixtures(tmp, ['cleaning_cycles_s1_CV-500mVs'])
    rec2 = build_detail_record(
        job_id='j7', sample_name='s', script='FC Electrode Cleaning',
        timestamp='t', input_files=[], output_dir=out2)
    idx2 = build_index_entry(rec2, 'bin7')
    check_true('lone condition-less bucket survives',
               any(d['Analysis'] == 'cleaning' for d in idx2['Data']))
finally:
    shutil.rmtree(tmp)

print("comparison exclusion")
check_true('registry name', is_comparison_script('Plot Comparison'))
check_true('legacy name', is_comparison_script('Compare Polcurves'))
check_true('substring match', is_comparison_script('Polcurve Comparison v2'))
check_true('compare prefix', is_comparison_script('Compare Anything'))
check_true('regular script not excluded',
           not is_comparison_script('FC Polarization Curve'))
check_true('none tolerated', not is_comparison_script(None))
check_true('empty tolerated', not is_comparison_script(''))

tmp = tempfile.mkdtemp()
try:
    out = write_fixtures(tmp, ['polcurve_b14_IV_80C_100RH_0o3V_0o2H2_0o2Air_0kPa'])
    # A comparison job must be refused outright, loudly.
    raised = False
    try:
        build_detail_record(
            job_id='jc', sample_name='A_B', script='Plot Comparison',
            timestamp='t', input_files=[], output_dir=out)
    except ValueError:
        raised = True
    check_true('comparison job refused', raised)

    # Comparison sidecars are skipped even inside an otherwise normal job.
    d = Path(out) / '_plot_data'
    (d / 'polcurve_cmp_flagged.json').write_text(json.dumps(
        {'plot_type': 'polcurve', 'comparison': True,
         'data': {'axes': [{'texts': [{'text': 'OCV = 0.1 V'}],
                            'axhlines': [], 'axvlines': [], 'lines': []}]}}))
    (d / 'polcurve_cmp_bytype.json').write_text(json.dumps(
        {'plot_type': 'polcurve_comparison',
         'data': {'axes': [{'texts': [{'text': 'OCV = 0.2 V'}],
                            'axhlines': [], 'axvlines': [], 'lines': []}]}}))
    (d / 'polcurve_cmp_meta.json').write_text(json.dumps(
        {'plot_type': 'polcurve', 'metadata': {'comparison': True},
         'data': {'axes': [{'texts': [{'text': 'OCV = 0.3 V'}],
                            'axhlines': [], 'axvlines': [], 'lines': []}]}}))

    # Regression: 'polcurve_hfrcompare' is the EIS-vs-current-interrupt
    # analysis, not comparison output. A substring match on 'compare' would
    # discard it and silently lose a real measurement.
    (d / 'polcurve_hfrcmp_real.json').write_text(json.dumps(
        {'plot_type': 'polcurve_hfrcompare',
         'data': {'axes': [{'texts': [{'text': 'OCV = 0.9 V'}],
                            'axhlines': [], 'axvlines': [], 'lines': []}]}}))

    sc = load_sidecars(out)
    check_true('hfrcompare analysis NOT excluded',
               'polcurve_hfrcmp_real' in sc)
    check('only real plots load', len(sc), 2)
    check_true('flagged sidecar excluded', 'polcurve_cmp_flagged' not in sc)
    check_true('plot_type sidecar excluded', 'polcurve_cmp_bytype' not in sc)
    check_true('metadata sidecar excluded', 'polcurve_cmp_meta' not in sc)

    rec = build_detail_record(
        job_id='j8', sample_name='260126_FCS6', script='FC Polarization Curve',
        timestamp='t', input_files=[], output_dir=out)
    # Two legitimate plots survive: the polcurve and the hfrcompare analysis.
    # The three comparison-flagged sidecars are gone.
    check('only legitimate plots in metrics',
          len(rec['metrics']['polcurve']), 2)
    idx = build_index_entry(rec, 'bin8')
    b14 = [d for d in idx['Data'] if d['step'] == 'b14']
    check('measurement unit present', len(b14), 1)
    check('index unaffected by comparison sidecars',
          b14[0]['key_values']['OCV'], 0.95)
finally:
    shutil.rmtree(tmp)

print("edge cases")
tmp = tempfile.mkdtemp()
try:
    empty = Path(tmp) / 'output'
    empty.mkdir(parents=True)
    rec = build_detail_record(
        job_id='j4', sample_name='', script='OCV Analysis',
        timestamp='2026-04-30T22:05:15Z', input_files=None, output_dir=empty)
    check('no sidecars -> empty metrics', rec['metrics'], {})
    check_true('no sidecars -> conditions null', rec['conditions'] is None)
    check_true('no sidecars key', 'sidecars' not in rec)
    idx = build_index_entry(rec, 'bin4')
    check('empty Data', idx['Data'], [])

    rec2 = build_detail_record(
        job_id='j5', sample_name='s', script='x',
        timestamp='t', input_files=[], output_dir=Path(tmp) / 'missing')
    check('missing dir tolerated', rec2['metrics'], {})
finally:
    shutil.rmtree(tmp)

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
