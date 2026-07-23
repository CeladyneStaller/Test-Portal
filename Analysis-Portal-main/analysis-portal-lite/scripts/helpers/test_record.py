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
    is_comparison_script, load_sidecars, decode_sidecars, strip_sidecars,
    parse_run_date, merge_detail_record, merge_index_entry, touched_units,
    KEY_VALUE_UNITS,
    select_sidecars, sidecar_bucket_sizes, sidecar_sizes, attach_sidecars,
    SCHEMA_VERSION, SIDECAR_ENCODING,
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

print("parse_conditions — both naming conventions")
# Convention A: 'o' decimal marker. Must not regress.
check('convention A parses fully',
      parse_conditions('polcurve_c6_IV_80C_60RH_0o3V_0o4H2_0o4Air_100kPa.png'),
      {'step': 'c6', 'T_C': 80.0, 'RH_pct': 60.0, 'P_value': 100.0,
       'P_unit': 'kPa', 'H2_slpm': 0.4, 'Air_slpm': 0.4, 'V_setpoint': 0.3})

# Convention B: literal decimals, Cell suffix, W/D marker, A for air.
check('convention B parses fully',
      parse_conditions(
          'polcurve_b17b_GSMA-1_PolarizationCurve_80.3Cell-95RH_0.2WH2-0.2WA.png'),
      {'step': 'b17b', 'T_C': 80.3, 'RH_pct': 95.0,
       'H2_slpm': 0.2, 'Air_slpm': 0.2})
check('integer Cell temperature',
      parse_conditions('ocv_a0_X_83Cell-100RH_0.2WN2-0.2DN2B3.png').get('T_C'), 83.0)
check('dry marker accepted',
      parse_conditions('ocv_a0_X_83Cell-100RH_0.2DN2.png').get('N2_slpm'), 0.2)
check('meaningless B3 suffix does not block the gas match',
      parse_conditions('ocv_b21b_X_80.3Cell-100RH_0.05WN2B3.png').get('N2_slpm'), 0.05)

# Trailing-letter steps. Without these, distinct measurements collapse into one
# index unit — b21b through b24b all merged before this was fixed.
for st in ('b17b', 'b21b', 'b45b', 'c6', 'a10', 't1'):
    check(f'step {st}', parse_conditions(
        f'ocv_{st}_GSMA-1_OCV_80.3Cell-50RH_0.05WH2.png').get('step'), st)

# os.path.splitext splits on the last dot, which here is a decimal point: it
# would strip '.2WA' as an extension and silently drop a real gas flow.
check('decimal point is not treated as an extension',
      parse_conditions('polcurve_b4_X_80.3Cell-95RH_0.2WH2-0.2WA').get('Air_slpm'), 0.2)
check('a real extension is still stripped',
      parse_conditions('polcurve_b4_X_80.3Cell-95RH_0.2WH2-0.2WA.png').get('Air_slpm'), 0.2)

# Must not read sample-name fragments as steps.
check('date prefix and sample name are not steps',
      parse_conditions('polcurve_260126_FCS6_IV_80C.png').get('step'), None)
check('multi-letter fragment is not a step',
      parse_conditions('ocv_260511_Gen2-260506_OCV_80C_100RH.png').get('step'), None)

print("run_date")
for name, want in (("260421_GSMA-Qual-1", "2026-04-21"),
                   ("260511_Gen2-260506", "2026-05-11"),   # anchored: not 05-06
                   ("260407_BM1-Qual1", "2026-04-07"),
                   ("260126_FCS6", "2026-01-26"),
                   ("260421", "2026-04-21")):
    check(f'run_date {name}', parse_run_date(name), want)
# Not a date, or not recoverable — must return None so the consumer can fall
# back to the analysis timestamp knowingly.
for name in ("NoDatePrefix-Cell7", "269999_Bad", "123456_X", "260230_BadDay",
             "2604211_TooLong", "", None):
    check(f'run_date rejects {name!r}', parse_run_date(name), None)

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
# Current density at fixed voltages: promoted from the summary, sparse by
# design. A missing key must mean "curve did not reach that voltage", so the
# whitelist must never invent one.
check('j@V promoted from summary',
      build_key_values('polcurve', {}, {'j_at_0.65V': 0.8641009}),
      {'j @ 0.65 V': 0.864101})
check('all five j@V targets promoted',
      sorted(build_key_values('polcurve', {}, {
          'j_at_0.7V': 0.70, 'j_at_0.65V': 0.86, 'j_at_0.6V': 1.02,
          'j_at_0.5V': 1.34, 'j_at_0.4V': 1.66}).keys()),
      ['j @ 0.4 V', 'j @ 0.5 V', 'j @ 0.6 V', 'j @ 0.65 V', 'j @ 0.7 V'])
check('absent j@V targets stay absent',
      sorted(build_key_values('polcurve', {}, {
          'j_at_0.7V': 0.70, 'j_at_0.65V': 0.86}).keys()),
      ['j @ 0.65 V', 'j @ 0.7 V'])
check('j@V units documented',
      [KEY_VALUE_UNITS.get(f'j @ {v:g} V') for v in (0.7, 0.65, 0.6, 0.5, 0.4)],
      ['A/cm²'] * 5)
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

print("per-unit summary matching")
tmp = tempfile.mkdtemp()
try:
    # Two polcurves at different steps, each with its own summary row. Before
    # label matching, both units reported the first row's values.
    d = Path(tmp) / 'output' / '_plot_data'
    d.mkdir(parents=True)
    for step, ocv in (('b17', 0.910), ('b20', 0.880)):
        (d / f'polcurve_{step}_IV_80C_95RH.json').write_text(json.dumps(
            sidecar('polcurve')))
    rec = build_detail_record(
        job_id='jm', sample_name='GSMA-1', script='Fuel Cell Full Analysis',
        timestamp='t', input_files=[], output_dir=Path(tmp) / 'output',
        summary=[
            {'Label': 'b17_IV_80C_95RH', 'Analysis': 'polcurve',
             'OCV': 0.910, 'j_at_0.65V': 0.86},
            {'Label': 'b20_IV_80C_95RH', 'Analysis': 'polcurve',
             'OCV': 0.880, 'j_at_0.65V': 0.70},
        ])
    idx = build_index_entry(rec, 'binm')
    by_step = {u['step']: (u.get('key_values') or {}) for u in idx['Data']}
    check('b17 gets its own OCV', by_step.get('b17', {}).get('OCV'), 0.91)
    check('b20 gets its own OCV', by_step.get('b20', {}).get('OCV'), 0.88)
    check('b17 gets its own j@V', by_step.get('b17', {}).get('j @ 0.65 V'), 0.86)
    check('b20 gets its own j@V', by_step.get('b20', {}).get('j @ 0.65 V'), 0.70)
    check_true('units are not identical',
               by_step.get('b17') != by_step.get('b20'))

    # A single unmatched row is unambiguous and still applies.
    rec2 = build_detail_record(
        job_id='js', sample_name='s', script='FC Polarization Curve',
        timestamp='t', input_files=[], output_dir=Path(tmp) / 'output',
        summary=[{'Label': 'totally-different-name', 'OCV': 0.77}])
    idx2 = build_index_entry(rec2, 'bins')
    check_true('lone unmatched row still applied',
               any((u.get('key_values') or {}).get('OCV') == 0.77
                   for u in idx2['Data']))

    # Several unmatched rows are ambiguous — refuse rather than misattribute.
    rec3 = build_detail_record(
        job_id='ja', sample_name='s', script='Fuel Cell Full Analysis',
        timestamp='t', input_files=[], output_dir=Path(tmp) / 'output',
        summary=[{'Label': 'nope-one', 'OCV': 0.11},
                 {'Label': 'nope-two', 'OCV': 0.22}])
    idx3 = build_index_entry(rec3, 'bina')
    check_true('ambiguous rows are not guessed',
               all('OCV' not in (u.get('key_values') or {}) for u in idx3['Data']))
finally:
    shutil.rmtree(tmp)

print("run_date on the index entry")
tmp = tempfile.mkdtemp()
try:
    out = write_fixtures(tmp, ['polcurve_b14_IV_80C_100RH_0o3V_0o2H2_0o2Air_0kPa'])
    rec = build_detail_record(
        job_id='jd', sample_name='260421_GSMA-Qual-1',
        script='FC Polarization Curve', timestamp='2026-07-22T22:26:04Z',
        input_files=[], output_dir=out)
    idx = build_index_entry(rec, 'bind')
    check('run_date emitted', idx.get('run_date'), '2026-04-21')
    check('analysis timestamp preserved', idx.get('timestamp'),
          '2026-07-22T22:26:04Z')

    rec2 = build_detail_record(
        job_id='jd2', sample_name='NoDatePrefix-Cell7',
        script='FC Polarization Curve', timestamp='2026-07-22T22:26:04Z',
        input_files=[], output_dir=out)
    idx2 = build_index_entry(rec2, 'bind2')
    check_true('run_date omitted when unrecoverable', 'run_date' not in idx2)
finally:
    shutil.rmtree(tmp)

print("Conditions trim")
tmp = tempfile.mkdtemp()
try:
    d = Path(tmp) / 'output' / '_plot_data'
    d.mkdir(parents=True)
    # One polcurve (promotes key_values) and three OCV plots (do not).
    (d / 'polcurve_b17b_X_80.3Cell-95RH_0.2WH2-0.2WA.json').write_text(json.dumps(
        sidecar('polcurve', ['OCV = 0.95 V'])))
    for st, rh in (('b21b', 100), ('b22b', 100), ('b23b', 50)):
        (d / f'ocv_{st}_X_80.3Cell-{rh}RH_0.05WH2.json').write_text(
            json.dumps(sidecar('ocv')))
    rec = build_detail_record(
        job_id='jt', sample_name='s', script='Fuel Cell Full Analysis',
        timestamp='t', input_files=[], output_dir=Path(tmp) / 'output')
    idx = build_index_entry(rec, 'bint')

    withkv = [u for u in idx['Data'] if u.get('key_values')]
    nokv = [u for u in idx['Data'] if not u.get('key_values')]
    check('key-value unit keeps Conditions',
          all('Conditions' in u for u in withkv), True)
    check('key-less units drop Conditions',
          any('Conditions' in u for u in nokv), False)
    check('trimmed units keep Analysis and step',
          all(set(u) == {'Analysis', 'step'} for u in nokv), True)
    # Steps differ, so all three OCV units survive the trim distinctly.
    check('distinct steps survive the trim',
          sorted(u['step'] for u in nokv), ['b21b', 'b22b', 'b23b'])

    # Conditions must remain intact in the detail bin — run detail reads them.
    ocv_plots = rec['metrics']['ocv']
    check_true('detail bin keeps per-plot conditions',
               all(p['conditions'].get('T_C') == 80.3 for p in ocv_plots.values()))
finally:
    shutil.rmtree(tmp)

print("sidecar compression")
tmp = tempfile.mkdtemp()
try:
    out = write_fixtures(tmp, list(FIXTURES.keys()))
    comp = build_detail_record(
        job_id='jz', sample_name='s', script='FC Polarization Curve',
        timestamp='t', input_files=[], output_dir=out)
    plain = build_detail_record(
        job_id='jz', sample_name='s', script='FC Polarization Curve',
        timestamp='t', input_files=[], output_dir=out,
        compress_sidecars=False)

    check('compressed by default', comp['sidecars']['encoding'],
          SIDECAR_ENCODING)
    check_true('carries plot count', comp['sidecars']['n_plots'] > 0)
    check('round-trip lossless', decode_sidecars(comp), plain['sidecars'])
    check('plain form decodes too', decode_sidecars(plain), plain['sidecars'])
    check('no sidecars decodes to empty', decode_sidecars({'schema': 2}), {})
    check_true('compression shrinks the record',
               len(json.dumps(comp, ensure_ascii=False).encode())
               < len(json.dumps(plain, ensure_ascii=False).encode()))

    stripped = strip_sidecars(comp, 'too big')
    check_true('strip removes sidecars', 'sidecars' not in stripped)
    check('strip records why', stripped['sidecars_omitted'], 'too big')
    check_true('strip keeps metrics', bool(stripped['metrics']))
    check_true('strip keeps conditions key', 'conditions' in stripped)
finally:
    shutil.rmtree(tmp)

print("sidecar selection")
SC = {
    'cleaning_cycles_a4': {'plot_type': 'cleaning_cycles',
                           'data': {'axes': [{'lines': [{'y': list(range(400))}]}]}},
    'cleaning_diag_a4': {'plot_type': 'cleaning_diagnostics',
                         'data': {'axes': [{'lines': [{'y': list(range(300))}]}]}},
    'polcurve_b14': {'plot_type': 'polcurve',
                     'data': {'axes': [{'lines': [{'y': [1, 2, 3]}]}]}},
    'eis_b14': {'plot_type': 'eis',
                'data': {'axes': [{'lines': [{'y': [1, 2]}]}]}},
}
buckets = sidecar_bucket_sizes(SC)
check('buckets accounted', sorted(buckets), ['cleaning', 'eis', 'polcurve'])
check_true('cleaning is largest', list(buckets)[0] == 'cleaning')
check_true('sizes ranked descending',
           list(sidecar_sizes(SC).values()) == sorted(
               sidecar_sizes(SC).values(), reverse=True))

kept, dropped = select_sidecars(SC, budget_bytes=10**9,
                                exclude_buckets={'cleaning'})
check('cleaning excluded by bucket', sorted(kept), ['eis_b14', 'polcurve_b14'])
check('both cleaning plots dropped', len(dropped), 2)

kept, dropped = select_sidecars(SC, budget_bytes=10**9, exclude_buckets=set())
check('nothing excluded when set empty', len(kept), 4)
check('nothing dropped', dropped, [])

# Budget retention keeps the smallest first, so a tight budget buys the most
# plots rather than one heavy one.
kept, dropped = select_sidecars(SC, budget_bytes=200, exclude_buckets=set())
check_true('small plots survive a tight budget',
           'eis_b14' in kept and 'polcurve_b14' in kept)
check_true('heavy plots dropped first', 'cleaning_cycles_a4' in dropped)

kept, dropped = select_sidecars(SC, budget_bytes=0, exclude_buckets=set())
check('zero budget keeps nothing', kept, {})
check('zero budget drops all', len(dropped), 4)

rec = {'schema': 2, 'metrics': {}}
attach_sidecars(rec, {'polcurve_b14': SC['polcurve_b14']})
check_true('attach adds sidecars', 'sidecars' in rec)
check_true('attach adds bucket accounting', 'sidecar_bytes_by_bucket' in rec)
check('attach round-trips', decode_sidecars(rec),
      {'polcurve_b14': SC['polcurve_b14']})
attach_sidecars(rec, {})
check_true('attach with none clears', 'sidecars' not in rec)
check_true('attach with none clears accounting',
           'sidecar_bytes_by_bucket' not in rec)

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

print("merge — detail record")

def _mk(plots, job, ts, script='X', sample='260421_GSMA-Qual-1'):
    """A detail record from a set of (plot_name, plot_type, readout) triples."""
    tmp = tempfile.mkdtemp()
    try:
        d = Path(tmp) / 'o' / '_plot_data'
        d.mkdir(parents=True)
        for name, pt, txt in plots:
            (d / f'{name}.json').write_text(json.dumps(sidecar(pt, [txt] if txt else [])))
        return build_detail_record(
            job_id=job, sample_name=sample, script=script, timestamp=ts,
            input_files=[f'{job}.csv'], output_dir=Path(tmp) / 'o')
    finally:
        shutil.rmtree(tmp)

XO = lambda st: (f'crossover_{st}_G_H2X_80.3Cell-100RH_0.4WH2-0.4WN2',
                 'crossover', '|j_xover| = 1.2 mA/cm2')
PC = lambda st: (f'polcurve_{st}_G_PolarizationCurve_80.3Cell-95RH_0.2WH2-0.2WA',
                 'polcurve', 'OCV = 0.95 V')
EI = lambda st: (f'eis_{st}_G_GEIS_80.3Cell-100RH_0.05WH2', 'eis', None)

check('touched_units reads (analysis, step)',
      touched_units(_mk([XO('a6'), PC('b17b')], 'j', 't')),
      {('crossover', 'a6'), ('polcurve', 'b17b')})

# Case 2.1 — crossover exists, then Full Analysis covering one of its steps.
first = _mk([XO('a6'), XO('b28b')], 'job-1', '2026-07-20T10:00:00Z')
second = _mk([PC('b17b'), EI('a12'), XO('a6')], 'job-2', '2026-07-23T10:00:00Z')
m = merge_detail_record(first, second)
check('2.1 untouched step survives', ('crossover', 'b28b') in touched_units(m), True)
check('2.1 covered step replaced', ('crossover', 'a6') in touched_units(m), True)
check('2.1 new buckets added',
      sorted(m['metrics'].keys()), ['crossover', 'eis', 'polcurve'])
check('2.1 crossover holds both steps', len(m['metrics']['crossover']), 2)

# Case 2.2 — Full Analysis exists, then a standalone crossover.
full = _mk([PC('b17b'), PC('c6'), EI('a12'), XO('a6'), XO('b28b')],
           'job-3', '2026-07-23T10:00:00Z')
xo = _mk([XO('a6')], 'job-4', '2026-07-24T09:00:00Z', script='H2 Crossover')
m2 = merge_detail_record(full, xo)
check('2.2 nothing else disturbed', touched_units(m2), touched_units(full))
check('2.2 polcurve intact', len(m2['metrics']['polcurve']), 2)
check('2.2 eis intact', len(m2['metrics']['eis']), 1)
check('2.2 sidecars follow the same rule', len(decode_sidecars(m2)), 5)

# Provenance must survive records written before the `jobs` field existed.
chain = merge_detail_record(merge_detail_record(first, second), xo)
check('provenance accumulates',
      [j['job_id'] for j in chain['jobs']], ['job-1', 'job-2', 'job-4'])
check('input_files union deduped',
      sorted(chain['input_files']), ['job-1.csv', 'job-2.csv', 'job-4.csv'])
check_true('stale size-omission marker cleared',
           'sidecars_omitted' not in chain)

# First write for a sample: nothing to merge into.
solo = merge_detail_record(None, first)
check('first write seeds provenance',
      [j['job_id'] for j in solo['jobs']], ['job-1'])

print("merge — index entry")

def _ent(job, ts, units, bin_id='BIN', script='Fuel Cell Full Analysis'):
    return {'job_id': job, 'sample_name': '260421_GSMA-Qual-1', 'script': script,
            'timestamp': ts, 'bin_id': bin_id, 'run_date': '2026-04-21',
            'Data': [{'Analysis': a, 'step': st,
                      **({'key_values': kv} if kv else {})} for a, st, kv in units]}

# The live duplicate: a degraded push then its retry.
e1 = _ent('j1', '2026-07-22T21:58:41Z',
          [('polcurve', 'b17b', {'OCV': 0.95}), ('cleaning', 'a4', None)])
e2 = _ent('j2', '2026-07-22T22:26:04Z',
          [('polcurve', 'b17b', {'OCV': 0.951}), ('cleaning', 'a4', None)])
me = merge_index_entry(e1, e2)
check('duplicate collapses to one set of units', len(me['Data']), 2)
check('retry value wins',
      [u['key_values']['OCV'] for u in me['Data'] if u['Analysis'] == 'polcurve'][0],
      0.951)
check('job_id is the last writer', me['job_id'], 'j2')
check('n_jobs counts contributors', me['n_jobs'], 2)
check('run_date is a property of the sample', me['run_date'], '2026-04-21')

# Additive: a new step is added rather than replacing.
e3 = _ent('j3', '2026-07-24T09:00:00Z',
          [('crossover', 'b39b', {'|j_xover|': 1.4})], script='H2 Crossover')
ma = merge_index_entry(me, e3)
check('new step appended', len(ma['Data']), 3)
check('script reflects the last writer', ma['script'], 'H2 Crossover')
check('n_jobs increments again', ma['n_jobs'], 3)

first_entry = merge_index_entry(None, e1)
check('first entry gets n_jobs 1', first_entry['n_jobs'], 1)

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)