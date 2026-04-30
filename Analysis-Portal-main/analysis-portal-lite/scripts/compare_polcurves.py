"""
Polcurve Comparison Tool
========================
Generates overlay plots and combined Excel comparing the last cycle of
multiple polcurve analyses.

Reads `analysis_data.xlsx` from each job's output folder, extracts the
last cycle's polcurve data (raw V vs j), iR-corrected V if available,
and overlays all samples on a single plot.

Inputs (via params):
- sources: list of dicts with {job_id, label, output_dir}
- show_raw: bool (default True)
- show_irfree: bool (default True if iR data exists)
"""

import os
import json
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# ═══════════════════════════════════════════════════════════════════
#  Polcurve data extraction
# ═══════════════════════════════════════════════════════════════════

def find_polcurve_xlsx(output_dir):
    """Locate the analysis_data.xlsx in an output directory."""
    p = Path(output_dir)
    if not p.exists():
        return None
    for candidate in p.rglob('*.xlsx'):
        # Prefer 'analysis_data.xlsx' but accept any xlsx with polcurve data
        if 'analysis_data' in candidate.name.lower():
            return candidate
    # Fallback: any xlsx in directory
    xlsx_files = list(p.rglob('*.xlsx'))
    return xlsx_files[0] if xlsx_files else None


def _is_j_column(s):
    """Check if a header string represents a current density column."""
    if not s:
        return False
    cl = str(s).lower().strip()
    # Strip all brackets and spaces
    for ch in ' ()[]{}':
        cl = cl.replace(ch, '')
    # Match: jacm, jacm2, jacm², ja/cm, etc.
    return cl.startswith('jacm') or cl.startswith('ja/cm') or cl == 'j'


def _is_irfree_column(s):
    """Check if a header string represents an iR-free voltage column.
    Handles many formats: V_iR-free, V iR-free, V_irfree, V iRfree,
    iR-free, irfree, V_IR_free, etc.
    """
    if not s:
        return False
    cl = str(s).lower()
    # Strip non-alpha chars except keep meaning
    for ch in ' ()[]{}-_/\\.,':
        cl = cl.replace(ch, '')
    # Now look for 'irfree' substring
    return 'irfree' in cl


def _is_v_column(s):
    """Check if a header string represents a plain voltage column (not iR-free)."""
    if not s:
        return False
    if _is_irfree_column(s):
        return False
    cl = str(s).lower().strip()
    # Match: 'v', 'v (v)', 'v(v)', 'v [v]', 'voltage', 'voltage (v)'
    cl_compact = cl.replace(' ', '').replace('(', '').replace(')', '').replace('[','').replace(']','')
    return (cl_compact == 'v' or cl_compact == 'vv' or
            cl_compact.startswith('voltage'))


def extract_last_cycle_polcurve(xlsx_path):
    """
    Extract the last cycle's j, V, and optional iR-free V from a polcurve
    analysis Excel file.

    Handles two layout conventions:
      A) LONG format (electrolyzer_polcurve.py "Polcurve Data"):
         columns are [Cycle, Mode, V_setpoint, j, Step, Repeat]
         All cycles in one table, filtered by Cycle column.
      B) WIDE format (polcurve_analysis.py / fuelcell_analysis.py):
         "Pol Curve Data" or "Polcurve Data" with merged label rows
         and per-cycle column blocks (j, V, [HFR]).

    Returns
    -------
    dict with keys: 'j', 'V', 'V_irfree' (or None), 'cycle_num',
                    'source' (sheet/file info), or None if no data found.
    """
    try:
        wb = load_workbook(str(xlsx_path), read_only=True, data_only=True)
    except Exception as e:
        print(f"  ✗ Failed to load {xlsx_path}: {e}")
        return None

    polcurve_sheet_names = ['Polcurve Data', 'Pol Curve Data', 'Polcurve']
    sheet = None
    sheet_name = None
    for sn in polcurve_sheet_names:
        if sn in wb.sheetnames:
            sheet = wb[sn]
            sheet_name = sn
            break

    if sheet is None:
        for sn in wb.sheetnames:
            if 'pol' in sn.lower() and ('curve' in sn.lower() or 'data' in sn.lower()):
                sheet = wb[sn]
                sheet_name = sn
                break

    if sheet is None:
        wb.close()
        print(f"  ✗ No polcurve data sheet found in {xlsx_path.name}")
        return None

    rows = []
    for row in sheet.iter_rows(values_only=True):
        rows.append(list(row))
    wb.close()

    if len(rows) < 3:
        print(f"  ✗ Sheet '{sheet_name}' has too few rows in {xlsx_path.name}")
        return None

    # ── Detect format by checking first row ──
    first_row = rows[0]
    first_row_text = [str(c).lower() if c else '' for c in first_row]

    # LONG format check: header row contains both 'cycle' and 'j' columns
    has_cycle_col = any('cycle' in t and len(t) < 15 for t in first_row_text)
    has_j_col = any(_is_j_column(c) for c in first_row)

    if has_cycle_col and has_j_col:
        return _extract_long_format(rows, sheet_name, xlsx_path)
    else:
        return _extract_wide_format(rows, sheet_name, xlsx_path)


def _find_col(headers, *substrings):
    """Find first column index whose header contains all given substrings."""
    for ci, h in enumerate(headers):
        if h is None:
            continue
        hl = str(h).lower().replace(' ', '').replace('(', '').replace(')', '')
        if all(s in hl for s in substrings):
            return ci
    return None


def _extract_long_format(rows, sheet_name, xlsx_path):
    """Long format: one row per data point, with Cycle column."""
    headers = rows[0]
    cycle_col = _find_col(headers, 'cycle')
    j_col = None
    for ci, h in enumerate(headers):
        if _is_j_column(h):
            j_col = ci
            break
    v_col = _find_col(headers, 'v_setpoint') or _find_col(headers, 'voltage')
    if v_col is None:
        for ci, h in enumerate(headers):
            if _is_v_column(h):
                v_col = ci
                break
    virfree_col = None
    for ci, h in enumerate(headers):
        if _is_irfree_column(h):
            virfree_col = ci
            break

    if cycle_col is None or j_col is None or v_col is None:
        print(f"  ✗ Long-format columns not found (cycle={cycle_col}, "
              f"j={j_col}, V={v_col}) in {xlsx_path.name}")
        return None

    # Find max cycle number
    cycles_seen = set()
    for row in rows[1:]:
        if cycle_col < len(row) and row[cycle_col] is not None:
            try:
                cycles_seen.add(int(row[cycle_col]))
            except (ValueError, TypeError):
                pass

    if not cycles_seen:
        print(f"  ✗ No cycle numbers found in {xlsx_path.name}")
        return None

    last_cycle = max(cycles_seen)

    # Extract last cycle data
    j_vals, V_vals, Virf_vals = [], [], []
    for row in rows[1:]:
        if cycle_col >= len(row) or row[cycle_col] is None:
            continue
        try:
            cn = int(row[cycle_col])
        except (ValueError, TypeError):
            continue
        if cn != last_cycle:
            continue
        try:
            j_f = float(row[j_col])
            v_f = float(row[v_col])
            j_vals.append(j_f)
            V_vals.append(v_f)
            if virfree_col is not None and virfree_col < len(row) and row[virfree_col] is not None:
                Virf_vals.append(float(row[virfree_col]))
            else:
                Virf_vals.append(None)
        except (ValueError, TypeError):
            continue

    if not j_vals:
        print(f"  ✗ No data extracted for last cycle ({last_cycle}) in {xlsx_path.name}")
        return None

    V_irfree = (np.array(Virf_vals) if all(v is not None for v in Virf_vals)
                else None)

    j_arr = np.array(j_vals)
    V_arr = np.array(V_vals)
    order = np.argsort(j_arr)
    j_arr = j_arr[order]
    V_arr = V_arr[order]
    if V_irfree is not None:
        V_irfree = V_irfree[order]

    return {
        'j': j_arr,
        'V': V_arr,
        'V_irfree': V_irfree,
        'cycle_num': last_cycle,
        'source': f"{xlsx_path.name} ({sheet_name}, long-format, cycle {last_cycle})",
    }


def _extract_wide_format(rows, sheet_name, xlsx_path):
    """Wide format: column blocks per cycle with merged label rows above."""
    # Find header row (contains 'j (A/cm²)')
    header_row_idx = None
    for i, row in enumerate(rows[:8]):
        for cell in row:
            if _is_j_column(cell):
                header_row_idx = i
                break
        if header_row_idx is not None:
            break

    if header_row_idx is None:
        print(f"  ✗ Could not find j column header in {xlsx_path.name}")
        return None

    headers = rows[header_row_idx]

    # Find all 'j (A/cm²)' columns — each starts a new cycle block
    j_cols = [ci for ci, h in enumerate(headers) if _is_j_column(h)]

    if not j_cols:
        print(f"  ✗ No j columns found in {xlsx_path.name}")
        return None

    last_j_col = j_cols[-1]
    cycle_idx = len(j_cols)
    next_j_col = (j_cols[j_cols.index(last_j_col) + 1]
                  if j_cols.index(last_j_col) + 1 < len(j_cols) else len(headers))

    v_col = None
    virfree_col = None
    for ci in range(last_j_col + 1, next_j_col):
        if ci >= len(headers):
            break
        h = headers[ci]
        if _is_v_column(h) and v_col is None:
            v_col = ci
        if _is_irfree_column(h):
            virfree_col = ci

    if v_col is None:
        v_col = last_j_col + 2  # fallback: j, j_mA, V

    j_vals, V_vals, Virf_vals = [], [], []
    for row in rows[header_row_idx + 1:]:
        if last_j_col >= len(row) or v_col >= len(row):
            continue
        j_val = row[last_j_col]
        v_val = row[v_col]
        if j_val is None or v_val is None:
            continue
        try:
            j_f = float(j_val)
            v_f = float(v_val)
            j_vals.append(j_f)
            V_vals.append(v_f)
            if virfree_col is not None and virfree_col < len(row):
                vir = row[virfree_col]
                Virf_vals.append(float(vir) if vir is not None else None)
            else:
                Virf_vals.append(None)
        except (ValueError, TypeError):
            continue

    if not j_vals:
        print(f"  ✗ No numeric polcurve data extracted from {xlsx_path.name}")
        return None

    V_irfree = (np.array(Virf_vals) if all(v is not None for v in Virf_vals)
                else None)

    j_arr = np.array(j_vals)
    V_arr = np.array(V_vals)
    order = np.argsort(j_arr)
    j_arr = j_arr[order]
    V_arr = V_arr[order]
    if V_irfree is not None:
        V_irfree = V_irfree[order]

    return {
        'j': j_arr,
        'V': V_arr,
        'V_irfree': V_irfree,
        'cycle_num': cycle_idx,
        'source': f"{xlsx_path.name} ({sheet_name}, wide-format, cycle {cycle_idx})",
    }


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_polcurve_overlay(samples, show_raw=True, show_irfree=True,
                          title="Polcurve Comparison", save_path=None):
    """
    Overlay polcurves from multiple samples.

    samples : list of dicts with 'label', 'j', 'V', 'V_irfree'
    """
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)

    # Color cycle (use tab10 + custom for >10)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(samples))]

    has_any_irfree = any(s.get('V_irfree') is not None for s in samples)
    plot_irfree = show_irfree and has_any_irfree

    for i, s in enumerate(samples):
        c = colors[i]
        label = s.get('label', f'Sample {i+1}')

        if show_raw:
            raw_label = label if not plot_irfree else f'{label} (raw)'
            ax.plot(s['j'], s['V'], 'o-', color=c, ms=5, lw=1.5,
                    label=raw_label, alpha=0.85)

        if plot_irfree and s.get('V_irfree') is not None:
            ax.plot(s['j'], s['V_irfree'], 's--', color=c, ms=4, lw=1.2,
                    label=f'{label} (iR-free)', alpha=0.85,
                    markerfacecolor='none')

    ax.set_xlabel('Current density  j  [A/cm²]', fontsize=12)
    ax.set_ylabel('Cell voltage  [V]', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.set_xlim(left=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
    else:
        plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════
#  Excel export
# ═══════════════════════════════════════════════════════════════════

def export_comparison_excel(samples, filepath, show_raw=True, show_irfree=True):
    """Write all polcurve data side-by-side for easy comparison."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Polcurve Comparison"

    hdr_font = Font(bold=True, color="FFFFFF")
    hdr_fill = PatternFill("solid", fgColor="1F4E79")
    label_fill = PatternFill("solid", fgColor="2E75B6")
    label_font = Font(bold=True, color="FFFFFF")

    has_any_irfree = any(s.get('V_irfree') is not None for s in samples)
    plot_irfree = show_irfree and has_any_irfree

    col = 1
    for s in samples:
        label = s.get('label', 'Sample')
        n_cols_per_sample = 3 if plot_irfree else 2
        col_end = col + n_cols_per_sample - 1

        # Sample label spanning columns
        cell = ws.cell(row=1, column=col, value=label)
        cell.font = label_font
        cell.fill = label_fill
        cell.alignment = Alignment(horizontal='center')
        if col_end > col:
            ws.merge_cells(start_row=1, start_column=col,
                           end_row=1, end_column=col_end)

        # Headers
        headers = ['j (A/cm²)', 'V (V)']
        if plot_irfree:
            headers.append('V_iR-free (V)')
        for hi, h in enumerate(headers):
            cell = ws.cell(row=2, column=col + hi, value=h)
            cell.font = hdr_font
            cell.fill = hdr_fill
            cell.alignment = Alignment(horizontal='center')
            ws.column_dimensions[get_column_letter(col + hi)].width = 14

        # Data
        n_pts = len(s['j'])
        for ri in range(n_pts):
            ws.cell(row=ri + 3, column=col, value=round(float(s['j'][ri]), 6))
            ws.cell(row=ri + 3, column=col + 1, value=round(float(s['V'][ri]), 5))
            if plot_irfree:
                if s.get('V_irfree') is not None and ri < len(s['V_irfree']):
                    v = s['V_irfree'][ri]
                    if v is not None and not np.isnan(v):
                        ws.cell(row=ri + 3, column=col + 2, value=round(float(v), 5))

        col = col_end + 2  # gap column

    wb.save(filepath)
    print(f"  Excel exported: {filepath}")


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

def run(input_dir, output_dir, params=None):
    """
    Main entry. Expects params to contain 'sources' (JSON string of list).

    Each source is: {"job_id": str, "label": str, "output_dir": str}
    """
    p = params or {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Parse sources
    sources_raw = p.get('sources', '[]')
    if isinstance(sources_raw, str):
        try:
            sources = json.loads(sources_raw)
        except Exception as e:
            raise RuntimeError(f"Invalid sources JSON: {e}")
    else:
        sources = sources_raw

    if not sources or len(sources) < 2:
        raise RuntimeError(
            f"Need at least 2 sources to compare (got {len(sources) if sources else 0})"
        )

    show_raw = str(p.get('show_raw', 'true')).lower() in ('true', '1', 'yes')
    show_irfree = str(p.get('show_irfree', 'true')).lower() in ('true', '1', 'yes')
    title = p.get('title', 'Polcurve Comparison')
    image_format = p.get('image_format', 'png')

    print(f"\n{'='*60}")
    print(f"  Polcurve Comparison")
    print(f"{'='*60}")
    print(f"  Sources: {len(sources)}")
    for s in sources:
        print(f"    - {s.get('label', s.get('job_id', '?'))}: {s.get('output_dir', '?')}")
    print(f"  Show raw: {show_raw}")
    print(f"  Show iR-free: {show_irfree}")
    print()

    # Extract data from each source
    samples = []
    for s in sources:
        job_id = s.get('job_id', 'unknown')
        label = s.get('label', job_id)
        output_dir_s = s.get('output_dir', '')

        print(f"  Loading {label}...")
        xlsx = find_polcurve_xlsx(output_dir_s)
        if xlsx is None:
            print(f"  ✗ No analysis_data.xlsx found in {output_dir_s}")
            continue

        data = extract_last_cycle_polcurve(xlsx)
        if data is None:
            continue

        data['label'] = label
        data['job_id'] = job_id
        samples.append(data)
        print(f"  ✓ {label}: {len(data['j'])} points, "
              f"j = {data['j'].min():.3f}-{data['j'].max():.3f} A/cm², "
              f"V_irfree {'available' if data['V_irfree'] is not None else 'not available'}")

    if len(samples) < 2:
        raise RuntimeError(
            f"Could not extract polcurve data from enough sources "
            f"(got {len(samples)} valid out of {len(sources)})"
        )

    # Generate plot
    if image_format and image_format != 'none':
        plot_path = out / f'polcurve_comparison.{image_format}'
        plot_polcurve_overlay(samples, show_raw=show_raw, show_irfree=show_irfree,
                              title=title, save_path=str(plot_path))
        plt.close('all')

    # Generate Excel
    xlsx_path = out / 'polcurve_comparison.xlsx'
    export_comparison_excel(samples, str(xlsx_path),
                            show_raw=show_raw, show_irfree=show_irfree)

    # Collect output files
    output_files = [f.name for f in out.iterdir() if f.is_file()]

    return {
        'status': 'success',
        'files_produced': output_files,
        'n_samples': len(samples),
        'sample_labels': [s['label'] for s in samples],
    }
