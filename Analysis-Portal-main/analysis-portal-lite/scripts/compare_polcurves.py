"""
Plot Comparison Tool — sidecar JSON edition
============================================
Reads sidecar JSON files (`_plot_data/{plotname}.json`) written by the
plotting functions in electrolyzer_polcurve.py / polcurve_analysis.py /
fuelcell_analysis.py. Each JSON has the underlying numerical data for one
plot, so the comparison can faithfully overlay PNGs from multiple analyses.

Auto-groups selected plots by `plot_type` and runs the appropriate
comparison generator for each group. Currently supports: polcurve.
"""

import os
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# ═══════════════════════════════════════════════════════════════════
#  Sidecar loading
# ═══════════════════════════════════════════════════════════════════

def find_sidecar(output_dir, filename):
    """Locate sidecar JSON for a PNG: output_dir/_plot_data/{stem}.json"""
    p = Path(output_dir)
    if not p.exists():
        return None
    base = Path(filename).stem
    sidecar = p / '_plot_data' / f'{base}.json'
    if sidecar.exists():
        return sidecar
    sidecar2 = p / '_plot_data' / f'{filename}.json'
    if sidecar2.exists():
        return sidecar2
    return None


def load_sidecar(sidecar_path):
    try:
        with open(sidecar_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ✗ Failed to load sidecar {sidecar_path}: {e}")
        return None


def get_representative_cycle(sidecar):
    """Get the representative cycle dict from a polcurve sidecar."""
    data = sidecar.get('data', {})
    cycles = data.get('cycles', [])
    if not cycles:
        return None
    rep_idx = data.get('representative_cycle_idx')
    if rep_idx is not None and 0 <= rep_idx < len(cycles):
        return cycles[rep_idx]
    return cycles[-1]


# ═══════════════════════════════════════════════════════════════════
#  Polcurve overlay plot + Excel
# ═══════════════════════════════════════════════════════════════════

def plot_polcurve_overlay(samples, show_raw=True, show_irfree=True,
                          title="Polcurve Comparison", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(samples))]

    has_any_irfree = any(s['cycle'].get('V_irfree') is not None for s in samples)
    plot_irfree = show_irfree and has_any_irfree

    for i, s in enumerate(samples):
        c = colors[i]
        label = s.get('label', f'Sample {i+1}')
        cyc = s['cycle']

        j = np.array(cyc.get('j', []))
        V = np.array(cyc.get('V', []))
        if len(j) == 0:
            continue

        order = np.argsort(j)
        j = j[order]
        V = V[order]

        if show_raw:
            raw_label = label if not plot_irfree else f'{label} (raw)'
            ax.plot(j, V, 'o-', color=c, ms=5, lw=1.5,
                    label=raw_label, alpha=0.85)

        if plot_irfree and cyc.get('V_irfree') is not None:
            V_irf = np.array(cyc['V_irfree'])
            if len(V_irf) == len(j):
                V_irf = V_irf[order]
                ax.plot(j, V_irf, 's--', color=c, ms=4, lw=1.2,
                        label=f'{label} (iR-free)', alpha=0.85,
                        markerfacecolor='none')

    xlabel = 'Current density  j  [A/cm²]'
    ylabel = 'Cell voltage  [V]'
    if samples and samples[0].get('metadata'):
        md = samples[0]['metadata']
        xlabel = md.get('xlabel', xlabel)
        ylabel = md.get('ylabel', ylabel)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
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


def export_polcurve_excel(samples, filepath, plot_irfree=False):
    wb = Workbook()
    ws = wb.active
    ws.title = "Polcurve Comparison"

    hdr_font = Font(bold=True, color="FFFFFF")
    hdr_fill = PatternFill("solid", fgColor="1F4E79")
    label_fill = PatternFill("solid", fgColor="2E75B6")
    label_font = Font(bold=True, color="FFFFFF")

    col = 1
    for s in samples:
        label = s.get('label', 'Sample')
        cyc = s['cycle']
        n_cols_per_sample = 2 + (1 if plot_irfree and cyc.get('V_irfree') is not None else 0)
        col_end = col + n_cols_per_sample - 1

        cell = ws.cell(row=1, column=col, value=label)
        cell.font = label_font
        cell.fill = label_fill
        cell.alignment = Alignment(horizontal='center')
        if col_end > col:
            ws.merge_cells(start_row=1, start_column=col,
                           end_row=1, end_column=col_end)

        headers = ['j (A/cm²)', 'V (V)']
        if plot_irfree and cyc.get('V_irfree') is not None:
            headers.append('V_iR-free (V)')
        for hi, h in enumerate(headers):
            cell = ws.cell(row=2, column=col + hi, value=h)
            cell.font = hdr_font
            cell.fill = hdr_fill
            cell.alignment = Alignment(horizontal='center')
            ws.column_dimensions[get_column_letter(col + hi)].width = 14

        j_arr = cyc.get('j', [])
        V_arr = cyc.get('V', [])
        Virf_arr = cyc.get('V_irfree') or []

        if j_arr:
            order = sorted(range(len(j_arr)), key=lambda i: j_arr[i])
            for ri, i in enumerate(order):
                ws.cell(row=ri + 3, column=col, value=round(float(j_arr[i]), 6))
                ws.cell(row=ri + 3, column=col + 1, value=round(float(V_arr[i]), 5))
                if plot_irfree and Virf_arr and i < len(Virf_arr):
                    v = Virf_arr[i]
                    if v is not None:
                        ws.cell(row=ri + 3, column=col + 2,
                                value=round(float(v), 5))

        col = col_end + 2

    wb.save(filepath)
    print(f"  Excel exported: {filepath}")


# ═══════════════════════════════════════════════════════════════════
#  Comparison generators by plot type
# ═══════════════════════════════════════════════════════════════════

def _compare_polcurves(items, output_dir, params):
    """items: [{label, filename, sidecar}]"""
    show_raw = str(params.get('show_raw', 'true')).lower() in ('true', '1', 'yes')
    show_irfree = str(params.get('show_irfree', 'true')).lower() in ('true', '1', 'yes')
    title = params.get('title', 'Polcurve Comparison')
    image_format = params.get('image_format', 'png')

    samples = []
    for item in items:
        label = item['label']
        sidecar = item['sidecar']
        rep_cycle = get_representative_cycle(sidecar)
        if rep_cycle is None:
            print(f"    ✗ {label}: no cycle data")
            continue
        samples.append({
            'label': label,
            'cycle': rep_cycle,
            'metadata': sidecar.get('metadata', {}),
        })
        print(f"    ✓ {label}: cycle {rep_cycle.get('cycle_num', '?')} "
              f"({len(rep_cycle.get('j', []))} pts), "
              f"V_irfree {'yes' if rep_cycle.get('V_irfree') else 'no'}")

    if len(samples) < 2:
        return []

    out_files = []
    if image_format and image_format != 'none':
        plot_path = os.path.join(output_dir, f'polcurve_comparison.{image_format}')
        plot_polcurve_overlay(samples, show_raw=show_raw, show_irfree=show_irfree,
                              title=title, save_path=plot_path)
        plt.close('all')
        out_files.append(os.path.basename(plot_path))

    plot_irfree = show_irfree and any(s['cycle'].get('V_irfree') is not None for s in samples)
    xlsx_path = os.path.join(output_dir, 'polcurve_comparison.xlsx')
    export_polcurve_excel(samples, xlsx_path, plot_irfree=plot_irfree)
    out_files.append(os.path.basename(xlsx_path))

    return out_files


COMPARISON_GENERATORS = {
    'polcurve': _compare_polcurves,
}


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

def run(input_dir, output_dir, params=None):
    p = params or {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sources_raw = p.get('sources', '[]')
    if isinstance(sources_raw, str):
        try:
            sources = json.loads(sources_raw)
        except Exception as e:
            raise RuntimeError(f"Invalid sources JSON: {e}")
    else:
        sources = sources_raw

    if not sources:
        raise RuntimeError("No sources provided")

    print(f"\n{'='*60}")
    print(f"  Plot Comparison")
    print(f"{'='*60}")
    print(f"  Total selected: {len(sources)}")

    # Load all sidecars and group by plot_type
    by_type = {}
    skipped = []
    for src in sources:
        label = src.get('label', src.get('job_id', '?'))
        filename = src.get('filename', '')
        output_dir_s = src.get('output_dir', '')

        sidecar_path = find_sidecar(output_dir_s, filename)
        if sidecar_path is None:
            print(f"  ✗ {label} ({filename}): no sidecar in {output_dir_s}/_plot_data/")
            skipped.append(f"{label}/{filename}")
            continue

        sidecar = load_sidecar(sidecar_path)
        if sidecar is None:
            skipped.append(f"{label}/{filename}")
            continue

        plot_type = sidecar.get('plot_type', 'unknown')
        by_type.setdefault(plot_type, []).append({
            'label': label,
            'filename': filename,
            'sidecar': sidecar,
        })

    if not by_type:
        raise RuntimeError(
            "No comparable plot data found. Re-run the source analyses with "
            "the latest scripts that write sidecar JSON files."
            + (f"\nSkipped: {', '.join(skipped)}" if skipped else "")
        )

    print(f"\n  Grouped by plot type:")
    for ptype, items in by_type.items():
        print(f"    {ptype}: {len(items)} items")

    all_output_files = []
    for plot_type, items in by_type.items():
        if len(items) < 2:
            print(f"\n  Skipping '{plot_type}': only {len(items)} item, need ≥2")
            continue
        gen = COMPARISON_GENERATORS.get(plot_type)
        if gen is None:
            print(f"\n  Skipping '{plot_type}': no comparison generator yet")
            continue
        print(f"\n  Generating comparison for '{plot_type}'...")
        out_files = gen(items, str(out), p)
        all_output_files.extend(out_files)

    if not all_output_files:
        raise RuntimeError(
            "No comparison outputs generated. Need ≥2 plots of the same "
            "supported type (currently: " + ", ".join(COMPARISON_GENERATORS.keys()) + ")"
        )

    output_files = [f.name for f in out.iterdir() if f.is_file()]
    return {
        'status': 'success',
        'files_produced': output_files,
        'plot_types': list(by_type.keys()),
    }
