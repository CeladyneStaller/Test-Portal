"""
Generic Plot Comparison Tool
============================
Reads sidecar JSON files written by `save_with_sidecar()` and overlays
them in a same-shape figure with distinct colors per source.

Auto-groups selected plots by `plot_type` (the filename stem) and runs
one comparison per group via the generic overlay renderer.

Custom per-type generators may be registered in COMPARISON_GENERATORS;
unknown types fall back to the generic overlay renderer.
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

from scripts.helpers.plot_compare import (
    find_sidecar, load_sidecar, render_overlay_comparison
)


def export_comparison_excel(items, plot_type, filepath):
    """Side-by-side Excel: one sheet per source axis, each sample's traces."""
    wb = Workbook()
    wb.remove(wb.active)

    hdr_font = Font(bold=True, color="FFFFFF")
    hdr_fill = PatternFill("solid", fgColor="1F4E79")
    label_fill = PatternFill("solid", fgColor="2E75B6")
    label_font = Font(bold=True, color="FFFFFF")

    ref_axes = items[0]['sidecar'].get('data', {}).get('axes', [])
    primary_axes = [a for a in ref_axes if not a.get('is_twin')]

    for ax_idx, ref_ax in enumerate(primary_axes):
        sheet_name = f'Axis {ax_idx + 1}'
        if ref_ax.get('title'):
            t = str(ref_ax['title'])[:25]
            sheet_name = f'A{ax_idx + 1} - {t}'[:31]
        ws = wb.create_sheet(sheet_name)

        col = 1
        for item in items:
            label = item['label']
            s_axes = [a for a in item['sidecar'].get('data', {}).get('axes', [])
                      if not a.get('is_twin')]
            if ax_idx >= len(s_axes):
                continue
            s_ad = s_axes[ax_idx]
            traces = s_ad.get('lines', []) + s_ad.get('scatters', [])
            if not traces:
                continue

            n_cols_per_sample = 2 * len(traces)
            col_end = col + n_cols_per_sample - 1

            cell = ws.cell(row=1, column=col, value=label)
            cell.font = label_font
            cell.fill = label_fill
            cell.alignment = Alignment(horizontal='center')
            if col_end > col:
                ws.merge_cells(start_row=1, start_column=col,
                               end_row=1, end_column=col_end)

            xlabel = s_ad.get('xlabel', 'x')
            ylabel = s_ad.get('ylabel', 'y')

            for ti, trace in enumerate(traces):
                trace_label = trace.get('label') or f'Series {ti + 1}'
                tcol = col + 2 * ti
                tc = ws.cell(row=2, column=tcol, value=trace_label)
                tc.font = Font(bold=True, italic=True, color="333333")
                tc.alignment = Alignment(horizontal='center')
                ws.merge_cells(start_row=2, start_column=tcol,
                               end_row=2, end_column=tcol + 1)
                for cc, h in enumerate([xlabel, ylabel]):
                    hc = ws.cell(row=3, column=tcol + cc, value=h)
                    hc.font = hdr_font
                    hc.fill = hdr_fill
                    hc.alignment = Alignment(horizontal='center')
                    ws.column_dimensions[get_column_letter(tcol + cc)].width = 14

                xs = trace.get('x', [])
                ys = trace.get('y', [])
                for ri, (x, y) in enumerate(zip(xs, ys)):
                    if x is not None and isinstance(x, (int, float)):
                        ws.cell(row=ri + 4, column=tcol, value=round(float(x), 6))
                    elif x is not None:
                        ws.cell(row=ri + 4, column=tcol, value=x)
                    if y is not None and isinstance(y, (int, float)):
                        ws.cell(row=ri + 4, column=tcol + 1, value=round(float(y), 6))
                    elif y is not None:
                        ws.cell(row=ri + 4, column=tcol + 1, value=y)

            col = col_end + 2

    if not wb.sheetnames:
        wb.create_sheet('Empty')
    wb.save(filepath)
    print(f"  Excel exported: {filepath}")


def _compare_generic(items, output_dir, params):
    """Default generator: overlay + Excel."""
    plot_type = items[0]['sidecar'].get('plot_type', 'plot')
    title = params.get('title') or f'{plot_type} Comparison'
    image_format = params.get('image_format', 'png')

    print(f"    {len(items)} samples:")
    for it in items:
        print(f"      - {it['label']}")

    out_files = []

    if image_format and image_format != 'none':
        plot_path = os.path.join(output_dir, f'{plot_type}_comparison.{image_format}')
        try:
            fig = render_overlay_comparison(items, save_path=plot_path, title=title)
            if fig:
                plt.close(fig)
                out_files.append(os.path.basename(plot_path))
        except Exception as e:
            print(f"    ✗ Failed to render comparison: {e}")
            import traceback; traceback.print_exc()

    try:
        xlsx_path = os.path.join(output_dir, f'{plot_type}_comparison.xlsx')
        export_comparison_excel(items, plot_type, xlsx_path)
        out_files.append(os.path.basename(xlsx_path))
    except Exception as e:
        print(f"    ✗ Failed to export Excel: {e}")
        import traceback; traceback.print_exc()

    return out_files


COMPARISON_GENERATORS = {
    # Override generic for specific plot types here if needed.
}


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

    by_type = {}
    skipped = []
    for src in sources:
        label = src.get('label', src.get('job_id', '?'))
        filename = src.get('filename', '')
        output_dir_s = src.get('output_dir', '')

        sidecar_path = find_sidecar(output_dir_s, filename)
        if sidecar_path is None:
            print(f"  ✗ {label} ({filename}): no sidecar")
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
            "No comparable plot data found. Re-run source analyses with the "
            "latest scripts that write sidecar JSON files."
            + (f"\nSkipped: {', '.join(skipped)}" if skipped else "")
        )

    print(f"\n  Grouped by plot type:")
    for ptype, items in by_type.items():
        print(f"    {ptype}: {len(items)} items")

    all_output_files = []
    for plot_type, items in by_type.items():
        if len(items) < 2:
            print(f"\n  Skipping '{plot_type}': only {len(items)} item")
            continue
        gen = COMPARISON_GENERATORS.get(plot_type, _compare_generic)
        print(f"\n  Generating '{plot_type}' comparison...")
        try:
            out_files = gen(items, str(out), p)
            all_output_files.extend(out_files)
        except Exception as e:
            print(f"    ✗ Generator failed: {e}")
            import traceback; traceback.print_exc()

    if not all_output_files:
        raise RuntimeError(
            "No comparison outputs generated. Check that ≥2 plots of the "
            "same type were selected and their sidecars are valid."
        )

    output_files = [f.name for f in out.iterdir() if f.is_file()]
    return {
        'status': 'success',
        'files_produced': output_files,
        'plot_types': list(by_type.keys()),
    }
