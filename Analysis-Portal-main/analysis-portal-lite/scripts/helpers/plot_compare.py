"""
Generic plot sidecar writer + comparison helper.

This module enables the comparison feature for ANY plot in the portal
without per-plot code changes. Two patterns:

  1. **Save with sidecar:**

         from scripts.helpers.plot_compare import save_with_sidecar
         save_with_sidecar(fig, '/path/to/plot.png', plot_type='polcurve',
                           metadata={...})

     This both saves the figure AND writes a JSON sidecar containing the
     data extracted from each axis (lines, scatters, bars, scales, labels).

  2. **Auto-extract from existing fig.savefig calls:**

         Replace `fig.savefig(path, ...)` with
         `save_with_sidecar(fig, path, ...)` anywhere a sidecar is wanted.
         The plot_type defaults to the filename stem.

For the comparison side: `extract_axes_data(fig)` returns a list of axis
dicts that are JSON-safe and contain everything needed to overlay multiple
samples in a same-shape figure.
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle


def _to_jsonable(obj):
    """Convert numpy / matplotlib types to JSON-safe values."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    return str(obj)


def _color_to_rgba(c):
    """Convert any matplotlib color to (r,g,b,a) tuple of floats."""
    try:
        from matplotlib.colors import to_rgba
        rgba = to_rgba(c)
        return [float(x) for x in rgba]
    except Exception:
        return [0.0, 0.0, 0.0, 1.0]


def extract_axes_data(fig):
    """
    Extract plot data from every axis in a figure.

    Returns a list of axis dicts:
        [{
          'index': 0,
          'rows': 1, 'cols': 4,  # subplot grid layout
          'subplot_pos': (1, 1),  # row, col within grid (1-indexed)
          'title': str,
          'xlabel': str, 'ylabel': str,
          'xscale': 'linear'|'log', 'yscale': 'linear'|'log',
          'xlim': [lo, hi], 'ylim': [lo, hi],
          'lines': [
            {'x': [...], 'y': [...], 'label': str,
             'color': [r,g,b,a], 'linestyle': str, 'linewidth': float,
             'marker': str, 'markersize': float, 'alpha': float}
          ],
          'scatters': [{'x': [...], 'y': [...], 'label': str,
                        'color': [...], 'sizes': [...], 'marker': str}],
          'bars': [{'x': [...], 'heights': [...], 'widths': [...],
                    'color': [...], 'label': str}],
          'axhlines': [{'y': float, 'color': [...], 'linestyle': str}],
          'axvlines': [{'x': float, 'color': [...], 'linestyle': str}],
          'has_twin': bool,
          'is_twin': bool,
        }]
    """
    axes = fig.get_axes()
    n_axes = len(axes)
    if n_axes == 0:
        return []

    # Detect subplot grid layout from the first axis
    try:
        gs = axes[0].get_gridspec()
        n_rows = gs.nrows
        n_cols = gs.ncols
    except Exception:
        n_rows = 1
        n_cols = n_axes

    # Identify twin axes (shared x/y)
    twin_pairs = {}
    for i, ax in enumerate(axes):
        for j, ax2 in enumerate(axes):
            if i >= j:
                continue
            try:
                if ax.get_shared_x_axes().joined(ax, ax2) and ax.bbox.bounds == ax2.bbox.bounds:
                    twin_pairs[j] = i  # ax2 is twin of ax
            except Exception:
                pass

    out = []
    for i, ax in enumerate(axes):
        try:
            ss = ax.get_subplotspec()
            row, col = ss.rowspan.start + 1, ss.colspan.start + 1
        except Exception:
            row, col = 1, i + 1

        ad = {
            'index': i,
            'rows': n_rows,
            'cols': n_cols,
            'subplot_pos': [row, col],
            'title': str(ax.get_title()) or None,
            'xlabel': str(ax.get_xlabel()) or None,
            'ylabel': str(ax.get_ylabel()) or None,
            'xscale': ax.get_xscale(),
            'yscale': ax.get_yscale(),
            'xlim': list(ax.get_xlim()),
            'ylim': list(ax.get_ylim()),
            'lines': [],
            'scatters': [],
            'bars': [],
            'axhlines': [],
            'axvlines': [],
            'is_twin': i in twin_pairs,
            'twin_of': twin_pairs.get(i),
            'has_legend': ax.get_legend() is not None,
        }

        # Lines (from ax.plot())
        for line in ax.get_lines():
            xd = line.get_xdata()
            yd = line.get_ydata()
            # Skip axhline/axvline disguised as lines (length 2 with constant val)
            try:
                if len(xd) == 2 and len(yd) == 2:
                    if yd[0] == yd[1] and xd[0] != xd[1]:
                        ad['axhlines'].append({
                            'y': float(yd[0]),
                            'color': _color_to_rgba(line.get_color()),
                            'linestyle': line.get_linestyle(),
                        })
                        continue
                    if xd[0] == xd[1] and yd[0] != yd[1]:
                        ad['axvlines'].append({
                            'x': float(xd[0]),
                            'color': _color_to_rgba(line.get_color()),
                            'linestyle': line.get_linestyle(),
                        })
                        continue
            except Exception:
                pass

            ad['lines'].append({
                'x': _to_jsonable(np.asarray(xd)),
                'y': _to_jsonable(np.asarray(yd)),
                'label': line.get_label() if not line.get_label().startswith('_') else None,
                'color': _color_to_rgba(line.get_color()),
                'linestyle': line.get_linestyle(),
                'linewidth': float(line.get_linewidth()),
                'marker': str(line.get_marker()),
                'markersize': float(line.get_markersize()),
                'alpha': float(line.get_alpha()) if line.get_alpha() is not None else 1.0,
            })

        # Scatter plots
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                offsets = coll.get_offsets()
                if offsets is not None and len(offsets) > 0:
                    arr = np.asarray(offsets)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        x = arr[:, 0]
                        y = arr[:, 1]
                        sizes = coll.get_sizes()
                        ad['scatters'].append({
                            'x': _to_jsonable(x),
                            'y': _to_jsonable(y),
                            'sizes': _to_jsonable(sizes),
                            'label': coll.get_label() if coll.get_label() and
                                     not coll.get_label().startswith('_') else None,
                        })

        # Bars / patches
        bar_xs, bar_heights, bar_widths, bar_colors = [], [], [], []
        for patch in ax.patches:
            if isinstance(patch, Rectangle):
                x = patch.get_x() + patch.get_width() / 2
                bar_xs.append(float(x))
                bar_heights.append(float(patch.get_height()))
                bar_widths.append(float(patch.get_width()))
                bar_colors.append(_color_to_rgba(patch.get_facecolor()))
        if bar_xs:
            ad['bars'].append({
                'x': bar_xs,
                'heights': bar_heights,
                'widths': bar_widths,
                'color': bar_colors,
            })

        out.append(ad)

    return out


def save_with_sidecar(fig, save_path, plot_type=None, metadata=None,
                      sidecar_subdir='_plot_data', **savefig_kwargs):
    """
    Save a matplotlib figure AND write a sidecar JSON with extracted plot data.

    Drop-in replacement for fig.savefig() — same args plus optional
    plot_type and metadata dicts.

    plot_type defaults to the filename stem (e.g. 'polcurve' for 'polcurve.png').
    """
    if not save_path:
        return
    # Save the figure first
    if 'bbox_inches' not in savefig_kwargs:
        savefig_kwargs['bbox_inches'] = 'tight'
    fig.savefig(save_path, **savefig_kwargs)

    # Determine plot_type
    if plot_type is None:
        plot_type = os.path.splitext(os.path.basename(save_path))[0]

    # Extract axis data
    try:
        axes_data = extract_axes_data(fig)
    except Exception as e:
        print(f"  Warning: failed to extract axes data for sidecar: {e}")
        axes_data = []

    # Capture suptitle if present
    sup = None
    try:
        if fig._suptitle is not None:
            sup = fig._suptitle.get_text()
    except Exception:
        pass

    payload = {
        'plot_type': plot_type,
        'data': {
            'axes': axes_data,
            'figsize': list(fig.get_size_inches()),
            'suptitle': sup,
        },
        'metadata': _to_jsonable(metadata or {}),
    }

    # Write sidecar to subdirectory
    plot_dir = os.path.dirname(save_path)
    plot_name = os.path.splitext(os.path.basename(save_path))[0]
    sidecar_dir = os.path.join(plot_dir, sidecar_subdir)
    os.makedirs(sidecar_dir, exist_ok=True)
    sidecar_path = os.path.join(sidecar_dir, f'{plot_name}.json')

    try:
        with open(sidecar_path, 'w') as f:
            json.dump(payload, f, indent=2, allow_nan=False, default=str)
    except Exception as e:
        print(f"  Warning: failed to write sidecar JSON {sidecar_path}: {e}")


# ═══════════════════════════════════════════════════════════════════
#  Generic comparison renderer
# ═══════════════════════════════════════════════════════════════════

def find_sidecar(output_dir, filename):
    """Locate sidecar JSON for a saved plot.

    Handles filenames containing subfolder paths
    (e.g. 'polcurve/foo.png' → 'output_dir/polcurve/_plot_data/foo.json').
    """
    p = Path(output_dir)
    if not p.exists():
        return None
    fn_path = Path(filename)
    parent = fn_path.parent
    base = fn_path.stem
    if str(parent) != '.':
        candidate = p / parent / '_plot_data' / f'{base}.json'
        if candidate.exists():
            return candidate
    candidate = p / '_plot_data' / f'{base}.json'
    if candidate.exists():
        return candidate
    for cand in p.rglob(f'{base}.json'):
        if '_plot_data' in cand.parts:
            return cand
    return None


def load_sidecar(sidecar_path):
    try:
        with open(sidecar_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ✗ Failed to load sidecar {sidecar_path}: {e}")
        return None


def render_overlay_comparison(items, save_path=None, title=None,
                               sample_label_in_legend=True):
    """
    Generic comparison renderer that overlays N samples' axis data on a
    same-shape figure.

    items : list of {'label': str, 'sidecar': dict}
    """
    if not items:
        return None

    # Use first sample as the layout reference
    ref = items[0]['sidecar'].get('data', {})
    ref_axes = ref.get('axes', [])
    if not ref_axes:
        print("  No axes data in reference sidecar")
        return None

    # Filter out twin axes (they share position with their parent)
    primary_axes = [a for a in ref_axes if not a.get('is_twin', False)]

    # Filter out axes with no plot data (e.g. colorbars, empty placeholders)
    def _has_data(a):
        return (len(a.get('lines', [])) > 0 or
                len(a.get('scatters', [])) > 0 or
                len(a.get('bars', [])) > 0)
    primary_axes = [a for a in primary_axes if _has_data(a)]

    if not primary_axes:
        print("  No axes with data in reference sidecar")
        return None

    n_primary = len(primary_axes)

    # Compute actual layout from filtered axes positions
    rows_used = sorted(set(a.get('subplot_pos', [1, 1])[0] for a in primary_axes))
    cols_used = sorted(set(a.get('subplot_pos', [1, 1])[1] for a in primary_axes))
    n_rows_eff = len(rows_used)
    n_cols_eff = len(cols_used)
    # Fallback: if all in same row use 1 × n
    if n_rows_eff == 1 and n_cols_eff == 1 and n_primary > 1:
        n_cols_eff = n_primary
    figsize = ref.get('figsize', [4 * n_cols_eff, 4 * n_rows_eff])
    # Cap figsize to reasonable bounds
    figsize = [min(figsize[0], 5 * n_cols_eff + 2),
               min(figsize[1], 5 * n_rows_eff + 1)]

    fig, axes = plt.subplots(n_rows_eff, n_cols_eff, figsize=tuple(figsize),
                             squeeze=False)

    # Map original (row, col) → (new_row, new_col) based on used positions
    row_map = {orig: new for new, orig in enumerate(rows_used)}
    col_map = {orig: new for new, orig in enumerate(cols_used)}

    flat_axes = []
    for ax_data in primary_axes:
        row, col = ax_data.get('subplot_pos', [1, 1])
        new_r = row_map[row]
        new_c = col_map[col]
        flat_axes.append(axes[new_r, new_c])

    # Hide unused subplot slots
    used = set()
    for a in primary_axes:
        row, col = a.get('subplot_pos', [1, 1])
        used.add((row_map[row], col_map[col]))
    for r in range(n_rows_eff):
        for c in range(n_cols_eff):
            if (r, c) not in used:
                axes[r, c].axis('off')

    # Distinct color per sample
    cmap = plt.get_cmap('tab10')
    sample_colors = [cmap(i % 10) for i in range(len(items))]

    # Overlay each sample's primary axes
    for sample_idx, item in enumerate(items):
        s_label = item['label']
        s_sidecar = item['sidecar'].get('data', {})
        s_axes = [a for a in s_sidecar.get('axes', []) if not a.get('is_twin', False)]

        # Match s_axes to primary_axes by index
        for ai, ax_target in enumerate(flat_axes):
            if ai >= len(s_axes):
                continue
            s_ad = s_axes[ai]
            color = sample_colors[sample_idx]

            visible_lines = list(s_ad.get('lines', []))

            # Detect if axis has many lines of same kind (e.g. cycles overlay).
            # If >5 unlabeled lines, treat as "many cycles" — show one label
            # per sample for the first line only, draw all in same color/style.
            many_lines = len(visible_lines) > 5

            for li, ln in enumerate(visible_lines):
                if many_lines:
                    # Only label first line, no style variation
                    label = s_label if (ai == 0 and li == 0) else None
                    ls = ln.get('linestyle', '-')
                    if ls in ('None', 'none', ''):
                        ls = '-'
                    alpha = 0.4 if li > 0 else 0.9  # de-emphasize secondary lines
                else:
                    line_styles = ['-', '--', '-.', ':']
                    ls = line_styles[li % len(line_styles)] if len(visible_lines) > 1 else '-'
                    if ai == 0 and li == 0:
                        label = s_label
                    elif ai == 0 and len(visible_lines) > 1:
                        orig = ln.get('label') or f'series {li+1}'
                        if orig and not orig.startswith('_'):
                            label = f'{s_label} ({orig})'
                        else:
                            label = None
                    else:
                        label = None
                    alpha = ln.get('alpha', 1.0)

                marker = ln.get('marker', 'None')
                if marker in ('None', 'none', None, ''):
                    marker = None

                ax_target.plot(
                    ln['x'], ln['y'],
                    color=color,
                    linestyle=ls,
                    linewidth=ln.get('linewidth', 1.5) * 0.9,
                    marker=marker,
                    markersize=ln.get('markersize', 5),
                    alpha=alpha,
                    label=label,
                )

            # Scatters
            for sci, sc in enumerate(s_ad.get('scatters', [])):
                ax_target.scatter(sc['x'], sc['y'], color=color,
                                  s=20, alpha=0.7,
                                  label=s_label if (ai == 0 and sci == 0
                                                     and not visible_lines) else None)

            # Bars: side-by-side grouping
            n_samples = len(items)
            for bi, br in enumerate(s_ad.get('bars', [])):
                xs = np.array(br.get('x', []), dtype=float)
                heights = np.array(br.get('heights', []))
                widths = np.array(br.get('widths', []))
                if len(xs) == 0:
                    continue
                w = widths.mean() if len(widths) else 0.8
                offset = (sample_idx - (n_samples - 1) / 2) * (w / n_samples)
                ax_target.bar(xs + offset, heights, width=w / n_samples,
                              color=color, alpha=0.8,
                              label=s_label if (ai == 0 and bi == 0
                                                and not visible_lines) else None)

            # Reference lines (only from first sample to avoid duplicates)
            if sample_idx == 0:
                for hl in s_ad.get('axhlines', []):
                    ax_target.axhline(hl['y'], color='gray',
                                       linestyle=hl.get('linestyle', '--'),
                                       linewidth=0.8, alpha=0.5)
                for vl in s_ad.get('axvlines', []):
                    ax_target.axvline(vl['x'], color='gray',
                                       linestyle=vl.get('linestyle', '--'),
                                       linewidth=0.8, alpha=0.5)

    # Apply axis labels/titles/scales from reference
    for ai, ax_target in enumerate(flat_axes):
        if ai >= len(primary_axes):
            continue
        ad = primary_axes[ai]
        if ad.get('xlabel'):
            ax_target.set_xlabel(ad['xlabel'], fontsize=11)
        if ad.get('ylabel'):
            ax_target.set_ylabel(ad['ylabel'], fontsize=11)
        if ad.get('title'):
            ax_target.set_title(ad['title'], fontsize=11)
        try:
            if ad.get('xscale') == 'log':
                ax_target.set_xscale('log')
            if ad.get('yscale') == 'log':
                ax_target.set_yscale('log')
        except Exception:
            pass
        ax_target.grid(True, alpha=0.3)
        if ax_target.get_legend_handles_labels()[0]:
            ax_target.legend(fontsize=8, loc='best', framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Plot saved: {save_path}")
    return fig
