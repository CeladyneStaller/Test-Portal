"""
Condition Label Utilities
==========================
Parse test conditions from filenames and measured data columns.
Format them as human-readable labels for plot annotations.

Supported filename patterns:
  Scribner FCD: ..._80C-100RH_0o1H2-0o1N2_0kPa.fcd
  FCTS CSV:     ..._80.3Cell-95RH_0.2WH2-0.2WA.csv
"""

import re
import numpy as np


def parse_conditions_from_filename(filename):
    """
    Extract test conditions from a filename string.

    Returns dict with keys (all optional):
        T_C, RH_pct, H2_flow, cathode_gas, cathode_flow, P_kPa
    """
    name = filename.upper()
    cond = {}

    # Temperature: "80C" or "80.3CELL"
    m = re.search(r'(\d+\.?\d*)(?:CELL|C)(?=[-_\.])', name)
    if m:
        cond['T_C'] = float(m.group(1))

    # RH: "100RH" or "95RH"
    m = re.search(r'(\d+\.?\d*)RH', name)
    if m:
        cond['RH_pct'] = float(m.group(1))

    # H2 flow: "0o1H2" (Scribner, "o" = ".") or "0.05WH2" or "0.2WH2"
    m = re.search(r'(\d+[oO]\d+|\d+\.?\d*)(?:W)?H2', name)
    if m:
        val = m.group(1).replace('o', '.').replace('O', '.')
        cond['H2_flow'] = float(val)

    # Cathode gas + flow: "0o1N2" or "0.2WA" or "0.4WN2" or "0.2WA"
    m = re.search(r'(\d+[oO]\d+|\d+\.?\d*)(?:W)?(N2|A|AIR)(?=[-_\.\s]|$)', name)
    if m:
        val = m.group(1).replace('o', '.').replace('O', '.')
        cond['cathode_flow'] = float(val)
        gas = m.group(2)
        cond['cathode_gas'] = 'Air' if gas in ('A', 'AIR') else 'N₂'

    # Pressure: "0kPa" or "100kPa" (from filename or folder)
    m = re.search(r'(\d+\.?\d*)KPA', name)
    if m:
        cond['P_kPa'] = float(m.group(1))

    return cond


def parse_conditions_from_folder(folder_name):
    """Extract conditions from a folder name (e.g., '100kPa', 'BOL')."""
    name = folder_name.upper()
    cond = {}

    m = re.search(r'(\d+\.?\d*)KPA', name)
    if m:
        cond['P_kPa'] = float(m.group(1))

    return cond


def conditions_from_measured(measured_dict):
    """
    Convert measured condition columns (from polcurve condition_cols)
    to the standard conditions dict.

    measured_dict: e.g. {'T_cell (C)': array([80.1, ...]), 'H2_flow (slpm)': array([...]), ...}
    """
    if not measured_dict:
        return {}

    cond = {}
    for key, vals in measured_dict.items():
        mean_val = float(np.nanmean(vals))
        kl = key.lower()
        if 't_cell' in kl or 'temperature' in kl:
            cond['T_C'] = round(mean_val, 1)
        elif 't_anode' in kl and 'dp' in kl:
            cond['T_anode_dp'] = round(mean_val, 1)
        elif 't_cathode' in kl and 'dp' in kl:
            cond['T_cathode_dp'] = round(mean_val, 1)
        elif 'h2' in kl and 'flow' in kl:
            cond['H2_flow'] = round(mean_val, 3)
        elif ('air' in kl or 'cathode' in kl) and 'flow' in kl:
            cond['cathode_flow'] = round(mean_val, 3)
            cond['cathode_gas'] = 'Air'

    # Compute RH from dewpoints if available
    if 'T_C' in cond and 'T_cathode_dp' in cond:
        T = cond['T_C']
        Tdp = cond['T_cathode_dp']
        if T > 0 and Tdp > 0:
            # Magnus formula approximation
            a, b = 17.625, 243.04
            rh = 100 * np.exp((a * Tdp / (b + Tdp)) - (a * T / (b + T)))
            cond['RH_pct'] = round(min(rh, 100), 0)

    return cond


def merge_conditions(filename_cond, measured_cond):
    """
    Merge filename-parsed conditions with measured data.
    Measured data takes priority where available.
    """
    merged = dict(filename_cond)
    merged.update({k: v for k, v in measured_cond.items() if v is not None})
    return merged


def format_condition_label(cond, compact=False):
    """
    Format conditions dict as a human-readable label string.

    compact=True:  "80°C, 100%RH, H₂/Air 0.2/0.2, 0kPa"
    compact=False: "80°C | 100% RH | H₂: 0.2 / Air: 0.2 slpm | 0 kPag"
    """
    parts = []

    if 'T_C' in cond:
        parts.append(f"{cond['T_C']:.0f}°C")

    if 'RH_pct' in cond:
        parts.append(f"{cond['RH_pct']:.0f}%RH")

    # Flows
    h2 = cond.get('H2_flow')
    cath = cond.get('cathode_flow')
    cgas = cond.get('cathode_gas', 'N₂')
    if h2 is not None and cath is not None:
        if compact:
            parts.append(f"H₂/{cgas} {h2:.2g}/{cath:.2g}")
        else:
            parts.append(f"H₂: {h2:.2g} / {cgas}: {cath:.2g} slpm")
    elif h2 is not None:
        parts.append(f"H₂: {h2:.2g} slpm")

    if 'P_kPa' in cond:
        p = cond['P_kPa']
        if compact:
            parts.append(f"{p:.0f}kPa")
        else:
            parts.append(f"{p:.0f} kPag")

    sep = ', ' if compact else ' | '
    return sep.join(parts) if parts else ''


def get_condition_label(filepath=None, label=None, conditions=None,
                        folder_path=None, compact=False):
    """
    Convenience function: parse from all available sources and return
    a formatted label string.

    Parameters
    ----------
    filepath : str or Path, optional
        Full file path (filename is parsed)
    label : str, optional
        Filename stem (parsed if filepath not given)
    conditions : dict, optional
        Measured condition data (from condition_cols)
    folder_path : str, optional
        Folder name to extract pressure from
    compact : bool
        Use compact format for legend entries
    """
    from pathlib import Path

    # Parse from filename
    fname = ''
    if filepath:
        fname = Path(filepath).name
    elif label:
        fname = label

    file_cond = parse_conditions_from_filename(fname)

    # Parse from folder
    if folder_path:
        folder_cond = parse_conditions_from_folder(Path(folder_path).name)
        file_cond = merge_conditions(file_cond, folder_cond)

    # Merge with measured data
    if conditions:
        meas_cond = conditions_from_measured(conditions)
        final = merge_conditions(file_cond, meas_cond)
    else:
        final = file_cond

    return format_condition_label(final, compact=compact)


def img_ext_from_params(params):
    """Extract image extension from params dict. Returns ext string or None for 'no images'."""
    fmt = (params or {}).get('image_format', 'png')
    if fmt == 'none':
        return None
    return fmt


def img_path(save_dir, name, ext):
    """Build image save path, or return None if ext is None (no images)."""
    import os
    if ext is None or save_dir is None:
        return None
    return os.path.join(save_dir, f'{name}.{ext}')
