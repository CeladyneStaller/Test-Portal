#!/usr/bin/env python3
"""
Fuel Cell ECSA Analysis
=======================
Calculates electrochemical surface area from cyclic voltammetry data
using H_UPD integration and CO stripping methods.

Methods:
  1. H_UPD: Integrates hydrogen adsorption/desorption charge (0.05–0.40 V vs RHE)
  2. CO stripping: Integrates CO oxidation peak minus baseline CV

Reference charge densities:
  - H_UPD on polycrystalline Pt: 210 µC/cm²_Pt
  - CO stripping on Pt: 420 µC/cm²_Pt (2e⁻ per CO)

Usage:
  python ecsa_analysis.py                   # runs built-in demo
  python ecsa_analysis.py --file data.csv   # analyze real CV data
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.helpers.plot_compare import save_with_sidecar
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import argparse
import csv
import os
import glob
import subprocess
import sys


def run(input_dir: str, output_dir: str, params: dict = None) -> dict:
    """Analyze uploaded CV data for ECSA using H_UPD method (batch mode)."""
    from pathlib import Path
    p = params or {}

    inp = Path(input_dir)
    out = Path(output_dir)
    all_files = sorted(
        [f for f in inp.rglob('*')
         if f.is_file() and f.suffix.lower() in ('.csv', '.txt', '.tsv', '.fcd')]
    )

    if not all_files:
        return {"status": "error", "message": "No CSV/TSV/FCD files found"}

    # Filter by ECSA keywords; fall back to all files if none match
    KEYWORDS = ['ECSA', 'CV-50MVS']
    filtered = [f for f in all_files
                if any(kw in f.name.upper() for kw in KEYWORDS)]
    csv_files = filtered if filtered else all_files

    # Test stand presets
    stand = int(p.get('stand', 0))

    # Auto-detect: CSV-only files → FCTS
    has_fcd = any(f.suffix.lower() == '.fcd' for f in csv_files)
    if not has_fcd and stand == 0:
        has_csv_only = all(f.suffix.lower() in ('.csv', '.txt', '.tsv') for f in csv_files)
        if has_csv_only:
            stand = 1

    STAND_PRESETS = {
        0: {'delimiter': '\t', 'skip': 76, 'v_col': 5, 'i_col': 1, 'i_scale': 1.0},
        1: {'delimiter': ',',  'skip': 1,  'v_col': 2, 'i_col': 3, 'i_scale': 1.0},
    }

    has_tab_files = any(f.suffix.lower() in ('.fcd', '.tsv') for f in csv_files)

    if stand in STAND_PRESETS:
        preset = STAND_PRESETS[stand]
        delimiter = preset['delimiter']
        skip = preset['skip']
        v_col = preset['v_col']
        i_col = preset['i_col']
        i_scale = preset['i_scale']
    elif has_tab_files:
        delimiter = '\t'
        skip, v_col, i_col, i_scale = 1, 0, 1, 1.0
    else:
        delimiter = ','
        skip, v_col, i_col, i_scale = 1, 0, 1, 1.0

    scan_rate = float(p.get('scan_rate', 0.050))
    geo_area = float(p.get('geo_area', 5.0))
    loading = float(p.get('loading', 0.20))
    v_low = float(p.get('v_low', 0.08))
    v_high = float(p.get('v_high', 0.40))
    cycle = p.get('cycle', '2')
    # Convert numeric string to int for select_cycle
    try:
        cycle = int(cycle)
    except (ValueError, TypeError):
        pass  # keep as string ('last', 'first', 'average')

    filepaths = [str(f) for f in csv_files]
    labels = [f.stem for f in csv_files]

    from scripts.helpers.conditions import img_ext_from_params
    image_ext = img_ext_from_params(p)

    all_results = run_batch(
        filepaths, labels, scan_rate, geo_area, loading,
        delimiter, skip, v_col, i_col, v_low, v_high,
        i_scale, cycle=cycle, save_dir=str(out), image_ext=image_ext
    )

    output_files = [str(f.relative_to(out)) for f in out.rglob('*') if f.is_file()]
    if not output_files:
        raise RuntimeError(
            f"Analysis produced no output. {len(csv_files)} file(s) were found "
            f"but none could be processed. Check file format and parameters."
        )
    return {
        "status": "success",
        "files_processed": len(csv_files),
        "files_produced": output_files,
    }

# ─── Physical constants ───────────────────────────────────────────────
Q_H_UPD = 210e-6       # C/cm²_Pt — monolayer H on polycrystalline Pt
Q_CO    = 420e-6        # C/cm²_Pt — monolayer CO on polycrystalline Pt (2e⁻)
F       = 96485.0       # C/mol


# ═══════════════════════════════════════════════════════════════════════
#  Data I/O
# ═══════════════════════════════════════════════════════════════════════

def load_cv_data(filepath, v_col=0, i_col=1, delimiter=',', skip_header=1):
    """
    Load CV data from a CSV/TSV file.

    Expected columns: potential (V vs RHE), current (A)
    Returns arrays sorted by potential for the anodic sweep.
    """
    V, I = [], []
    # Clean filepath: PowerShell drag-drop adds "& '...'" wrapper
    filepath = filepath.strip()
    if filepath.startswith('& '):
        filepath = filepath[2:]
    filepath = filepath.strip().strip('"').strip("'")
    filepath = filepath.strip('\u2018\u2019\u201c\u201d')  # smart quotes
    filepath = filepath.strip('\u202a\u200b')
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for _ in range(skip_header):
            next(reader)
        for row in reader:
            try:
                V.append(float(row[v_col]))
                I.append(float(row[i_col]))
            except (ValueError, IndexError):
                continue
    return np.array(V), np.array(I)


def extract_cycles(V, I, min_points=20):
    """
    Split multi-cycle CV data into individual full cycles.

    A full cycle is one complete anodic + cathodic sweep (valley → peak → valley)
    or (peak → valley → peak). Turning points are detected from sign changes in
    the smoothed dV signal with a minimum segment length filter to reject noise.

    Parameters
    ----------
    V, I : array
        Full CV data (may contain many cycles).
    min_points : int
        Minimum number of points in a half-sweep to count as real (rejects noise).

    Returns
    -------
    cycles : list of (V_cycle, I_cycle) tuples
        One entry per complete cycle.
    turn_indices : array
        Indices of all turning points in the original data.
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)

    # Smooth dV to suppress noise-induced false reversals
    dV = np.diff(V)
    if len(dV) > 15:
        wl = min(11, len(dV) // 2 * 2 - 1)
        if wl >= 5:
            dV_smooth = savgol_filter(dV, wl, 2)
        else:
            dV_smooth = dV
    else:
        dV_smooth = dV

    # Find sign changes in dV (turning points)
    signs = np.sign(dV_smooth)
    signs[signs == 0] = 1  # treat zero as positive to avoid double-counts
    raw_turns = np.where(np.diff(signs) != 0)[0] + 1

    # Filter out segments shorter than min_points (noise)
    turns = []
    prev = 0
    for t in raw_turns:
        if t - prev >= min_points:
            turns.append(t)
            prev = t
    # Also require the tail segment to be long enough
    if len(turns) > 0 and (len(V) - turns[-1]) < min_points:
        turns.pop()
    turns = np.array(turns)

    if len(turns) < 2:
        # Single cycle or monotonic — return entire dataset as one cycle
        return [(V, I)], turns

    # Determine if the data starts near a voltage extremum (valley or peak).
    # If the leading segment before turns[0] is a valid half-sweep, prepend
    # index 0 as an implicit turning point so we don't lose the first cycle.
    V_range = V.max() - V.min()
    v_start = V[0]
    near_min = (v_start - V.min()) < 0.15 * V_range
    near_max = (V.max() - v_start) < 0.15 * V_range

    if turns[0] >= min_points and (near_min or near_max):
        # Data starts near an extremum with a full half-sweep before first turn
        turns = np.concatenate(([0], turns))

    # Also check if the tail after the last turn is a valid half-sweep
    tail_len = len(V) - turns[-1]
    if tail_len >= min_points:
        v_end = V[-1]
        end_near_min = (v_end - V.min()) < 0.15 * V_range
        end_near_max = (V.max() - v_end) < 0.15 * V_range
        if end_near_min or end_near_max:
            turns = np.concatenate((turns, [len(V) - 1]))

    # Group pairs of turning points into full cycles
    # Each cycle spans two half-sweeps: turns[i] → turns[i+2]
    cycles = []
    i = 0
    while i + 2 <= len(turns) - 1:
        start = turns[i]
        end = turns[i + 2]
        cycles.append((V[start:end + 1].copy(), I[start:end + 1].copy()))
        i += 2

    # If no full cycles were assembled, fall back to entire dataset
    if len(cycles) == 0:
        cycles = [(V, I)]

    return cycles, turns


def select_cycle(cycles, choice='last'):
    """
    Select or average cycles from a multi-cycle dataset.

    Parameters
    ----------
    cycles : list of (V, I) tuples
        Output of extract_cycles().
    choice : str or int
        'last'      — last complete cycle (default, most stable)
        'first'     — first complete cycle
        'average'   — average all cycles (interpolated onto common V grid)
        'all'       — return list of all cycles (for overlay plots)
        int         — specific cycle number (1-indexed)

    Returns
    -------
    V, I : arrays (or list of tuples if choice='all')
    """
    if isinstance(choice, int):
        idx = choice - 1  # convert to 0-indexed
        if idx < 0 or idx >= len(cycles):
            raise ValueError(f'Cycle {choice} requested but only {len(cycles)} cycles found')
        return cycles[idx]

    if choice == 'last':
        return cycles[-1]
    elif choice == 'first':
        return cycles[0]
    elif choice == 'all':
        return cycles
    elif choice == 'average':
        # CV data is non-monotonic (V sweeps up then down), so simple
        # interpolation fails. Strategy:
        #   1. If all cycles have the same length, average point-by-point
        #      (they share the same V trajectory).
        #   2. Otherwise, extract anodic sweeps, interpolate on a common
        #      sorted V grid, and reconstruct.
        lengths = [len(Vc) for Vc, _ in cycles]
        if len(set(lengths)) == 1:
            # All same length — direct point-wise average
            I_stack = np.array([Ic for _, Ic in cycles])
            I_avg = np.mean(I_stack, axis=0)
            return cycles[0][0].copy(), I_avg
        else:
            # Variable lengths — average anodic sweeps on common grid
            anodic_pairs = []
            for Vc, Ic in cycles:
                V_a, I_a, _, _ = split_sweeps(Vc, Ic)
                idx = np.argsort(V_a)
                anodic_pairs.append((V_a[idx], I_a[idx]))
            # Common grid from the first cycle's anodic sweep
            V_ref = anodic_pairs[0][0]
            I_stack = []
            for Va, Ia in anodic_pairs:
                I_stack.append(np.interp(V_ref, Va, Ia))
            I_avg = np.mean(I_stack, axis=0)
            return V_ref, I_avg
    else:
        raise ValueError(f'Unknown choice: {choice!r}')


def split_sweeps(V, I):
    """
    Split a single CV cycle into anodic (V-increasing) and cathodic
    (V-decreasing) sweeps.

    Uses the voltage extremum as the turning point, which is robust
    against noise-induced dV sign changes that can cause the entire
    cycle to be treated as one sweep.
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)

    if len(V) < 5:
        return V, I, None, None

    # Determine cycle type from start/end vs middle voltages
    v_start = V[0]
    v_mid_region = V[len(V) // 3: 2 * len(V) // 3]

    if np.mean(v_mid_region) > v_start:
        # Valley → Peak → Valley: split at V_max
        turn = np.argmax(V)
    else:
        # Peak → Valley → Peak: split at V_min
        turn = np.argmin(V)

    # Ensure the turn isn't at the very start or end
    if turn < 3 or turn > len(V) - 3:
        # Monotonic — determine direction from overall slope
        if V[-1] > V[0]:
            return V, I, None, None        # anodic
        else:
            return None, None, V, I        # cathodic

    V1, I1 = V[:turn], I[:turn]           # before turning point
    V2, I2 = V[turn + 1:], I[turn + 1:]   # after turning point

    # Assign by sweep direction: anodic = V increasing
    if V1[-1] > V1[0]:
        return V1, I1, V2, I2   # first half is anodic
    else:
        return V2, I2, V1, I1   # second half is anodic


# ═══════════════════════════════════════════════════════════════════════
#  H_UPD Method
# ═══════════════════════════════════════════════════════════════════════

def _integrate_sweep(V_sweep, I_sweep, scan_rate, geo_area, v_low, v_high):
    """
    Integrate H_UPD charge on a single sorted sweep.

    Returns (Q_hupd, j, j_baseline, V_int, j_net) or None if sweep is empty.
    """
    if V_sweep is None or len(V_sweep) < 3:
        return None

    # Sort ascending for integration
    idx = np.argsort(V_sweep)
    V_s = V_sweep[idx]
    I_s = I_sweep[idx]

    j = I_s / geo_area

    # Constant DL baseline from flat region just above H_UPD window
    dl_low = v_high
    dl_high = min(v_high + 0.08, V_s.max())
    mask_dl = (V_s >= dl_low) & (V_s <= dl_high)
    if mask_dl.sum() >= 2:
        j_dl = np.mean(j[mask_dl])
    else:
        idx_bl = np.argmin(np.abs(V_s - v_high))
        j_dl = j[idx_bl]
    j_baseline = np.full_like(j, j_dl)

    mask = (V_s >= v_low) & (V_s <= v_high)
    V_int = V_s[mask]
    j_net = j[mask] - j_baseline[mask]

    if len(V_int) < 2:
        return None

    Q_hupd = np.abs(trapezoid(j_net, V_int)) / scan_rate

    return Q_hupd, V_s, j, j_baseline, V_int, j_net


def compute_ecsa_hupd(V, I, scan_rate, geo_area,
                      v_low=0.05, v_high=0.40,
                      loading_mg_cm2=None,
                      cycle='last'):
    """
    ECSA from H_UPD charge integration on both anodic and cathodic sweeps.

    Parameters
    ----------
    V, I : array
        Full CV data (potential in V vs RHE, current in A).
        May contain multiple cycles.
    scan_rate : float
        Scan rate in V/s.
    geo_area : float
        Geometric electrode area in cm².
    v_low, v_high : float
        H_UPD integration window (V vs RHE).
    loading_mg_cm2 : float or None
        Pt loading in mg/cm² for mass-specific ECSA.
    cycle : str or int
        Which cycle to analyze: 'last', 'first', 'average', or cycle number (1-indexed).

    Returns
    -------
    dict with ECSA results for anodic, cathodic, and average, plus plotting data.
    """
    cycles, turns = extract_cycles(V, I)
    n_cycles = len(cycles)
    V_cyc, I_cyc = select_cycle(cycles, choice=cycle)

    V_an, I_an, V_ca, I_ca = split_sweeps(V_cyc, I_cyc)

    an_result = _integrate_sweep(V_an, I_an, scan_rate, geo_area, v_low, v_high)
    ca_result = _integrate_sweep(V_ca, I_ca, scan_rate, geo_area, v_low, v_high)

    results = {
        'method': 'H_UPD',
        'v_low': v_low,
        'v_high': v_high,
        'scan_rate_mVs': scan_rate * 1e3,
        'n_cycles_detected': n_cycles,
        'cycle_used': cycle,
        'geo_area': geo_area,
        '_V_raw': V, '_I_raw': I,
        '_cycles': cycles,
    }

    for label, sweep_result in [('anodic', an_result), ('cathodic', ca_result)]:
        if sweep_result is not None:
            Q, V_s, j, j_bl, V_int, j_net = sweep_result
            ecsa = Q / Q_H_UPD * geo_area
            rf = ecsa / geo_area

            results[f'{label}_Q_mC_cm2'] = Q * 1e3
            results[f'{label}_ECSA_cm2'] = ecsa
            results[f'{label}_RF'] = rf
            results[f'_{label}_V'] = V_s
            results[f'_{label}_j'] = j
            results[f'_{label}_j_bl'] = j_bl
            results[f'_{label}_V_int'] = V_int
            results[f'_{label}_j_net'] = j_net

            if loading_mg_cm2 is not None and loading_mg_cm2 > 0:
                m_Pt = loading_mg_cm2 * geo_area * 1e-3
                results[f'{label}_ECSA_m2_per_g'] = ecsa / m_Pt / 1e4

    # Compute averages where both sweeps exist
    has_an = 'anodic_ECSA_cm2' in results
    has_ca = 'cathodic_ECSA_cm2' in results
    if has_an and has_ca:
        results['average_Q_mC_cm2'] = (results['anodic_Q_mC_cm2'] + results['cathodic_Q_mC_cm2']) / 2
        results['average_ECSA_cm2'] = (results['anodic_ECSA_cm2'] + results['cathodic_ECSA_cm2']) / 2
        results['average_RF'] = (results['anodic_RF'] + results['cathodic_RF']) / 2
        if 'anodic_ECSA_m2_per_g' in results:
            results['average_ECSA_m2_per_g'] = (results['anodic_ECSA_m2_per_g'] + results['cathodic_ECSA_m2_per_g']) / 2

    # Backward-compatible keys (point to average if available, else anodic)
    primary = 'average' if has_an and has_ca else ('anodic' if has_an else 'cathodic')
    results['ECSA_cm2'] = results[f'{primary}_ECSA_cm2']
    results['roughness_factor'] = results[f'{primary}_RF']
    results['Q_hupd_mC_cm2'] = results[f'{primary}_Q_mC_cm2']
    if f'{primary}_ECSA_m2_per_g' in results:
        results['ECSA_m2_per_g'] = results[f'{primary}_ECSA_m2_per_g']

    if loading_mg_cm2 is not None:
        results['loading_mg_cm2'] = loading_mg_cm2

    return results


# ═══════════════════════════════════════════════════════════════════════
#  CO Stripping Method
# ═══════════════════════════════════════════════════════════════════════

def compute_ecsa_co_strip(V_strip, I_strip, V_base, I_base,
                          scan_rate, geo_area,
                          v_low=0.40, v_high=1.00,
                          loading_mg_cm2=None,
                          cycle_strip='first', cycle_base='last'):
    """
    ECSA from CO stripping voltammetry.

    Parameters
    ----------
    V_strip, I_strip : array
        CO stripping CV (may contain multiple cycles).
    V_base, I_base : array
        Baseline CV (may contain multiple cycles).
    scan_rate : float
        Scan rate in V/s.
    geo_area : float
        Geometric electrode area in cm².
    v_low, v_high : float
        CO oxidation integration window (V vs RHE).
    loading_mg_cm2 : float or None
        Pt loading for mass-specific ECSA.
    cycle_strip : str or int
        Which cycle from the stripping file ('first', 'last', int).
    cycle_base : str or int
        Which cycle from the baseline file ('first', 'last', int).

    Returns
    -------
    dict with ECSA results and plotting data.
    """
    # Extract requested cycles
    cycles_s, _ = extract_cycles(V_strip, I_strip)
    cycles_b, _ = extract_cycles(V_base, I_base)
    V_strip_c, I_strip_c = select_cycle(cycles_s, choice=cycle_strip)
    V_base_c, I_base_c   = select_cycle(cycles_b, choice=cycle_base)

    # Use anodic sweeps only
    V_s, I_s, _, _ = split_sweeps(V_strip_c, I_strip_c)
    V_b, I_b, _, _ = split_sweeps(V_base_c, I_base_c)

    # Sort ascending
    for arrs in [(V_s, I_s), (V_b, I_b)]:
        idx = np.argsort(arrs[0])
        arrs_sorted = (arrs[0][idx], arrs[1][idx])

    idx_s = np.argsort(V_s); V_s, I_s = V_s[idx_s], I_s[idx_s]
    idx_b = np.argsort(V_b); V_b, I_b = V_b[idx_b], I_b[idx_b]

    # Interpolate baseline onto stripping potential grid
    V_common = V_s[(V_s >= v_low) & (V_s <= v_high)]
    j_s = np.interp(V_common, V_s, I_s / geo_area)
    j_b = np.interp(V_common, V_b, I_b / geo_area)

    j_net = j_s - j_b
    Q_co = np.abs(trapezoid(j_net, V_common)) / scan_rate

    ecsa_cm2 = Q_co / Q_CO * geo_area
    rf = ecsa_cm2 / geo_area

    results = {
        'method': 'CO stripping',
        'Q_co_mC_cm2': Q_co * 1e3,
        'ECSA_cm2': ecsa_cm2,
        'roughness_factor': rf,
        'v_low': v_low,
        'v_high': v_high,
        'scan_rate_mVs': scan_rate * 1e3,
        '_V_common': V_common, '_j_strip': j_s, '_j_base': j_b, '_j_net': j_net,
    }

    if loading_mg_cm2 is not None and loading_mg_cm2 > 0:
        m_Pt = loading_mg_cm2 * geo_area * 1e-3
        ecsa_mass = ecsa_cm2 / m_Pt / 1e4
        results['ECSA_m2_per_g'] = ecsa_mass
        results['loading_mg_cm2'] = loading_mg_cm2

    return results


# ═══════════════════════════════════════════════════════════════════════
#  ECSA Degradation Tracker
# ═══════════════════════════════════════════════════════════════════════

def ecsa_degradation_summary(ecsa_values, cycle_numbers=None):
    """
    Summarize ECSA loss over an AST campaign.

    Parameters
    ----------
    ecsa_values : list of float
        ECSA at each checkpoint (m²/g or cm²).
    cycle_numbers : list of int or None
        Corresponding AST cycle counts.

    Returns
    -------
    dict with degradation metrics.
    """
    ecsa = np.array(ecsa_values, dtype=float)
    if cycle_numbers is None:
        cycle_numbers = np.arange(len(ecsa))
    else:
        cycle_numbers = np.array(cycle_numbers, dtype=float)

    ecsa_norm = ecsa / ecsa[0] * 100.0  # % of BOL
    loss_pct = 100.0 - ecsa_norm[-1]

    # Fit exponential decay: ECSA = ECSA_0 * exp(-k * N)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_ratio = np.log(ecsa / ecsa[0])
    valid = np.isfinite(ln_ratio)
    if valid.sum() >= 2:
        k_fit = -np.polyfit(cycle_numbers[valid], ln_ratio[valid], 1)[0]
    else:
        k_fit = np.nan

    return {
        'ecsa_values': ecsa.tolist(),
        'cycles': cycle_numbers.tolist(),
        'ecsa_normalized_pct': ecsa_norm.tolist(),
        'total_loss_pct': loss_pct,
        'decay_rate_k': k_fit,
        'half_life_cycles': np.log(2) / k_fit if k_fit > 0 else np.inf,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_hupd_analysis(results, save_path=None):
    """
    Plot H_UPD ECSA analysis: full CV, anodic sweep, cathodic sweep,
    and both net integrations overlaid with average.
    """
    cycles = results.get('_cycles', [])
    n_cyc = results['n_cycles_detected']
    geo_area = results['geo_area']
    has_an = '_anodic_V' in results
    has_ca = '_cathodic_V' in results

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # ── Panel 1: Full CV — all cycles overlaid ──
    ax = axes[0]
    if n_cyc > 1:
        cmap = plt.cm.viridis
        colors = [cmap(i / max(n_cyc - 1, 1)) for i in range(n_cyc)]
        for i, (Vc, Ic) in enumerate(cycles):
            jc = Ic / geo_area * 1e3
            ax.plot(Vc, jc, '-', color=colors[i], lw=1.0, alpha=0.7,
                    label=f'Cycle {i+1}' if n_cyc <= 10 else None)
        if n_cyc <= 10:
            ax.legend(fontsize=7, loc='upper left', ncol=2)
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                        norm=plt.Normalize(1, n_cyc))
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
            cb.set_label('Cycle #', fontsize=9)
    else:
        V_raw = results['_V_raw']
        I_raw = results['_I_raw']
        ax.plot(V_raw, I_raw / geo_area * 1e3, 'b-', lw=1.2)

    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('E vs RHE (V)')
    ax.set_ylabel('j (mA cm$^{-2}$)')
    ax.set_title(f'Full CV — {n_cyc} cycle{"s" if n_cyc != 1 else ""}')

    # ── Helper: plot a single sweep panel ──
    def _plot_sweep(ax, label, color):
        V_s = results[f'_{label}_V']
        j_s = results[f'_{label}_j'] * 1e3
        j_bl = results[f'_{label}_j_bl'] * 1e3
        V_int = results[f'_{label}_V_int']
        j_net = results[f'_{label}_j_net'] * 1e3

        ax.plot(V_s, j_s, '-', color=color, lw=1.5, label=f'{label.capitalize()} sweep')
        ax.plot(V_s, j_bl, 'r--', lw=1.0, label='DL baseline')
        ax.axvline(results['v_low'], color='gray', ls=':', alpha=0.6)
        ax.axvline(results['v_high'], color='gray', ls=':', alpha=0.6)
        ax.axhline(0, color='k', lw=0.5)
        j_bl_int = np.interp(V_int, V_s, j_bl)
        ax.fill_between(V_int, j_net + j_bl_int, j_bl_int, alpha=0.25, color=color)
        ax.set_xlabel('E vs RHE (V)')
        ax.set_ylabel('j (mA cm$^{-2}$)')
        Q = results[f'{label}_Q_mC_cm2']
        ax.set_title(f'{label.capitalize()} — Q = {Q:.2f} mC/cm²')
        ax.legend(fontsize=8)

    # ── Panel 2: Anodic sweep ──
    if has_an:
        _plot_sweep(axes[1], 'anodic', 'steelblue')
    else:
        axes[1].set_visible(False)

    # ── Panel 3: Cathodic sweep ──
    if has_ca:
        _plot_sweep(axes[2], 'cathodic', 'darkorange')
    else:
        axes[2].set_visible(False)

    # ── Panel 4: Both net integrations overlaid (absolute charge) ──
    ax = axes[3]
    if has_an:
        V_int_a = results['_anodic_V_int']
        j_net_a = np.abs(results['_anodic_j_net']) * 1e3
        ax.fill_between(V_int_a, j_net_a, 0, alpha=0.25, color='steelblue')
        ax.plot(V_int_a, j_net_a, '-', color='steelblue', lw=1.5,
                label=f'Anodic ({results["anodic_ECSA_cm2"]:.1f} cm²)')
    if has_ca:
        V_int_c = results['_cathodic_V_int']
        j_net_c = np.abs(results['_cathodic_j_net']) * 1e3
        ax.fill_between(V_int_c, j_net_c, 0, alpha=0.25, color='darkorange')
        ax.plot(V_int_c, j_net_c, '--', color='darkorange', lw=1.5,
                label=f'Cathodic ({results["cathodic_ECSA_cm2"]:.1f} cm²)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('E vs RHE (V)')
    ax.set_ylabel('|j$_{net}$| (mA cm$^{-2}$)')
    ax.set_title('H$_{UPD}$ Comparison')
    ax.legend(fontsize=8)

    # ── Annotation box ──
    lines = []
    if has_an:
        line = f'Anodic:   RF={results["anodic_RF"]:.1f}'
        if 'anodic_ECSA_m2_per_g' in results:
            line += f'  {results["anodic_ECSA_m2_per_g"]:.1f} m²/g'
        lines.append(line)
    if has_ca:
        line = f'Cathodic: RF={results["cathodic_RF"]:.1f}'
        if 'cathodic_ECSA_m2_per_g' in results:
            line += f'  {results["cathodic_ECSA_m2_per_g"]:.1f} m²/g'
        lines.append(line)
    if 'average_ECSA_cm2' in results:
        line = f'Average:  RF={results["average_RF"]:.1f}'
        if 'average_ECSA_m2_per_g' in results:
            line += f'  {results["average_ECSA_m2_per_g"]:.1f} m²/g'
        lines.append(line)
    txt = '\n'.join(lines)
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.9))

    cycle_label = results['cycle_used']
    from scripts.helpers.conditions import get_condition_label
    cond_label = get_condition_label(label=results.get('label', ''))
    title = f'ECSA Analysis — H$_{{UPD}}$ (cycle: {cycle_label})'
    if cond_label:
        title += f'\n{cond_label}'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    fig.tight_layout()

    if save_path:
        save_with_sidecar(fig, save_path, dpi=200, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


def plot_co_stripping(results, save_path=None):
    """Plot CO stripping analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    V = results['_V_common']
    j_s = results['_j_strip'] * 1e3
    j_b = results['_j_base'] * 1e3
    j_net = results['_j_net'] * 1e3

    ax = axes[0]
    ax.plot(V, j_s, 'b-', lw=1.5, label='CO strip (1st cycle)')
    ax.plot(V, j_b, 'r--', lw=1.2, label='Baseline (2nd cycle)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('E vs RHE (V)')
    ax.set_ylabel('j (mA cm$^{-2}$)')
    ax.set_title('CO Stripping Voltammetry')
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.fill_between(V, j_net, 0, alpha=0.35, color='coral')
    ax.plot(V, j_net, 'r-', lw=1.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('E vs RHE (V)')
    ax.set_ylabel('j$_{net}$ (mA cm$^{-2}$)')
    ax.set_title(f'CO Oxidation — Q = {results["Q_co_mC_cm2"]:.2f} mC/cm²')

    txt = f'ECSA = {results["ECSA_cm2"]:.2f} cm²$_{{Pt}}$\nRF = {results["roughness_factor"]:.1f}'
    if 'ECSA_m2_per_g' in results:
        txt += f'\n{results["ECSA_m2_per_g"]:.1f} m²/g$_{{Pt}}$'
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.9))

    fig.suptitle('ECSA Analysis — CO Stripping', fontsize=13, fontweight='bold')
    fig.tight_layout()

    if save_path:
        save_with_sidecar(fig, save_path, dpi=200, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


def plot_degradation(deg, save_path=None):
    """Plot ECSA degradation over AST cycling."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    cycles = deg['cycles']
    ecsa = deg['ecsa_values']
    ecsa_n = deg['ecsa_normalized_pct']

    ax = axes[0]
    ax.plot(cycles, ecsa, 'o-', color='steelblue', lw=2, ms=7)
    ax.set_xlabel('AST Cycles')
    ax.set_ylabel('ECSA (m²/g$_{Pt}$)')
    ax.set_title('Absolute ECSA')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(cycles, ecsa_n, 's-', color='firebrick', lw=2, ms=7)
    ax.axhline(100, color='k', ls=':', alpha=0.4)
    ax.set_xlabel('AST Cycles')
    ax.set_ylabel('ECSA Retention (%)')
    ax.set_title(f'Degradation — {deg["total_loss_pct"]:.1f}% loss')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    txt = f'k = {deg["decay_rate_k"]:.2e} /cycle\nt₁/₂ = {deg["half_life_cycles"]:.0f} cycles'
    ax.text(0.97, 0.03, txt, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.9))

    fig.suptitle('ECSA Degradation Tracking', fontsize=13, fontweight='bold')
    fig.tight_layout()

    if save_path:
        save_with_sidecar(fig, save_path, dpi=200, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Synthetic Data Generator (for demo / validation)
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_cv(ecsa_cm2=50.0, geo_area=5.0, scan_rate=0.050,
                          cdl_uF_per_cm2_real=30.0, noise_level=0.002):
    """
    Generate a realistic synthetic PEM fuel cell CV for H_UPD analysis.

    Models:
      - H_UPD adsorption/desorption with two Gaussian peaks (110, 270 mV)
      - Double-layer capacitive current
      - Pt oxide formation/reduction (>0.7 V)
      - Gaussian noise

    Returns V (full cycle), I (A).
    """
    N = 500
    V_fwd = np.linspace(0.05, 1.0, N)
    V_rev = np.linspace(1.0, 0.05, N)

    rf = ecsa_cm2 / geo_area  # roughness factor

    def hupd_current(V, sign=1):
        """Two-peak H_UPD model (weakly and strongly bound H)."""
        # Peak positions, widths, relative heights
        peaks = [(0.12, 0.035, 0.6), (0.27, 0.045, 1.0)]
        j = np.zeros_like(V)
        for V0, sigma, h in peaks:
            j += h * np.exp(-0.5 * ((V - V0) / sigma)**2)
        # Scale to match ECSA: total charge = Q_H * rf
        # Integral of Gaussian = sigma * sqrt(2π) * height
        total_charge_factor = sum(h * sigma * np.sqrt(2 * np.pi) for _, sigma, h in peaks)
        scale = Q_H_UPD * rf * scan_rate / total_charge_factor
        return sign * j * scale

    def oxide_current(V, sign=1):
        """Pt oxide formation (anodic) / reduction (cathodic)."""
        if sign > 0:  # formation — onset ~0.8 V
            j = 5e-5 * rf * np.exp(10.0 * (V - 1.0))
        else:  # reduction — sharper peak ~0.75 V
            j = -8e-4 * rf * np.exp(-0.5 * ((V - 0.75) / 0.05)**2)
        return j

    # Double-layer capacitance: µF/cm²_real * cm²_real / cm²_geo → F/cm²_geo
    cdl = cdl_uF_per_cm2_real * 1e-6 * rf  # F/cm²_geo

    # Anodic sweep
    j_fwd = (hupd_current(V_fwd, +1) +
             cdl * scan_rate +
             oxide_current(V_fwd, +1))

    # Cathodic sweep
    j_rev = (hupd_current(V_rev, -1) -
             cdl * scan_rate +
             oxide_current(V_rev, -1))

    V = np.concatenate([V_fwd, V_rev])
    j = np.concatenate([j_fwd, j_rev])

    # Add noise
    j += np.random.normal(0, noise_level * cdl * scan_rate, len(j))

    I = j * geo_area  # Convert to total current (A)
    return V, I


def generate_synthetic_co_strip(ecsa_cm2=50.0, geo_area=5.0, scan_rate=0.020):
    """Generate synthetic CO stripping + baseline CVs."""
    N = 400
    V = np.linspace(0.05, 1.1, N)
    rf = ecsa_cm2 / geo_area

    cdl = 30e-6 * rf  # F/cm²_geo  (30 µF/cm²_real)

    # Baseline (clean Pt)
    j_base = cdl * scan_rate + 1e-4 * rf * np.exp(5.0 * (V - 0.95))

    # CO stripping peak — scale to give correct charge
    # Target: ∫ co_peak dV / ν = Q_CO * rf
    co_sigma1, co_h1 = 0.04, 1.0
    co_sigma2, co_h2 = 0.06, 0.25
    total_co_integral = (co_h1 * co_sigma1 + co_h2 * co_sigma2) * np.sqrt(2 * np.pi)
    co_scale = Q_CO * rf * scan_rate / total_co_integral

    co_peak = co_scale * (co_h1 * np.exp(-0.5 * ((V - 0.78) / co_sigma1)**2) +
                          co_h2 * np.exp(-0.5 * ((V - 0.70) / co_sigma2)**2))

    j_strip = j_base + co_peak

    # Baseline has small H_UPD (clean surface); strip doesn't (CO blocks sites)
    hupd_mag = Q_H_UPD * rf * scan_rate / (0.04 * np.sqrt(2 * np.pi))
    hupd_base = hupd_mag * 0.3 * np.exp(-0.5 * ((V - 0.12) / 0.04)**2)
    j_base += hupd_base

    noise = np.random.normal(0, 0.01 * cdl * scan_rate, N)
    I_strip = (j_strip + noise) * geo_area
    I_base = (j_base + noise * 0.8) * geo_area

    return V, I_strip, V, I_base


# ═══════════════════════════════════════════════════════════════════════
#  CLI + Demo
# ═══════════════════════════════════════════════════════════════════════

def print_results(r):
    """Pretty-print ECSA results."""
    print(f'\n{"═" * 65}')
    print(f'  Method:           {r["method"]}')
    if 'n_cycles_detected' in r:
        n = r['n_cycles_detected']
        used = r['cycle_used']
        print(f'  Cycles detected:  {n}  (used: {used})')
    print(f'  Scan rate:        {r["scan_rate_mVs"]:.0f} mV/s')
    print(f'  Window:           {r["v_low"]:.2f} – {r["v_high"]:.2f} V vs RHE')

    has_an = 'anodic_ECSA_cm2' in r
    has_ca = 'cathodic_ECSA_cm2' in r
    has_avg = 'average_ECSA_cm2' in r

    if 'H_UPD' in r.get('method', '') and has_an and has_ca:
        print(f'  {"─" * 61}')
        print(f'  {"":20s} {"Anodic":>12s}  {"Cathodic":>12s}  {"Average":>12s}')
        print(f'  {"─" * 61}')
        print(f'  {"Charge (mC/cm²)":20s} {r["anodic_Q_mC_cm2"]:>12.3f}  {r["cathodic_Q_mC_cm2"]:>12.3f}  {r["average_Q_mC_cm2"]:>12.3f}')
        print(f'  {"ECSA (cm²_Pt)":20s} {r["anodic_ECSA_cm2"]:>12.2f}  {r["cathodic_ECSA_cm2"]:>12.2f}  {r["average_ECSA_cm2"]:>12.2f}')
        print(f'  {"Roughness factor":20s} {r["anodic_RF"]:>12.1f}  {r["cathodic_RF"]:>12.1f}  {r["average_RF"]:>12.1f}')
        if 'anodic_ECSA_m2_per_g' in r:
            print(f'  {"ECSA (m²/g_Pt)":20s} {r["anodic_ECSA_m2_per_g"]:>12.1f}  {r["cathodic_ECSA_m2_per_g"]:>12.1f}  {r["average_ECSA_m2_per_g"]:>12.1f}')
        print(f'  {"─" * 61}')
        if 'loading_mg_cm2' in r:
            print(f'  Loading:          {r["loading_mg_cm2"]:.3f} mg/cm²')
    else:
        # Single-sweep H_UPD or CO stripping
        key = 'Q_hupd_mC_cm2' if 'H_UPD' in r.get('method', '') else 'Q_co_mC_cm2'
        if key in r:
            print(f'  Charge:           {r[key]:.3f} mC/cm²')
        print(f'  ECSA:             {r["ECSA_cm2"]:.2f} cm²_Pt')
        print(f'  Roughness factor: {r["roughness_factor"]:.1f}')
        if 'ECSA_m2_per_g' in r:
            print(f'  Mass ECSA:        {r["ECSA_m2_per_g"]:.1f} m²/g_Pt')
        if 'loading_mg_cm2' in r:
            print(f'  Loading:          {r["loading_mg_cm2"]:.3f} mg/cm²')

    print(f'{"═" * 65}\n')


def run_demo(save_dir=None):
    """Run full demo with synthetic data."""
    print('\n' + '▓' * 60)
    print('  FUEL CELL ECSA ANALYSIS — DEMO')
    print('▓' * 60)

    geo_area = 5.0         # cm²
    scan_rate = 0.050      # V/s
    loading = 0.10         # mg_Pt/cm²
    target_ecsa = 45.0     # cm²_Pt (known input)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    np.random.seed(42)

    # ── H_UPD Analysis ──
    print('\n[1] H_UPD Method')
    print(f'    Target ECSA: {target_ecsa:.1f} cm²_Pt  (RF = {target_ecsa/geo_area:.1f})')

    V, I = generate_synthetic_cv(ecsa_cm2=target_ecsa, geo_area=geo_area,
                                  scan_rate=scan_rate)

    res_hupd = compute_ecsa_hupd(V, I, scan_rate, geo_area,
                                  loading_mg_cm2=loading)
    print_results(res_hupd)

    err = (res_hupd['ECSA_cm2'] - target_ecsa) / target_ecsa * 100
    print(f'    Recovery error: {err:+.1f}%')

    p1 = os.path.join(save_dir, 'ecsa_hupd.png') if save_dir else None
    plot_hupd_analysis(res_hupd, save_path=p1)

    # ── CO Stripping ──
    print('\n[2] CO Stripping Method')
    target_ecsa_co = 43.0

    V_s, I_s, V_b, I_b = generate_synthetic_co_strip(
        ecsa_cm2=target_ecsa_co, geo_area=geo_area, scan_rate=0.020)

    res_co = compute_ecsa_co_strip(V_s, I_s, V_b, I_b,
                                    scan_rate=0.020, geo_area=geo_area,
                                    loading_mg_cm2=loading)
    print_results(res_co)

    p2 = os.path.join(save_dir, 'ecsa_co_strip.png') if save_dir else None
    plot_co_stripping(res_co, save_path=p2)

    # ── Degradation Tracking ──
    print('\n[3] ECSA Degradation (AST simulation)')
    cycles = [0, 1000, 3000, 5000, 10000, 15000, 20000, 30000]
    ecsa_bol = 55.0  # m²/g
    k_true = 5e-5
    ecsa_ast = [ecsa_bol * np.exp(-k_true * n) * (1 + np.random.normal(0, 0.02))
                for n in cycles]

    deg = ecsa_degradation_summary(ecsa_ast, cycles)
    print(f'    BOL ECSA:     {deg["ecsa_values"][0]:.1f} m²/g')
    print(f'    EOT ECSA:     {deg["ecsa_values"][-1]:.1f} m²/g')
    print(f'    Total loss:   {deg["total_loss_pct"]:.1f}%')
    print(f'    Decay rate k: {deg["decay_rate_k"]:.2e} /cycle')
    print(f'    Half-life:    {deg["half_life_cycles"]:.0f} cycles')

    p3 = os.path.join(save_dir, 'ecsa_degradation.png') if save_dir else None
    plot_degradation(deg, save_path=p3)

    if save_dir is None:
        plt.show()

    return res_hupd, res_co, deg


def parse_fcd_header(filepath):
    """Parse Scribner .fcd header for skip count and column indices."""
    if not filepath.lower().endswith('.fcd'):
        return None
    try:
        with open(filepath, 'r', errors='replace') as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line)
                if 'End Comments' in line:
                    skip = i + 1; break
            else:
                return None
    except Exception:
        return None
    if len(lines) < 2:
        return None
    cols = lines[-2].strip().split('\t')
    result = {'skip': skip}
    for ci, name in enumerate(cols):
        n = name.strip()
        if n == 'I (A)': result['i_col'] = ci
        elif n == 'E_Stack (V)': result['v_col'] = ci
        elif n == 'Ctrl_Mode': result['mode_col'] = ci
    return result


def _prompt(label, default=None, cast=float):
    """Prompt for input with a default value. Returns None if blank and no default."""
    suffix = f' [{default}]' if default is not None else ''
    raw = input(f'  {label}{suffix}: ').strip()
    if raw == '':
        return default
    if cast is not None:
        return cast(raw)
    return raw


def _clean_path(p):
    """Clean a file path from drag-drop artifacts and smart quotes."""
    p = p.strip()
    if p.startswith('& '):
        p = p[2:]
    p = p.strip().strip('"').strip("'")
    p = p.strip('\u2018\u2019\u201c\u201d')  # smart single/double quotes
    p = p.strip('\u202a\u200b')               # LTR mark, zero-width space
    return p


def plot_ecsa_overlay(all_results, save_path=None):
    """
    Overlay full CVs from multiple ECSA analyses on a single plot.
    """
    from scripts.helpers.conditions import get_condition_label

    fig, ax = plt.subplots(figsize=(8, 6))
    n = len(all_results)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for i, r in enumerate(all_results):
        V_raw = r['_V_raw']
        I_raw = r['_I_raw']
        geo = r['geo_area']
        j_raw = I_raw / geo * 1e3  # mA/cm²
        lbl = r.get('label', f'File {i+1}')
        cond = get_condition_label(label=lbl, compact=True)
        legend_lbl = f'{lbl}\n  {cond}' if cond else lbl
        ax.plot(V_raw, j_raw, '-', color=colors[i], lw=1.2, alpha=0.8, label=legend_lbl)

    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('E vs RHE (V)')
    ax.set_ylabel('j (mA cm$^{-2}$)')
    ax.set_title('ECSA — CV Overlay')
    ax.legend(fontsize=7, loc='best', ncol=max(1, n // 8))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_with_sidecar(fig, save_path, dpi=200, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


def run_batch(filepaths, labels, scan_rate, geo_area, loading,
              delimiter, skip, v_col, i_col, v_low, v_high, i_scale,
              cycle='last', save_dir=None, image_ext='png'):
    """
    Batch-process multiple ECSA CV files: analyze each, generate summary CSV,
    individual plots, and a combined CV overlay.

    Returns list of result dicts.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_results = []
    summary_rows = []

    print(f'\n  Processing {len(filepaths)} files...\n')

    for i, (fp, lbl) in enumerate(zip(filepaths, labels)):
        print(f'  [{i+1}/{len(filepaths)}] {lbl}')
        try:
            fcd = parse_fcd_header(fp)
            f_skip = fcd['skip'] if fcd else skip
            f_v = fcd.get('v_col', v_col) if fcd else v_col
            f_i = fcd.get('i_col', i_col) if fcd else i_col
            V, I = load_cv_data(fp, v_col=f_v, i_col=f_i,
                                delimiter=delimiter, skip_header=f_skip)
            I *= i_scale

            cycles_found, _ = extract_cycles(V, I)
            print(f'         {len(V)} pts, {len(cycles_found)} cycle(s), '
                  f'V: {V.min():.3f}–{V.max():.3f} V')

            r = compute_ecsa_hupd(V, I, scan_rate, geo_area,
                                   v_low=v_low, v_high=v_high,
                                   loading_mg_cm2=loading, cycle=cycle)
            r['label'] = lbl
            r['filepath'] = fp
            all_results.append(r)

            # Print one-line summary
            has_avg = 'average_ECSA_cm2' in r
            if has_avg:
                print(f'         Anodic={r["anodic_ECSA_cm2"]:.1f}  '
                      f'Cathodic={r["cathodic_ECSA_cm2"]:.1f}  '
                      f'Avg={r["average_ECSA_cm2"]:.1f} cm²_Pt')
            else:
                print(f'         ECSA={r["ECSA_cm2"]:.1f} cm²_Pt')

            # Individual plot
            if save_dir and image_ext:
                safe_name = lbl.replace(' ', '_').replace('/', '-').replace('\\', '-')
                plot_hupd_analysis(r, save_path=os.path.join(save_dir, f'ecsa_{safe_name}.{image_ext}'))
                plt.close()

            # Summary row
            row = {'Label': lbl, 'File': os.path.basename(fp)}
            if 'anodic_ECSA_cm2' in r:
                row['Anodic Q (mC/cm²)'] = r['anodic_Q_mC_cm2']
                row['Anodic ECSA (cm²)'] = r['anodic_ECSA_cm2']
                row['Anodic RF'] = r['anodic_RF']
            if 'cathodic_ECSA_cm2' in r:
                row['Cathodic Q (mC/cm²)'] = r['cathodic_Q_mC_cm2']
                row['Cathodic ECSA (cm²)'] = r['cathodic_ECSA_cm2']
                row['Cathodic RF'] = r['cathodic_RF']
            if 'average_ECSA_cm2' in r:
                row['Average Q (mC/cm²)'] = r['average_Q_mC_cm2']
                row['Average ECSA (cm²)'] = r['average_ECSA_cm2']
                row['Average RF'] = r['average_RF']
            if 'anodic_ECSA_m2_per_g' in r:
                row['Anodic ECSA (m²/g)'] = r['anodic_ECSA_m2_per_g']
            if 'cathodic_ECSA_m2_per_g' in r:
                row['Cathodic ECSA (m²/g)'] = r['cathodic_ECSA_m2_per_g']
            if 'average_ECSA_m2_per_g' in r:
                row['Average ECSA (m²/g)'] = r['average_ECSA_m2_per_g']
            summary_rows.append(row)

        except Exception as e:
            print(f'         ERROR: {e}')
            continue

    if not all_results:
        print('\n  No files processed successfully.')
        return []

    # ── Write summary CSV ──
    if save_dir and summary_rows:
        csv_path = os.path.join(save_dir, 'ecsa_batch_summary.csv')
        all_cols = list(summary_rows[0].keys())
        for row in summary_rows[1:]:
            for k in row:
                if k not in all_cols:
                    all_cols.append(k)

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=all_cols)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f'\n  Summary CSV: {csv_path}')

    # ── Combined CV overlay ──
    if save_dir and image_ext:
        overlay_path = os.path.join(save_dir, f'ecsa_batch_overlay.{image_ext}')
        plot_ecsa_overlay(all_results, save_path=overlay_path)
        plt.close()
    elif not save_dir:
        plot_ecsa_overlay(all_results)
        plt.show()

    # ── Print summary table ──
    has_loading = any('average_ECSA_m2_per_g' in r for r in all_results)
    print(f'\n  {"═" * 80}')
    hdr = f'  {"Label":25s} {"Anodic":>10s} {"Cathodic":>10s} {"Average":>10s} {"RF":>6s}'
    if has_loading:
        hdr += f' {"m²/g":>8s}'
    print(hdr)
    print(f'  {"":25s} {"(cm²_Pt)":>10s} {"(cm²_Pt)":>10s} {"(cm²_Pt)":>10s}')
    print(f'  {"─" * 80}')
    for r in all_results:
        an = f'{r["anodic_ECSA_cm2"]:.1f}' if 'anodic_ECSA_cm2' in r else '—'
        ca = f'{r["cathodic_ECSA_cm2"]:.1f}' if 'cathodic_ECSA_cm2' in r else '—'
        avg = f'{r["average_ECSA_cm2"]:.1f}' if 'average_ECSA_cm2' in r else '—'
        rf = f'{r["roughness_factor"]:.1f}'
        line = f'  {r["label"]:25s} {an:>10s} {ca:>10s} {avg:>10s} {rf:>6s}'
        if has_loading and 'average_ECSA_m2_per_g' in r:
            line += f' {r["average_ECSA_m2_per_g"]:>8.1f}'
        elif has_loading:
            line += f' {"—":>8s}'
        print(line)
    print(f'  {"═" * 80}\n')

    return all_results


def run_interactive():
    """Walk the user through ECSA analysis step by step."""
    print('\n' + '▓' * 60)
    print('  FUEL CELL ECSA ANALYSIS — INTERACTIVE MODE')
    print('▓' * 60)

    # ── Method selection ──
    print('\n  Methods:')
    print('    1 = H_UPD (single CV file)')
    print('    2 = CO stripping (stripping + baseline CV files)')
    print('    3 = Batch — folder (analyze all ECSA files in a directory)')
    print('    4 = Batch — file list (paste paths one at a time)')
    print('    5 = Run built-in demo')
    method = int(_prompt('Select method', default=1, cast=int))

    if method == 5:
        save = _prompt('Save plots to directory? (path or Enter to show)', default=None, cast=None)
        if save:
            save = _clean_path(save)
        run_demo(save_dir=save if save else None)
        return

    # ── Common parameters ──
    print('\n  ── Electrode Parameters ──')
    geo_area   = _prompt('Geometric area (cm²)', default=5.0)
    scan_rate  = _prompt('Scan rate (mV/s)', default=50.0) / 1000.0  # convert to V/s
    loading    = _prompt('Pt loading (mg/cm², Enter to skip)', default=None)

    # ── File input ──
    print('\n  ── Data Files ──')
    print('  Tip: drag-and-drop a file into the terminal to paste its path')

    if method == 1:
        filepath = _prompt('CV data file path', cast=None)
        filepath = filepath.strip('"').strip("'")

        # ── Test stand presets ──
        print('\n  ── Measurement Test Stand ──')
        print('    0 = Scribner')
        print('    1 = FCTS')
        stand = int(_prompt('Test stand', default=0, cast=int))

        STAND_PRESETS = {
            0: {'name': 'Scribner', 'delimiter': '\t', 'skip': 76,
                'v_col': 5, 'i_col': 1, 'v_low': 0.08, 'v_high': 0.40, 'i_scale': 1.0},
            1: {'name': 'FCTS',     'delimiter': ',',  'skip': 1,
                'v_col': 2, 'i_col': 3, 'v_low': 0.08, 'v_high': 0.40, 'i_scale': 1.0},
        }

        if stand in STAND_PRESETS:
            p = STAND_PRESETS[stand]
            delimiter = p['delimiter']
            skip      = p['skip']
            v_col     = p['v_col']
            i_col     = p['i_col']
            v_low     = p['v_low']
            v_high    = p['v_high']
            i_scale   = p['i_scale']
            delim_name = 'tab' if delimiter == '\t' else 'comma'
            print(f'  → {p["name"]}: {delim_name}-delimited, {skip} header rows,')
            print(f'    V=col {v_col}, I=col {i_col}, {v_low}–{v_high} V, current in A')
        else:
            print(f'  Unknown test stand {stand}, using manual entry')
            print('\n  ── File Format ──')
            delim_choice = _prompt('Delimiter: 1=comma  2=tab  3=semicolon', default=1, cast=int)
            delimiter = {1: ',', 2: '\t', 3: ';'}.get(delim_choice, ',')
            skip      = int(_prompt('Header rows to skip', default=1, cast=int))
            v_col     = int(_prompt('Potential column index (0-based)', default=0, cast=int))
            i_col     = int(_prompt('Current column index (0-based)', default=1, cast=int))
            print('\n  ── Integration Window ──')
            v_low  = _prompt('H_UPD lower bound (V vs RHE)', default=0.08)
            v_high = _prompt('H_UPD upper bound (V vs RHE)', default=0.40)
            print('\n  ── Current Units ──')
            print('    1 = A      2 = mA      3 = µA')
            i_unit = int(_prompt('Current unit', default=1, cast=int))
            i_scale = {1: 1.0, 2: 1e-3, 3: 1e-6}.get(i_unit, 1.0)

        print(f'\n  Loading: {filepath}')
        fcd = parse_fcd_header(filepath)
        if fcd:
            skip = fcd['skip']
            v_col = fcd.get('v_col', v_col)
            i_col = fcd.get('i_col', i_col)
        V, I = load_cv_data(filepath, v_col=v_col, i_col=i_col,
                            delimiter=delimiter, skip_header=skip)
        I *= i_scale

        print(f'  Read {len(V)} data points  |  V range: {V.min():.3f} – {V.max():.3f} V')

        # ── Cycle detection ──
        cycles, turns = extract_cycles(V, I)
        n = len(cycles)
        print(f'  Detected {n} complete cycle{"s" if n != 1 else ""}')

        cycle_choice = 'last'
        if n > 1:
            print('\n  ── Cycle Selection ──')
            print(f'    L = last cycle (most stable, default)')
            print(f'    F = first cycle')
            print(f'    A = average all {n} cycles')
            print(f'    1–{n} = specific cycle number')
            raw = _prompt('Cycle to analyze', default='L', cast=None)
            raw = raw.strip().upper()
            if raw == 'L' or raw == '':
                cycle_choice = 'last'
            elif raw == 'F':
                cycle_choice = 'first'
            elif raw == 'A':
                cycle_choice = 'average'
            else:
                try:
                    cycle_choice = int(raw)
                except ValueError:
                    print(f'  Unrecognized input "{raw}", defaulting to last cycle')
                    cycle_choice = 'last'

        results = compute_ecsa_hupd(V, I, scan_rate, geo_area,
                                     v_low=v_low, v_high=v_high,
                                     loading_mg_cm2=loading,
                                     cycle=cycle_choice)
        print_results(results)

        save = _prompt('\n  Save plot to directory? (path or Enter to show)', default=None, cast=None)
        if save:
            save = _clean_path(save)
        if save:
            os.makedirs(save, exist_ok=True)
            plot_hupd_analysis(results, save_path=os.path.join(save, 'ecsa_hupd.png'))
        else:
            plot_hupd_analysis(results)
            plt.show()

    elif method == 2:
        print('  CO stripping CV (1st cycle after CO adsorption):')
        strip_path = _prompt('    Stripping file path', cast=None).strip('"').strip("'")
        print('  Baseline CV (2nd cycle, clean surface):')
        base_path  = _prompt('    Baseline file path', cast=None).strip('"').strip("'")

        print('\n  ── File Format ──')
        delim_choice = _prompt('Delimiter: 1=comma  2=tab  3=semicolon', default=1, cast=int)
        delimiter = {1: ',', 2: '\t', 3: ';'}.get(delim_choice, ',')
        skip      = int(_prompt('Header rows to skip', default=1, cast=int))
        v_col     = int(_prompt('Potential column index (0-based)', default=0, cast=int))
        i_col     = int(_prompt('Current column index (0-based)', default=1, cast=int))

        print('\n  ── Current Units ──')
        print('    1 = A      2 = mA      3 = µA')
        i_unit = int(_prompt('Current unit', default=1, cast=int))
        i_scale = {1: 1.0, 2: 1e-3, 3: 1e-6}.get(i_unit, 1.0)

        print('\n  ── Integration Window ──')
        v_low  = _prompt('CO oxidation lower bound (V vs RHE)', default=0.40)
        v_high = _prompt('CO oxidation upper bound (V vs RHE)', default=1.00)

        fcd_s = parse_fcd_header(strip_path)
        fcd_b = parse_fcd_header(base_path)
        V_s, I_s = load_cv_data(strip_path, v_col=fcd_s.get('v_col', v_col) if fcd_s else v_col,
                                 i_col=fcd_s.get('i_col', i_col) if fcd_s else i_col,
                                 delimiter=delimiter, skip_header=fcd_s['skip'] if fcd_s else skip)
        V_b, I_b = load_cv_data(base_path, v_col=fcd_b.get('v_col', v_col) if fcd_b else v_col,
                                 i_col=fcd_b.get('i_col', i_col) if fcd_b else i_col,
                                 delimiter=delimiter, skip_header=fcd_b['skip'] if fcd_b else skip)
        I_s *= i_scale
        I_b *= i_scale

        print(f'  Stripping: {len(V_s)} pts  |  Baseline: {len(V_b)} pts')

        # ── Cycle detection ──
        cycles_s, _ = extract_cycles(V_s, I_s)
        cycles_b, _ = extract_cycles(V_b, I_b)
        print(f'  Stripping file: {len(cycles_s)} cycle{"s" if len(cycles_s)!=1 else ""}  |  '
              f'Baseline file: {len(cycles_b)} cycle{"s" if len(cycles_b)!=1 else ""}')

        cycle_strip = 'first'
        cycle_base = 'last'
        if len(cycles_s) > 1 or len(cycles_b) > 1:
            print('\n  ── Cycle Selection ──')
            if len(cycles_s) > 1:
                print(f'  Stripping file has {len(cycles_s)} cycles.')
                print(f'    Typically cycle 1 contains the CO oxidation peak.')
                raw = _prompt('  Stripping cycle (F=first, L=last, or number)', default='F', cast=None).strip().upper()
                if raw == 'F' or raw == '': cycle_strip = 'first'
                elif raw == 'L': cycle_strip = 'last'
                else:
                    try: cycle_strip = int(raw)
                    except ValueError: cycle_strip = 'first'

            if len(cycles_b) > 1:
                print(f'  Baseline file has {len(cycles_b)} cycles.')
                print(f'    Typically the last cycle is cleanest.')
                raw = _prompt('  Baseline cycle (F=first, L=last, or number)', default='L', cast=None).strip().upper()
                if raw == 'L' or raw == '': cycle_base = 'last'
                elif raw == 'F': cycle_base = 'first'
                else:
                    try: cycle_base = int(raw)
                    except ValueError: cycle_base = 'last'

        results = compute_ecsa_co_strip(V_s, I_s, V_b, I_b,
                                         scan_rate, geo_area,
                                         v_low=v_low, v_high=v_high,
                                         loading_mg_cm2=loading,
                                         cycle_strip=cycle_strip,
                                         cycle_base=cycle_base)
        print_results(results)

        save = _prompt('\n  Save plot to directory? (path or Enter to show)', default=None, cast=None)
        if save:
            save = _clean_path(save)
        if save:
            os.makedirs(save, exist_ok=True)
            plot_co_stripping(results, save_path=os.path.join(save, 'ecsa_co_strip.png'))
        else:
            plot_co_stripping(results)
            plt.show()

    elif method in (3, 4):
        # ── Batch processing ──
        print('\n  ── Data Files ──')
        print('  Tip: drag-and-drop a file or folder into the terminal')

        # ── Test stand presets ──
        print('\n  ── Measurement Test Stand ──')
        print('    0 = Scribner')
        print('    1 = FCTS')
        stand = int(_prompt('Test stand', default=0, cast=int))

        STAND_PRESETS = {
            0: {'name': 'Scribner', 'delimiter': '\t', 'skip': 76,
                'v_col': 5, 'i_col': 1, 'v_low': 0.08, 'v_high': 0.40, 'i_scale': 1.0},
            1: {'name': 'FCTS',     'delimiter': ',',  'skip': 1,
                'v_col': 2, 'i_col': 3, 'v_low': 0.08, 'v_high': 0.40, 'i_scale': 1.0},
        }

        if stand in STAND_PRESETS:
            p = STAND_PRESETS[stand]
            b_delimiter = p['delimiter']
            b_skip      = p['skip']
            b_v_col     = p['v_col']
            b_i_col     = p['i_col']
            b_v_low     = p['v_low']
            b_v_high    = p['v_high']
            b_i_scale   = p['i_scale']
            delim_name = 'tab' if b_delimiter == '\t' else 'comma'
            print(f'  → {p["name"]}: {delim_name}-delimited, {b_skip} header rows,')
            print(f'    V=col {b_v_col}, I=col {b_i_col}, {b_v_low}–{b_v_high} V, current in A')
        else:
            print(f'  Unknown test stand {stand}, using manual entry')
            print('\n  ── File Format ──')
            delim_choice = _prompt('Delimiter: 1=comma  2=tab  3=semicolon', default=1, cast=int)
            b_delimiter = {1: ',', 2: '\t', 3: ';'}.get(delim_choice, ',')
            b_skip      = int(_prompt('Header rows to skip', default=1, cast=int))
            b_v_col     = int(_prompt('Potential column index (0-based)', default=0, cast=int))
            b_i_col     = int(_prompt('Current column index (0-based)', default=1, cast=int))
            print('\n  ── Integration Window ──')
            b_v_low  = _prompt('H_UPD lower bound (V vs RHE)', default=0.08)
            b_v_high = _prompt('H_UPD upper bound (V vs RHE)', default=0.40)
            print('\n  ── Current Units ──')
            print('    1 = A      2 = mA      3 = µA')
            i_unit = int(_prompt('Current unit', default=1, cast=int))
            b_i_scale = {1: 1.0, 2: 1e-3, 3: 1e-6}.get(i_unit, 1.0)

        # ── Cycle selection for batch ──
        print('\n  ── Cycle Selection (applied to all files) ──')
        print('    L = last cycle (most stable, default)')
        print('    F = first cycle')
        print('    A = average all cycles')
        raw = _prompt('Cycle to analyze', default='L', cast=None).strip().upper()
        if raw == 'F':
            batch_cycle = 'first'
        elif raw == 'A':
            batch_cycle = 'average'
        else:
            batch_cycle = 'last'

        save_dir = _clean_path(_prompt('Output directory for results', default='ecsa_batch_output', cast=None))

        filepaths = []
        labels = []

        if method == 3:
            # ── Folder batch ──
            folder = _prompt('Folder path containing ECSA files', cast=None)
            folder = _clean_path(folder)

            extensions = ['*.csv', '*.txt', '*.tsv', '*.fcd', '*.CSV', '*.TXT', '*.TSV', '*.FCD']
            for ext in extensions:
                filepaths.extend(glob.glob(os.path.join(folder, ext)))
            filepaths = sorted(set(filepaths))

            # Only include files with "ECSA" or "CV-50mVs" in the filename
            filepaths = [fp for fp in filepaths
                         if 'ECSA' in os.path.basename(fp).upper()
                         or 'CV-50MVS' in os.path.basename(fp).upper()]

            if not filepaths:
                print(f'  No files containing "ECSA" or "CV-50mVs" found in: {folder}')
                return

            print(f'  Found {len(filepaths)} ECSA files:')
            for fp in filepaths:
                name = os.path.splitext(os.path.basename(fp))[0]
                labels.append(name)
                print(f'    {name}')

        elif method == 4:
            # ── File list batch ──
            print('  Enter file paths one per line. Type "done" when finished.')
            while True:
                raw = input('  File path (or "done"): ').strip()
                if raw.lower() == 'done':
                    break
                fp = _clean_path(raw)
                if not fp:
                    continue
                if not os.path.isfile(fp):
                    print(f'    File not found: {fp}')
                    continue
                default_label = os.path.splitext(os.path.basename(fp))[0]
                lbl = _prompt('  Label', default=default_label, cast=None)
                filepaths.append(fp)
                labels.append(lbl)

            if not filepaths:
                print('  No files entered.')
                return

        run_batch(filepaths, labels, scan_rate, geo_area, loading,
                  b_delimiter, b_skip, b_v_col, b_i_col, b_v_low, b_v_high,
                  b_i_scale, cycle=batch_cycle, save_dir=save_dir)


def main():
    parser = argparse.ArgumentParser(description='Fuel Cell ECSA Analysis')
    parser.add_argument('--file', type=str, help='CSV file with CV data (V, I columns)')
    parser.add_argument('--scan-rate', type=float, default=0.050,
                        help='Scan rate in V/s (default: 0.050)')
    parser.add_argument('--area', type=float, default=5.0,
                        help='Geometric electrode area in cm² (default: 5.0)')
    parser.add_argument('--loading', type=float, default=None,
                        help='Pt loading in mg/cm² (optional)')
    parser.add_argument('--v-low', type=float, default=0.05,
                        help='H_UPD lower bound in V (default: 0.05)')
    parser.add_argument('--v-high', type=float, default=0.40,
                        help='H_UPD upper bound in V (default: 0.40)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots (default: show interactively)')
    parser.add_argument('--demo', action='store_true',
                        help='Run built-in demo with synthetic data')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run interactive guided mode')
    parser.add_argument('--cycle', type=str, default='last',
                        help='Cycle to analyze: last, first, average, or number (default: last)')
    args = parser.parse_args()

    # Parse cycle argument
    cycle_choice = args.cycle
    if cycle_choice not in ('last', 'first', 'average'):
        try:
            cycle_choice = int(cycle_choice)
        except ValueError:
            cycle_choice = 'last'

    if args.interactive:
        run_interactive()
    elif args.demo:
        run_demo(save_dir=args.save_dir)
    elif args.file:
        V, I = load_cv_data(args.file)
        cycles, _ = extract_cycles(V, I)
        print(f'  Detected {len(cycles)} cycle{"s" if len(cycles)!=1 else ""} — using: {cycle_choice}')
        results = compute_ecsa_hupd(V, I, args.scan_rate, args.area,
                                     v_low=args.v_low, v_high=args.v_high,
                                     loading_mg_cm2=args.loading,
                                     cycle=cycle_choice)
        print_results(results)
        plot_hupd_analysis(results,
                           save_path=os.path.join(args.save_dir, 'ecsa_hupd.png')
                           if args.save_dir else None)
        if args.save_dir is None:
            plt.show()
    else:
        # No arguments at all → launch interactive mode
        run_interactive()


if __name__ == '__main__':
    main()
