#!/usr/bin/env python3
"""
Fuel Cell EIS Analysis
======================
Analyzes electrochemical impedance spectroscopy data for PEM fuel cells.

Features:
  - Nyquist and Bode plots
  - HFR (high-frequency resistance) extraction
  - Equivalent circuit fitting (R-RC, R-RC-RC, Randles+Warburg)
  - Parameter extraction: R_ohm, R_ct, R_mt, CPE elements
  - Multi-file overlay for degradation tracking

Usage:
  python eis_analysis.py                    # interactive mode
  python eis_analysis.py --file data.csv    # analyze EIS data
  python eis_analysis.py --demo             # run built-in demo
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.helpers.plot_compare import save_with_sidecar
from scipy.optimize import least_squares
from scipy.integrate import trapezoid
import argparse
import csv
import os
import glob


def run(input_dir: str, output_dir: str, params: dict = None) -> dict:
    """Batch EIS analysis: fit impedance spectra and generate Nyquist plots."""
    from pathlib import Path
    p = params or {}

    inp = Path(input_dir)
    all_files = sorted([f for f in inp.rglob("*")
                    if f.is_file() and f.suffix.lower() in ('.csv', '.txt', '.tsv', '.fcd')])

    if not all_files:
        return {"status": "error", "message": "No data files found"}

    # Filter by EIS keywords; fall back to all files if none match
    KEYWORDS = ['EIS']
    filtered = [f for f in all_files
                if any(kw in f.name.upper() for kw in KEYWORDS)]
    files = filtered if filtered else all_files

    has_tab_files = any(f.suffix.lower() in ('.fcd', '.tsv') for f in files)
    delimiter = '\t' if has_tab_files else ','

    filepaths = [str(f) for f in files]
    labels = [f.stem for f in files]

    from scripts.helpers.conditions import img_ext_from_params
    image_ext = img_ext_from_params(p)

    results = run_batch(
        filepaths, labels,
        model_name=p.get('model_name', 'R-RC'),
        geo_area=float(p.get('geo_area', 5.0)),
        delimiter=delimiter,
        skip=1,
        freq_col=0,
        zreal_col=1,
        zimag_col=2,
        zimag_sign=1,
        save_dir=str(output_dir),
        image_ext=image_ext,
    )

    output_files = [str(f.relative_to(Path(output_dir))) for f in Path(output_dir).rglob('*') if f.is_file()]
    if not output_files:
        raise RuntimeError(
            f'Analysis produced no output. {len(files)} file(s) were found '
            f'but none could be processed. Check file format and parameters.'
        )
    return {"status": "success", "files_processed": len(files), "files_produced": output_files}


# ═══════════════════════════════════════════════════════════════════════
#  Data I/O
# ═══════════════════════════════════════════════════════════════════════

def load_eis_data(filepath, freq_col=0, zreal_col=1, zimag_col=2,
                  delimiter=',', skip_header=1, zimag_sign='negative'):
    """
    Load EIS data from a CSV/TSV file.

    Parameters
    ----------
    filepath : str
        Path to data file.
    freq_col, zreal_col, zimag_col : int
        Column indices for frequency (Hz), Z_real (Ohm), Z_imag (Ohm).
    delimiter : str
        Column delimiter.
    skip_header : int
        Number of header rows to skip.
    zimag_sign : str
        'negative' — file stores Z_imag as negative (convention: capacitive arc
        below x-axis in raw data, plotted with -Z_imag on y-axis).
        'positive' — file already stores -Z_imag (positive for capacitive arc).

    Returns
    -------
    freq, Z_real, Z_imag : arrays
        Frequency in Hz, Z' in Ohm, Z'' in Ohm (Z'' negative for capacitive).
    """
    freq, zr, zi = [], [], []
    # Clean filepath: PowerShell drag-drop adds "& '...'" wrapper,
    # cmd "Copy as path" adds quotes, and Unicode invisible chars appear too.
    filepath = filepath.strip()
    if filepath.startswith('& '):
        filepath = filepath[2:]
    filepath = filepath.strip().strip('"').strip("'")
    filepath = filepath.strip('\u2018\u2019\u201c\u201d')  # smart quotes
    filepath = filepath.strip('\u202a\u200b')
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f'File not found: "{filepath}"\n'
            f'  Check the path and try again. On Windows, right-click the file\n'
            f'  in Explorer → Copy as path, then paste into the terminal.')
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for _ in range(skip_header):
            next(reader)
        for row in reader:
            try:
                freq.append(float(row[freq_col]))
                zr.append(float(row[zreal_col]))
                zi.append(float(row[zimag_col]))
            except (ValueError, IndexError):
                continue

    freq = np.array(freq)
    zr = np.array(zr)
    zi = np.array(zi)

    # Ensure Z_imag follows the convention: negative for capacitive arcs
    if zimag_sign == 'positive':
        zi = -zi  # convert -Z'' (positive) to Z'' (negative)

    # Sort by descending frequency (high → low)
    idx = np.argsort(freq)[::-1]
    return freq[idx], zr[idx], zi[idx]


# ═══════════════════════════════════════════════════════════════════════
#  HFR Extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_hfr(freq, Z_real, Z_imag, method='x_intercept'):
    """
    Extract the high-frequency resistance (HFR / R_ohm).

    Parameters
    ----------
    freq, Z_real, Z_imag : array
        EIS data sorted by descending frequency.
    method : str
        'x_intercept' — find where -Z_imag crosses zero (interpolated).
        'min_zimag'   — Z_real at the minimum |Z_imag| in the HF region.
        'highest_freq' — Z_real at the highest measured frequency.

    Returns
    -------
    float : HFR in Ohm.
    """
    if method == 'highest_freq':
        return Z_real[0]

    if method == 'min_zimag':
        # Use the upper half of the frequency range
        n_hf = max(len(freq) // 2, 3)
        idx = np.argmin(np.abs(Z_imag[:n_hf]))
        return Z_real[idx]

    # x_intercept: interpolate where -Z_imag crosses zero from positive to negative
    neg_zimag = -Z_imag  # positive = capacitive arc
    for i in range(len(neg_zimag) - 1):
        # Look for crossing from negative (inductive) to positive (capacitive)
        # or simply the first zero crossing at high frequency
        if neg_zimag[i] <= 0 and neg_zimag[i + 1] > 0:
            # Linear interpolation
            frac = -neg_zimag[i] / (neg_zimag[i + 1] - neg_zimag[i])
            hfr = Z_real[i] + frac * (Z_real[i + 1] - Z_real[i])
            return hfr

    # Fallback: if no crossing found, use min |Z_imag|
    return extract_hfr(freq, Z_real, Z_imag, method='min_zimag')


# ═══════════════════════════════════════════════════════════════════════
#  Equivalent Circuit Models
# ═══════════════════════════════════════════════════════════════════════

def _Z_cpe(omega, Q, n):
    """Impedance of a constant phase element: Z = 1 / (Q * (j*omega)^n)."""
    # Guard against zero frequency
    omega_safe = np.where(omega > 0, omega, 1e-30)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        Z = 1.0 / (Q * (1j * omega_safe) ** n)
    return Z


def _Z_warburg(omega, Aw):
    """Finite-frequency Warburg impedance (semi-infinite): Z = Aw / sqrt(omega) * (1 - j)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        Zw = np.where(omega > 0,
                      Aw / np.sqrt(omega) * (1 - 1j),
                      np.inf + 0j)
    return Zw


def model_R_RC(params, omega):
    """
    R-(RC) model: R_ohm in series with R_ct || CPE_dl.

    params: [R_ohm, R_ct, Q_dl, n_dl]
    """
    R_ohm, R_ct, Q_dl, n_dl = params
    Z_cpe = _Z_cpe(omega, Q_dl, n_dl)
    Z_arc = 1.0 / (1.0 / R_ct + 1.0 / Z_cpe)
    return R_ohm + Z_arc


def model_R_RC_RC(params, omega):
    """
    R-(RC)(RC) model: R_ohm + (R_ct || CPE_ct) + (R_mt || CPE_mt).

    params: [R_ohm, R_ct, Q_ct, n_ct, R_mt, Q_mt, n_mt]
    """
    R_ohm, R_ct, Q_ct, n_ct, R_mt, Q_mt, n_mt = params
    Z_cpe1 = _Z_cpe(omega, Q_ct, n_ct)
    Z_arc1 = 1.0 / (1.0 / R_ct + 1.0 / Z_cpe1)
    Z_cpe2 = _Z_cpe(omega, Q_mt, n_mt)
    Z_arc2 = 1.0 / (1.0 / R_mt + 1.0 / Z_cpe2)
    return R_ohm + Z_arc1 + Z_arc2


def model_Randles_W(params, omega):
    """
    Randles + Warburg: R_ohm + (R_ct + W) || CPE_dl.

    params: [R_ohm, R_ct, Q_dl, n_dl, Aw]
    """
    R_ohm, R_ct, Q_dl, n_dl, Aw = params
    Zw = _Z_warburg(omega, Aw)
    Z_cpe = _Z_cpe(omega, Q_dl, n_dl)
    Z_branch = R_ct + Zw
    Z_arc = 1.0 / (1.0 / Z_branch + 1.0 / Z_cpe)
    return R_ohm + Z_arc


MODELS = {
    'R-RC': {
        'func': model_R_RC,
        'param_names': ['R_ohm', 'R_ct', 'Q_dl', 'n_dl'],
        'units': ['Ω·cm²', 'Ω·cm²', 'F·s^(n-1)/cm²', ''],
        'n_params': 4,
        'description': 'R_ohm + (R_ct || CPE_dl)',
    },
    'R-RC-RC': {
        'func': model_R_RC_RC,
        'param_names': ['R_ohm', 'R_ct', 'Q_ct', 'n_ct', 'R_mt', 'Q_mt', 'n_mt'],
        'units': ['Ω·cm²', 'Ω·cm²', 'F·s^(n-1)/cm²', '', 'Ω·cm²', 'F·s^(n-1)/cm²', ''],
        'n_params': 7,
        'description': 'R_ohm + (R_ct || CPE_ct) + (R_mt || CPE_mt)',
    },
    'Randles-W': {
        'func': model_Randles_W,
        'param_names': ['R_ohm', 'R_ct', 'Q_dl', 'n_dl', 'Aw'],
        'units': ['Ω·cm²', 'Ω·cm²', 'F·s^(n-1)/cm²', '', 'Ω·s^0.5/cm²'],
        'n_params': 5,
        'description': 'R_ohm + (R_ct + W) || CPE_dl',
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Fitting
# ═══════════════════════════════════════════════════════════════════════

def _residuals(params, omega, Z_data, model_func):
    """Weighted residuals for complex impedance fitting."""
    Z_model = model_func(params, omega)
    diff = Z_model - Z_data
    # Weight by 1/|Z| to give equal importance to HF and LF arcs
    Z_mag = np.abs(Z_data)
    Z_mag = np.where(Z_mag > 0, Z_mag, 1e-30)  # avoid division by zero
    weight = 1.0 / Z_mag
    res = np.concatenate([diff.real * weight, diff.imag * weight])
    # Replace any non-finite residuals with a large penalty
    res = np.where(np.isfinite(res), res, 1e6)
    return res


def _auto_initial_guess(freq, Z_real, Z_imag, model_name):
    """Generate reasonable initial guesses from the data."""
    hfr = extract_hfr(freq, Z_real, Z_imag)
    hfr = max(hfr, 1e-6)  # ensure positive

    R_total = Z_real[-1] - hfr   # total arc width
    if R_total <= 0:
        # Data may have inductive loop or unusual shape; estimate from max Z_real
        R_total = Z_real.max() - hfr
    R_total = max(R_total, 1e-6)

    # Estimate dominant time constant from peak frequency of -Z_imag
    neg_zi = -Z_imag
    # Only look at points where -Z_imag is positive (capacitive region)
    capacitive = neg_zi > 0
    if capacitive.any():
        peak_idx = np.where(capacitive)[0][np.argmax(neg_zi[capacitive])]
        f_peak = freq[peak_idx]
    else:
        f_peak = np.sqrt(freq.max() * max(freq.min(), 0.1))

    f_peak = max(f_peak, 0.1)  # floor to avoid division issues
    tau = 1.0 / (2 * np.pi * f_peak)

    if model_name == 'R-RC':
        R_ct = R_total
        Q_dl = max(tau / R_ct, 1e-6)
        return [hfr, R_ct, Q_dl, 0.85]

    elif model_name == 'R-RC-RC':
        R_ct = R_total * 0.6
        R_mt = R_total * 0.4
        Q_ct = max(tau / R_ct, 1e-6)
        Q_mt = max(tau * 5 / R_mt, 1e-6)
        return [hfr, R_ct, Q_ct, 0.85, R_mt, Q_mt, 0.80]

    elif model_name == 'Randles-W':
        R_ct = R_total * 0.7
        Q_dl = max(tau / R_ct, 1e-6)
        Aw = R_total * 0.3 * np.sqrt(2 * np.pi * max(freq.min(), 0.1))
        return [hfr, R_ct, Q_dl, 0.85, max(Aw, 1e-4)]


def _param_bounds(model_name, hfr_est):
    """Parameter bounds for fitting."""
    lb_R = 0.0
    ub_R = 100.0
    lb_Q = 1e-8
    ub_Q = 10.0
    lb_n = 0.3
    ub_n = 1.0

    if model_name == 'R-RC':
        lb = [lb_R, lb_R, lb_Q, lb_n]
        ub = [ub_R, ub_R, ub_Q, ub_n]
    elif model_name == 'R-RC-RC':
        lb = [lb_R, lb_R, lb_Q, lb_n, lb_R, lb_Q, lb_n]
        ub = [ub_R, ub_R, ub_Q, ub_n, ub_R, ub_Q, ub_n]
    elif model_name == 'Randles-W':
        lb = [lb_R, lb_R, lb_Q, lb_n, 0.0]
        ub = [ub_R, ub_R, ub_Q, ub_n, 100.0]
    else:
        raise ValueError(f'Unknown model: {model_name}')

    return lb, ub


def fit_eis(freq, Z_real, Z_imag, model_name='R-RC', geo_area=None, p0=None):
    """
    Fit an equivalent circuit model to EIS data.

    Parameters
    ----------
    freq, Z_real, Z_imag : array
        EIS data (Z_imag negative for capacitive arcs).
    model_name : str
        One of 'R-RC', 'R-RC-RC', 'Randles-W'.
    geo_area : float or None
        If provided, impedance is assumed in Ohm and will be reported in Ω·cm².
    p0 : list or None
        Initial guess (auto-generated if None).

    Returns
    -------
    dict with fitted parameters, model info, and data for plotting.
    """
    if model_name not in MODELS:
        raise ValueError(f'Unknown model: {model_name}. Choose from {list(MODELS.keys())}')

    model = MODELS[model_name]
    Z_data = Z_real + 1j * Z_imag

    # Filter out zero/negative frequencies
    valid = freq > 0
    if not np.all(valid):
        print(f'  Warning: removing {np.sum(~valid)} points with freq <= 0')
        freq = freq[valid]
        Z_data = Z_data[valid]

    omega = 2 * np.pi * freq

    # Scale to area-specific if needed
    scale = geo_area if geo_area is not None else 1.0
    Z_data_scaled = Z_data * scale
    Z_real_s = Z_data_scaled.real
    Z_imag_s = Z_data_scaled.imag

    if p0 is None:
        p0 = _auto_initial_guess(freq, Z_real_s, Z_imag_s, model_name)

    hfr_est = extract_hfr(freq, Z_real_s, Z_imag_s)
    lb, ub = _param_bounds(model_name, hfr_est)

    # Clip initial guess to bounds
    p0 = [max(lb[i], min(ub[i], p0[i])) for i in range(len(p0))]

    # Validate initial residuals
    r0 = _residuals(p0, omega, Z_data_scaled, model['func'])
    if not np.all(np.isfinite(r0)):
        print('  Warning: initial guess produced non-finite residuals.')
        print(f'    p0 = {[f"{v:.4g}" for v in p0]}')
        print(f'    freq range: {freq.min():.2g} – {freq.max():.2g} Hz')
        print(f'    Z range: {np.abs(Z_data_scaled).min():.4g} – {np.abs(Z_data_scaled).max():.4g}')
        print('    Retrying with conservative fallback guess...')
        # Fallback: simple guess from data range
        Z_mag_mid = np.median(np.abs(Z_data_scaled))
        if model_name == 'R-RC':
            p0 = [Z_mag_mid * 0.3, Z_mag_mid * 0.7, 0.001, 0.85]
        elif model_name == 'R-RC-RC':
            p0 = [Z_mag_mid * 0.2, Z_mag_mid * 0.4, 0.001, 0.85,
                   Z_mag_mid * 0.3, 0.01, 0.80]
        elif model_name == 'Randles-W':
            p0 = [Z_mag_mid * 0.3, Z_mag_mid * 0.5, 0.001, 0.85, 0.01]
        p0 = [max(lb[i], min(ub[i], p0[i])) for i in range(len(p0))]

    result = least_squares(
        _residuals, p0, args=(omega, Z_data_scaled, model['func']),
        bounds=(lb, ub), method='trf', max_nfev=10000
    )

    pfit = result.x
    Z_fit = model['func'](pfit, omega)

    # Goodness of fit
    ss_res = np.sum(np.abs(Z_fit - Z_data_scaled) ** 2)
    ss_tot = np.sum(np.abs(Z_data_scaled - np.mean(Z_data_scaled)) ** 2)
    R_squared = 1.0 - ss_res / ss_tot

    # Generate smooth fit curve for plotting
    f_smooth = np.logspace(np.log10(freq.min()), np.log10(freq.max()), 500)
    omega_smooth = 2 * np.pi * f_smooth
    Z_fit_smooth = model['func'](pfit, omega_smooth)

    results = {
        'model_name': model_name,
        'description': model['description'],
        'param_names': model['param_names'],
        'param_values': pfit.tolist(),
        'param_units': model['units'],
        'R_squared': R_squared,
        'cost': result.cost,
        'geo_area': geo_area,
        'HFR': extract_hfr(freq, Z_real_s, Z_imag_s),
        # Data for plotting
        '_freq': freq, '_Z_real': Z_real_s, '_Z_imag': Z_imag_s,
        '_Z_fit_real': Z_fit.real, '_Z_fit_imag': Z_fit.imag,
        '_f_smooth': f_smooth,
        '_Z_smooth_real': Z_fit_smooth.real,
        '_Z_smooth_imag': Z_fit_smooth.imag,
    }

    # Extract physically meaningful summary
    results['R_ohm'] = pfit[0]
    if model_name == 'R-RC':
        results['R_ct'] = pfit[1]
        results['R_total'] = pfit[0] + pfit[1]
    elif model_name == 'R-RC-RC':
        results['R_ct'] = pfit[1]
        results['R_mt'] = pfit[4]
        results['R_total'] = pfit[0] + pfit[1] + pfit[4]
    elif model_name == 'Randles-W':
        results['R_ct'] = pfit[1]
        results['R_total'] = pfit[0] + pfit[1]

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_eis(results, save_path=None):
    """
    4-panel EIS plot: Nyquist, Bode magnitude, Bode phase, and
    parameter summary.
    """
    freq = results['_freq']
    Zr = results['_Z_real']
    Zi = results['_Z_imag']
    f_sm = results['_f_smooth']
    Zr_sm = results['_Z_smooth_real']
    Zi_sm = results['_Z_smooth_imag']

    area_label = 'Ω·cm²' if results['geo_area'] else 'Ω'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── Panel 1: Nyquist ──
    ax = axes[0]
    ax.plot(Zr, -Zi, 'o', color='steelblue', ms=5, alpha=0.7, label='Data')
    ax.plot(Zr_sm, -Zi_sm, '-', color='firebrick', lw=1.5, label=f'Fit ({results["model_name"]})')
    # Mark HFR
    hfr = results['HFR']
    ax.axvline(hfr, color='green', ls='--', lw=1.0, alpha=0.7, label=f'HFR = {hfr:.4f} {area_label}')
    ax.set_xlabel(f"Z' ({area_label})")
    ax.set_ylabel(f"−Z'' ({area_label})")
    ax.set_title('Nyquist Plot')
    ax.legend(fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Bode magnitude ──
    ax = axes[1]
    Z_mag = np.sqrt(Zr**2 + Zi**2)
    Z_mag_sm = np.sqrt(Zr_sm**2 + Zi_sm**2)
    ax.loglog(freq, Z_mag, 'o', color='steelblue', ms=5, alpha=0.7, label='Data')
    ax.loglog(f_sm, Z_mag_sm, '-', color='firebrick', lw=1.5, label='Fit')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(f'|Z| ({area_label})')
    ax.set_title('Bode — Magnitude')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # ── Panel 3: Bode phase ──
    ax = axes[2]
    phase = np.degrees(np.arctan2(-Zi, Zr))
    phase_sm = np.degrees(np.arctan2(-Zi_sm, Zr_sm))
    ax.semilogx(freq, phase, 'o', color='steelblue', ms=5, alpha=0.7, label='Data')
    ax.semilogx(f_sm, phase_sm, '-', color='firebrick', lw=1.5, label='Fit')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('−Phase (°)')
    ax.set_title('Bode — Phase')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    from scripts.helpers.conditions import get_condition_label
    cond_label = get_condition_label(label=results.get('label', ''))
    title = f'EIS Analysis — {results["description"]}  (R² = {results["R_squared"]:.4f})'
    if cond_label:
        title += f'\n{cond_label}'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    fig.tight_layout()

    if save_path:
        save_with_sidecar(fig, save_path, plot_type='eis', dpi=200, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


def plot_nyquist_overlay(datasets, labels=None, save_path=None):
    """
    Overlay multiple Nyquist plots for degradation tracking.

    Parameters
    ----------
    datasets : list of (freq, Z_real, Z_imag) tuples
    labels : list of str or None
    """
    from scripts.helpers.conditions import get_condition_label

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.viridis
    n = len(datasets)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for i, (freq, Zr, Zi) in enumerate(datasets):
        lbl = labels[i] if labels else f'Dataset {i+1}'
        cond = get_condition_label(label=lbl, compact=True)
        legend_lbl = f'{lbl}\n  {cond}' if cond else lbl
        ax.plot(Zr, -Zi, 'o-', color=colors[i], ms=4, lw=1.2, alpha=0.8, label=legend_lbl)

    ax.set_xlabel("Z' (Ω·cm²)")
    ax.set_ylabel("−Z'' (Ω·cm²)")
    ax.set_title('Nyquist Overlay')
    ax.legend(fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_with_sidecar(fig, save_path, plot_type='nyquist_overlay', dpi=200, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Print Results
# ═══════════════════════════════════════════════════════════════════════

def print_results(r):
    """Pretty-print EIS fit results."""
    area_label = 'Ω·cm²' if r['geo_area'] else 'Ω'

    print(f'\n{"═" * 55}')
    print(f'  Model:        {r["model_name"]}  ({r["description"]})')
    print(f'  R²:           {r["R_squared"]:.6f}')
    if r['geo_area']:
        print(f'  Geo. area:    {r["geo_area"]:.2f} cm²')
    print(f'  {"─" * 51}')
    print(f'  {"Parameter":20s} {"Value":>14s}  {"Unit"}')
    print(f'  {"─" * 51}')
    for name, val, unit in zip(r['param_names'], r['param_values'], r['param_units']):
        if unit:
            print(f'  {name:20s} {val:>14.6g}  {unit}')
        else:
            print(f'  {name:20s} {val:>14.4f}')
    print(f'  {"─" * 51}')
    print(f'  {"HFR (x-intercept)":20s} {r["HFR"]:>14.6g}  {area_label}')
    print(f'  {"R_ohm (fit)":20s} {r["R_ohm"]:>14.6g}  {area_label}')
    if 'R_ct' in r:
        print(f'  {"R_ct":20s} {r["R_ct"]:>14.6g}  {area_label}')
    if 'R_mt' in r:
        print(f'  {"R_mt":20s} {r["R_mt"]:>14.6g}  {area_label}')
    print(f'  {"R_total":20s} {r["R_total"]:>14.6g}  {area_label}')
    print(f'{"═" * 55}\n')


# ═══════════════════════════════════════════════════════════════════════
#  Synthetic Data Generator
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_eis(model='R-RC-RC', geo_area=5.0, noise=0.002):
    """
    Generate synthetic EIS data for a PEM fuel cell.

    Parameters are specified in Ω·cm². Returns freq, Z_real, Z_imag in Ohm
    (as a potentiostat would measure), so the fit function can scale back
    to area-specific by multiplying by geo_area.
    """
    freq = np.logspace(np.log10(0.1), np.log10(1e5), 60)
    omega = 2 * np.pi * freq

    if model == 'R-RC':
        # Area-specific params: [R_ohm, R_ct, Q_dl, n_dl]  (Ω·cm²)
        params_area = [0.05, 0.15, 0.01, 0.88]
    elif model == 'R-RC-RC':
        # [R_ohm, R_ct, Q_ct, n_ct, R_mt, Q_mt, n_mt]  (Ω·cm²)
        params_area = [0.05, 0.10, 0.005, 0.88, 0.08, 0.05, 0.75]
    else:
        # Randles-W: [R_ohm, R_ct, Q_dl, n_dl, Aw]
        params_area = [0.05, 0.12, 0.008, 0.85, 0.03]

    # Compute Z in Ω·cm² then convert to Ohm
    model_func = MODELS[model]['func']
    Z_area = model_func(params_area, omega)   # Ω·cm²
    Z_ohm = Z_area / geo_area                 # Ohm

    # Add noise
    noise_scale = noise * np.abs(Z_ohm).max()
    Z_ohm += noise_scale * (np.random.randn(len(Z_ohm)) + 1j * np.random.randn(len(Z_ohm)))

    return freq, Z_ohm.real, Z_ohm.imag, params_area


# ═══════════════════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════════════════

def run_demo(save_dir=None):
    """Run full demo with synthetic data."""
    print('\n' + '▓' * 60)
    print('  FUEL CELL EIS ANALYSIS — DEMO')
    print('▓' * 60)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    np.random.seed(42)
    geo_area = 5.0

    # ── Demo 1: R-RC-RC fit ──
    print('\n[1] R-RC-RC Model (two-arc)')
    freq, Zr, Zi, true_p = generate_synthetic_eis(model='R-RC-RC', geo_area=geo_area)
    print(f'    True params (Ω·cm²): R_ohm={true_p[0]}, R_ct={true_p[1]}, R_mt={true_p[4]}')

    r = fit_eis(freq, Zr, Zi, model_name='R-RC-RC', geo_area=geo_area)
    print_results(r)

    p1 = os.path.join(save_dir, 'eis_rrcrc.png') if save_dir else None
    plot_eis(r, save_path=p1)

    # ── Demo 2: R-RC fit ──
    print('\n[2] R-RC Model (single arc)')
    freq2, Zr2, Zi2, true_p2 = generate_synthetic_eis(model='R-RC', geo_area=geo_area)
    print(f'    True params (Ω·cm²): R_ohm={true_p2[0]}, R_ct={true_p2[1]}')

    r2 = fit_eis(freq2, Zr2, Zi2, model_name='R-RC', geo_area=geo_area)
    print_results(r2)

    p2 = os.path.join(save_dir, 'eis_rrc.png') if save_dir else None
    plot_eis(r2, save_path=p2)

    # ── Demo 3: Degradation overlay ──
    print('\n[3] Degradation Overlay (HFR growth)')
    datasets = []
    labels = []
    hfrs = []
    for i, r_ohm_scale in enumerate([1.0, 1.15, 1.35, 1.6]):
        p = [0.05 * r_ohm_scale, 0.10 * (1 + 0.1 * i), 0.005, 0.88,
             0.08 * (1 + 0.15 * i), 0.05, 0.75]
        f = np.logspace(np.log10(0.1), np.log10(1e5), 60)
        omega = 2 * np.pi * f
        Z_area = model_R_RC_RC(p, omega)   # Ω·cm²
        Z_ohm = Z_area / geo_area            # Ohm
        Z_ohm += 0.001 * np.abs(Z_ohm).max() * (np.random.randn(len(Z_ohm)) + 1j * np.random.randn(len(Z_ohm)))
        datasets.append((f, Z_ohm.real * geo_area, Z_ohm.imag * geo_area))
        labels.append(f'{i * 100}h')
        hfrs.append(p[0])

    p3 = os.path.join(save_dir, 'eis_degradation.png') if save_dir else None
    plot_nyquist_overlay(datasets, labels, save_path=p3)

    print('    HFR progression:')
    for lbl, h in zip(labels, hfrs):
        print(f'      {lbl:>5s}: {h:.4f} Ω·cm²')

    if save_dir is None:
        plt.show()

    return r, r2


# ═══════════════════════════════════════════════════════════════════════
#  Interactive Mode
# ═══════════════════════════════════════════════════════════════════════

def parse_fcd_header(filepath):
    """
    Parse Scribner .fcd file header to extract skip count and column indices.
    Returns dict with 'skip' and column indices, or None for non-.fcd files.
    """
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
    cond_cols = {}
    for ci, name in enumerate(cols):
        n = name.strip()
        if n == 'I (A)': result['j_col'] = ci
        elif n == 'E_Stack (V)': result['v_col'] = ci
        elif n == 'HFR (mOhm)': result['hfr_col'] = ci
        elif n.startswith('Z_Freq'): result['freq_col'] = ci
        elif n.startswith('Z_Real'): result['zreal_col'] = ci
        elif n.startswith('Z_Imag'): result['zimag_col'] = ci
        elif n == 'Ctrl_Mode': result['mode_col'] = ci
        elif n == 'Cell (C)': cond_cols['T_cell (C)'] = ci
        elif n.startswith('Temp_Anode'): cond_cols['T_anode_dp (C)'] = ci
        elif n.startswith('Flow_Anode'): cond_cols['H2_flow (slpm)'] = ci
        elif n.startswith('Temp_Cathode'): cond_cols['T_cathode_dp (C)'] = ci
        elif n.startswith('Flow_Cathode'): cond_cols['Air_flow (slpm)'] = ci
    if cond_cols: result['condition_cols'] = cond_cols
    return result


def _prompt(label, default=None, cast=float):
    """Prompt for input with a default value."""
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
    # Strip ASCII and Unicode smart quotes, invisible chars
    p = p.strip().strip('"').strip("'")
    p = p.strip('\u2018\u2019\u201c\u201d')  # smart single/double quotes
    p = p.strip('\u202a\u200b')               # LTR mark, zero-width space
    return p


def run_batch(filepaths, labels, model_name, geo_area,
              delimiter, skip, freq_col, zreal_col, zimag_col, zimag_sign,
              save_dir=None, image_ext='png'):
    """
    Batch-process multiple EIS files: fit each, generate summary CSV,
    individual fit plots, and a combined Nyquist overlay.

    Parameters
    ----------
    filepaths : list of str
    labels : list of str
    model_name : str
    geo_area : float
    delimiter, skip, freq_col, zreal_col, zimag_col, zimag_sign :
        File format parameters.
    save_dir : str or None
        Directory for all output files. Required for batch mode.

    Returns
    -------
    list of result dicts
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_results = []
    datasets = []    # for overlay
    summary_rows = []

    print(f'\n  Processing {len(filepaths)} files...\n')

    for i, (fp, lbl) in enumerate(zip(filepaths, labels)):
        print(f'  [{i+1}/{len(filepaths)}] {lbl}')
        try:
            fcd = parse_fcd_header(fp)
            f_skip = fcd['skip'] if fcd else skip
            f_freq = fcd.get('freq_col', freq_col) if fcd else freq_col
            f_zr = fcd.get('zreal_col', zreal_col) if fcd else zreal_col
            f_zi = fcd.get('zimag_col', zimag_col) if fcd else zimag_col
            freq, Zr, Zi = load_eis_data(fp, freq_col=f_freq,
                                          zreal_col=f_zr, zimag_col=f_zi,
                                          delimiter=delimiter, skip_header=f_skip,
                                          zimag_sign=zimag_sign)
            print(f'         {len(freq)} pts, f: {freq.min():.2g}–{freq.max():.2g} Hz')

            r = fit_eis(freq, Zr, Zi, model_name=model_name, geo_area=geo_area)
            r['label'] = lbl
            r['filepath'] = fp
            all_results.append(r)

            datasets.append((freq, Zr * geo_area, Zi * geo_area))

            print(f'         HFR={r["HFR"]:.4f}  R_ct={r.get("R_ct",0):.4f}'
                  f'  R_total={r["R_total"]:.4f} Ω·cm²  R²={r["R_squared"]:.4f}')

            # Individual fit plot
            if save_dir and image_ext:
                safe_name = lbl.replace(' ', '_').replace('/', '-').replace('\\', '-')
                plot_eis(r, save_path=os.path.join(save_dir, f'eis_{safe_name}.{image_ext}'))
                plt.close()

            # Summary row
            row = {'Label': lbl, 'File': os.path.basename(fp),
                   'HFR (Ω·cm²)': r['HFR'],
                   'R_total (Ω·cm²)': r['R_total'],
                   'R²': r['R_squared']}
            if 'R_ct' in r:
                row['R_ct (Ω·cm²)'] = r['R_ct']
            if 'R_mt' in r:
                row['R_mt (Ω·cm²)'] = r['R_mt']
            # Add all fit params
            for name, val, unit in zip(r['param_names'], r['param_values'], r['param_units']):
                row[f'{name}'] = val
            summary_rows.append(row)

        except Exception as e:
            print(f'         ERROR: {e}')
            continue

    if not all_results:
        print('\n  No files processed successfully.')
        return []

    # ── Write summary CSV ──
    if save_dir and summary_rows:
        csv_path = os.path.join(save_dir, 'eis_batch_summary.csv')
        # Collect all column names in order
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

    # ── Combined Nyquist overlay ──
    overlay_labels = [r['label'] for r in all_results]
    if save_dir and image_ext:
        overlay_path = os.path.join(save_dir, f'eis_batch_overlay.{image_ext}')
        plot_nyquist_overlay(datasets, overlay_labels, save_path=overlay_path)
        plt.close()
    elif not save_dir:
        plot_nyquist_overlay(datasets, overlay_labels)
        plt.show()

    # ── Print summary table ──
    area_label = 'Ω·cm²'
    print(f'\n  {"═" * 75}')
    print(f'  {"Label":25s} {"HFR":>10s} {"R_ct":>10s} {"R_mt":>10s} {"R_total":>10s} {"R²":>8s}')
    print(f'  {"─" * 75}')
    for r in all_results:
        r_mt = f'{r["R_mt"]:.4f}' if 'R_mt' in r else '—'
        print(f'  {r["label"]:25s} {r["HFR"]:>10.4f} {r.get("R_ct",0):>10.4f}'
              f' {r_mt:>10s} {r["R_total"]:>10.4f} {r["R_squared"]:>8.4f}')
    print(f'  {"═" * 75}\n')

    return all_results


def run_interactive():
    """Walk the user through EIS analysis step by step."""
    print('\n' + '▓' * 60)
    print('  FUEL CELL EIS ANALYSIS — INTERACTIVE MODE')
    print('▓' * 60)

    # ── Mode ──
    print('\n  Analysis modes:')
    print('    1 = Single EIS file (fit + plot)')
    print('    2 = HFR extraction only (no fit)')
    print('    3 = Multi-file Nyquist overlay')
    print('    4 = Batch — folder (fit all files in a directory)')
    print('    5 = Batch — file list (paste paths one at a time)')
    print('    6 = Run built-in demo')
    mode = int(_prompt('Select mode', default=1, cast=int))

    if mode == 6:
        save = _prompt('Save plots to directory? (path or Enter to show)', default=None, cast=None)
        if save:
            save = _clean_path(save)
        run_demo(save_dir=save if save else None)
        return

    # ── Electrode parameters ──
    print('\n  ── Electrode Parameters ──')
    geo_area = _prompt('Geometric area (cm²)', default=5.0)

    # ── Data file(s) ──
    print('\n  ── Data Files ──')
    print('  Tip: drag-and-drop a file into the terminal to paste its path')

    # ── Test stand presets ──
    print('\n  ── Measurement Test Stand ──')
    print('    0 = Scribner')
    print('    1 = FCTS')
    stand = int(_prompt('Test stand', default=0, cast=int))

    STAND_PRESETS = {
        0: {'name': 'Scribner', 'delimiter': '\t', 'skip': 59,
            'freq_col': 31, 'zreal_col': 32, 'zimag_col': 33, 'zimag_sign': 'negative'},
        1: {'name': 'FCTS',     'delimiter': ',',  'skip': 1,
            'freq_col': 0, 'zreal_col': 1, 'zimag_col': 2, 'zimag_sign': 'negative'},
    }

    if stand in STAND_PRESETS:
        p = STAND_PRESETS[stand]
        delimiter   = p['delimiter']
        skip        = p['skip']
        freq_col    = p['freq_col']
        zreal_col   = p['zreal_col']
        zimag_col   = p['zimag_col']
        zimag_sign  = p['zimag_sign']
        delim_name = 'tab' if delimiter == '\t' else 'comma'
        print(f'  → {p["name"]}: {delim_name}-delimited, {skip} header rows,')
        print(f'    freq=col {freq_col}, Z\'=col {zreal_col}, Z\'\'=col {zimag_col}')
    else:
        print(f'  Unknown test stand {stand}, using manual entry')
        print('\n  ── File Format ──')
        delim_choice = _prompt('Delimiter: 1=comma  2=tab  3=semicolon', default=1, cast=int)
        delimiter = {1: ',', 2: '\t', 3: ';'}.get(delim_choice, ',')
        skip      = int(_prompt('Header rows to skip', default=1, cast=int))
        freq_col  = int(_prompt('Frequency column index (0-based)', default=0, cast=int))
        zreal_col = int(_prompt("Z' column index (0-based)", default=1, cast=int))
        zimag_col = int(_prompt("Z'' column index (0-based)", default=2, cast=int))
        print('\n  ── Z_imag Convention ──')
        print('    1 = Z\'\' stored as negative (capacitive = negative)')
        print('    2 = -Z\'\' stored as positive (capacitive = positive)')
        zi_conv = int(_prompt('Convention', default=1, cast=int))
        zimag_sign = 'negative' if zi_conv == 1 else 'positive'

    if mode == 1:
        # ── Single file fit ──
        filepath = _prompt('EIS data file path', cast=None).strip('"').strip("'")

        print(f'\n  Loading: {filepath}')
        fcd = parse_fcd_header(filepath)
        if fcd:
            skip = fcd['skip']
            freq_col = fcd.get('freq_col', freq_col)
            zreal_col = fcd.get('zreal_col', zreal_col)
            zimag_col = fcd.get('zimag_col', zimag_col)
        freq, Zr, Zi = load_eis_data(filepath, freq_col=freq_col,
                                      zreal_col=zreal_col, zimag_col=zimag_col,
                                      delimiter=delimiter, skip_header=skip,
                                      zimag_sign=zimag_sign)
        print(f'  Read {len(freq)} points  |  f range: {freq.min():.2g} – {freq.max():.2g} Hz')

        hfr = extract_hfr(freq, Zr * geo_area, Zi * geo_area)
        print(f'  HFR (auto): {hfr:.4f} Ω·cm²')

        print('\n  ── Equivalent Circuit Model ──')
        print('    1 = R-RC          (single arc)')
        print('    2 = R-RC-RC       (two arcs: kinetic + mass transport)')
        print('    3 = Randles-W     (Randles + Warburg diffusion)')
        model_choice = int(_prompt('Model', default=2, cast=int))
        model_name = {1: 'R-RC', 2: 'R-RC-RC', 3: 'Randles-W'}.get(model_choice, 'R-RC-RC')

        results = fit_eis(freq, Zr, Zi, model_name=model_name, geo_area=geo_area)
        print_results(results)

        save = _prompt('\n  Save plot to directory? (path or Enter to show)', default=None, cast=None)
        if save:
            save = _clean_path(save)
        if save:
            os.makedirs(save, exist_ok=True)
            plot_eis(results, save_path=os.path.join(save, 'eis_fit.png'))
        else:
            plot_eis(results)
            plt.show()

    elif mode == 2:
        # ── HFR extraction only ──
        filepath = _prompt('EIS data file path', cast=None).strip('"').strip("'")
        fcd = parse_fcd_header(filepath)
        f_skip = fcd['skip'] if fcd else skip
        f_freq = fcd.get('freq_col', freq_col) if fcd else freq_col
        f_zr = fcd.get('zreal_col', zreal_col) if fcd else zreal_col
        f_zi = fcd.get('zimag_col', zimag_col) if fcd else zimag_col
        freq, Zr, Zi = load_eis_data(filepath, freq_col=f_freq,
                                      zreal_col=f_zr, zimag_col=f_zi,
                                      delimiter=delimiter, skip_header=f_skip,
                                      zimag_sign=zimag_sign)

        hfr_interp = extract_hfr(freq, Zr * geo_area, Zi * geo_area, method='x_intercept')
        hfr_min    = extract_hfr(freq, Zr * geo_area, Zi * geo_area, method='min_zimag')
        hfr_hf     = extract_hfr(freq, Zr * geo_area, Zi * geo_area, method='highest_freq')

        print(f'\n  ── HFR Results ──')
        print(f'    x-intercept:    {hfr_interp:.6f} Ω·cm²')
        print(f'    min |Z\'\'|:      {hfr_min:.6f} Ω·cm²')
        print(f'    highest freq:   {hfr_hf:.6f} Ω·cm²')

    elif mode == 3:
        # ── Multi-file overlay ──
        n_files = int(_prompt('Number of files to overlay', default=2, cast=int))
        datasets = []
        labels = []
        for i in range(n_files):
            fp = _prompt(f'  File {i+1} path', cast=None).strip('"').strip("'")
            lbl = _prompt(f'  File {i+1} label', default=f'File {i+1}', cast=None)
            fcd = parse_fcd_header(fp)
            f_skip = fcd['skip'] if fcd else skip
            f_freq = fcd.get('freq_col', freq_col) if fcd else freq_col
            f_zr = fcd.get('zreal_col', zreal_col) if fcd else zreal_col
            f_zi = fcd.get('zimag_col', zimag_col) if fcd else zimag_col
            freq, Zr, Zi = load_eis_data(fp, freq_col=f_freq,
                                          zreal_col=f_zr, zimag_col=f_zi,
                                          delimiter=delimiter, skip_header=f_skip,
                                          zimag_sign=zimag_sign)
            datasets.append((freq, Zr * geo_area, Zi * geo_area))
            labels.append(lbl)
            hfr = extract_hfr(freq, Zr * geo_area, Zi * geo_area)
            print(f'    → {len(freq)} pts, HFR = {hfr:.4f} Ω·cm²')

        save = _prompt('\n  Save plot to directory? (path or Enter to show)', default=None, cast=None)
        if save:
            save = _clean_path(save)
        if save:
            os.makedirs(save, exist_ok=True)
            plot_nyquist_overlay(datasets, labels,
                                save_path=os.path.join(save, 'eis_overlay.png'))
        else:
            plot_nyquist_overlay(datasets, labels)
            plt.show()

    elif mode in (4, 5):
        # ── Batch processing ──
        print('\n  ── Equivalent Circuit Model ──')
        print('    1 = R-RC          (single arc)')
        print('    2 = R-RC-RC       (two arcs: kinetic + mass transport)')
        print('    3 = Randles-W     (Randles + Warburg diffusion)')
        model_choice = int(_prompt('Model', default=2, cast=int))
        model_name = {1: 'R-RC', 2: 'R-RC-RC', 3: 'Randles-W'}.get(model_choice, 'R-RC-RC')

        save_dir = _clean_path(_prompt('Output directory for results', default='eis_batch_output', cast=None))

        filepaths = []
        labels = []

        if mode == 4:
            # ── Folder batch ──
            folder = _prompt('Folder path containing EIS files', cast=None)
            folder = _clean_path(folder)

            # Collect all data files
            extensions = ['*.csv', '*.txt', '*.tsv', '*.fcd', '*.CSV', '*.TXT', '*.TSV', '*.FCD']
            for ext in extensions:
                filepaths.extend(glob.glob(os.path.join(folder, ext)))
            filepaths = sorted(set(filepaths))  # deduplicate and sort

            # Only include files with "EIS" in the filename
            filepaths = [fp for fp in filepaths if 'EIS' in os.path.basename(fp).upper()]

            if not filepaths:
                print(f'  No files containing "EIS" found in: {folder}')
                return

            print(f'  Found {len(filepaths)} files:')
            for fp in filepaths:
                name = os.path.splitext(os.path.basename(fp))[0]
                labels.append(name)
                print(f'    {name}')

        elif mode == 5:
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
                lbl = _prompt(f'  Label', default=default_label, cast=None)
                filepaths.append(fp)
                labels.append(lbl)

            if not filepaths:
                print('  No files entered.')
                return

        run_batch(filepaths, labels, model_name, geo_area,
                  delimiter, skip, freq_col, zreal_col, zimag_col, zimag_sign,
                  save_dir=save_dir)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Fuel Cell EIS Analysis')
    parser.add_argument('--file', type=str, help='EIS data file')
    parser.add_argument('--area', type=float, default=5.0,
                        help='Geometric electrode area in cm² (default: 5.0)')
    parser.add_argument('--model', type=str, default='R-RC-RC',
                        choices=['R-RC', 'R-RC-RC', 'Randles-W'],
                        help='Equivalent circuit model (default: R-RC-RC)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots')
    parser.add_argument('--demo', action='store_true',
                        help='Run built-in demo')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run interactive guided mode')
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    elif args.demo:
        run_demo(save_dir=args.save_dir)
    elif args.file:
        freq, Zr, Zi = load_eis_data(args.file)
        results = fit_eis(freq, Zr, Zi, model_name=args.model, geo_area=args.area)
        print_results(results)
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            plot_eis(results, save_path=os.path.join(args.save_dir, 'eis_fit.png'))
        else:
            plot_eis(results)
            plt.show()
    else:
        run_interactive()


if __name__ == '__main__':
    main()
