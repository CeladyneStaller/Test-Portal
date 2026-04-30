"""
Script Registry + Parameter Definitions
========================================
Maps display names to run() functions and defines the
user-configurable parameters each script accepts.

The "sample_name" field is prepended to all output filenames by the
job runner (main.py), so individual scripts don't need to handle it.
"""

from scripts.ecsa_analysis import run as ecsa_run
from scripts.eis_analysis import run as eis_run
from scripts.h2_crossover_analysis import run as crossover_run
from scripts.polcurve_analysis import run as polcurve_run
from scripts.electrolyzer_polcurve import run as elx_polcurve_run
from scripts.electrolyzer_durability import run as elx_durability_run
from scripts.fuelcell_analysis import run as fuelcell_run
from scripts.ocv_analysis import run as ocv_run
from scripts.compare_polcurves import run as compare_polcurves_run

SCRIPT_REGISTRY = {
    "Fuel Cell ECSA": ecsa_run,
    "EIS Analysis": eis_run,
    "H2 Crossover": crossover_run,
    "FC Polarization Curve": polcurve_run,
    "OCV Analysis": ocv_run,
    "Electrolyzer Pol Curve": elx_polcurve_run,
    "Electrolyzer Durability": elx_durability_run,
    "Fuel Cell Full Analysis": fuelcell_run,
    "Compare Polcurves": compare_polcurves_run,
}

# ─── Short labels for filename prefixing ─────────────────────────
SCRIPT_SHORT = {
    "Fuel Cell ECSA": "ECSA",
    "EIS Analysis": "EIS",
    "H2 Crossover": "H2Xover",
    "FC Polarization Curve": "PolCurve",
    "OCV Analysis": "OCV",
    "Electrolyzer Pol Curve": "ElxPolCurve",
    "Electrolyzer Durability": "ElxDurability",
    "Fuel Cell Full Analysis": "FCAnalysis",
    "Compare Polcurves": "Compare",
}

# ─── Common sample_name field (inserted first for every script) ──
_SAMPLE_FIELD = {"key": "sample_name", "label": "Sample Name", "type": "text", "default": ""}

_IMAGE_FORMAT_FIELD = {"key": "image_format", "label": "Image Format", "type": "select",
    "default": "png",
    "options": [{"value": "png", "label": "PNG"},
                {"value": "svg", "label": "SVG"},
                {"value": "pdf", "label": "PDF"},
                {"value": "tiff", "label": "TIFF"},
                {"value": "none", "label": "No Images"}]}

SCRIPT_PARAMS = {
    "Fuel Cell ECSA": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "stand", "label": "Test Stand", "type": "select", "default": "0",
         "options": [{"value": "0", "label": "Scribner"},
                     {"value": "1", "label": "FCTS"}]},
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "scan_rate", "label": "Scan Rate (V/s)", "type": "number",
         "default": 0.050, "step": 0.001, "min": 0.001},
        {"key": "loading", "label": "Cathode Pt Loading (mg/cm²)", "type": "number",
         "default": 0.20, "step": 0.01, "min": 0},
        {"key": "v_low", "label": "H_UPD Lower Bound (V vs RHE)", "type": "number",
         "default": 0.08, "step": 0.01, "min": 0},
        {"key": "v_high", "label": "H_UPD Upper Bound (V vs RHE)", "type": "number",
         "default": 0.40, "step": 0.01, "min": 0},
        {"key": "cycle", "label": "Cycle to Analyze", "type": "select", "default": "last",
         "options": [{"value": "last", "label": "Last (most stable)"},
                     {"value": "first", "label": "First"},
                     {"value": "average", "label": "Average all"}]},
    ],
    "EIS Analysis": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "model_name", "label": "Equivalent Circuit Model", "type": "select",
         "default": "R-RC",
         "options": [{"value": "R-RC", "label": "R-RC (simple)"},
                     {"value": "R-RC-RC", "label": "R-RC-RC (two arcs)"},
                     {"value": "Randles-W", "label": "Randles + Warburg"}]},
    ],
    "H2 Crossover": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "avg_V_min", "label": "Averaging Window Low (V)", "type": "number",
         "default": 0.35, "step": 0.01, "min": 0},
        {"key": "avg_V_max", "label": "Averaging Window High (V)", "type": "number",
         "default": 0.50, "step": 0.01, "min": 0},
        {"key": "membrane_thickness", "label": "Membrane Thickness (µm, 0 = skip)",
         "type": "number", "default": 0, "step": 1, "min": 0},
    ],
    "FC Polarization Curve": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "tafel_j_min", "label": "Tafel Region j_min (A/cm²)", "type": "number",
         "default": 0.01, "step": 0.001, "min": 0},
        {"key": "tafel_j_max", "label": "Tafel Region j_max (A/cm²)", "type": "number",
         "default": 0.10, "step": 0.001, "min": 0},
    ],
    "OCV Analysis": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "interval_s", "label": "Resampling Interval (seconds)", "type": "number",
         "default": 60.0, "step": 1, "min": 1},
    ],
    "Electrolyzer Pol Curve": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "cell_id", "label": "Cell ID (for folder scan)", "type": "text",
         "default": "a1"},
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "T_C", "label": "Temperature (°C)", "type": "number",
         "default": 80.0, "step": 1, "min": 0},
        {"key": "p_cathode_barg", "label": "Cathode Pressure (barg)", "type": "number",
         "default": 0.0, "step": 0.1, "min": 0},
        {"key": "p_anode_barg", "label": "Anode Pressure (barg)", "type": "number",
         "default": 0.0, "step": 0.1, "min": 0},
        {"key": "eis_ref_voltage", "label": "EIS Reference Voltage (V, blank=skip)",
         "type": "number", "default": "", "step": 0.01, "min": 0},
    ],
    "Electrolyzer Durability": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 25.0, "step": 0.1, "min": 0.1},
        {"key": "eis_ref_voltage", "label": "EIS Reference Voltage (V)",
         "type": "number", "default": 1.25, "step": 0.01, "min": 0},
        {"key": "data_interval_min", "label": "Durability Data Interval (minutes, blank=all)",
         "type": "number", "default": "", "step": 1, "min": 1},
    ],
    "Fuel Cell Full Analysis": [
        _SAMPLE_FIELD,
        _IMAGE_FORMAT_FIELD,
        {"key": "stand", "label": "Test Stand", "type": "select", "default": "0",
         "options": [{"value": "0", "label": "Scribner"},
                     {"value": "1", "label": "FCTS"}]},
        {"key": "geo_area", "label": "Geometric Area (cm²)", "type": "number",
         "default": 5.0, "step": 0.1, "min": 0.1},
        {"key": "loading", "label": "Cathode Pt Loading (mg/cm²)", "type": "number",
         "default": 0.20, "step": 0.01, "min": 0},
        {"key": "interval_s", "label": "OCV Resampling Interval (seconds)", "type": "number",
         "default": 60.0, "step": 1, "min": 1},
    ],
}
