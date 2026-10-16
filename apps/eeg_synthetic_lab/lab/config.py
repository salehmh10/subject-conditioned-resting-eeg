"""Presentation labels and stable visual encodings; no scientific observations."""
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = APP_ROOT / "data" / "presentation_bundle"
COLORS = {"Real": "#233C58", "B0": "#C0782F", "B1": "#087F8C", "B2": "#625AB4"}
DASHES = {"Real": "solid", "B0": "dot", "B1": "dash", "B2": "dashdot"}
CLASSIFIERS = {
    "LOGBANDPOWER_LOGREG": "LogBandpower + Logistic Regression",
    "RIEMANNIAN_SVM": "OAS/Riemannian + Linear SVM",
    "OAS_RIEMANNIAN_SVM": "OAS/Riemannian + Linear SVM",
    "EEGNET": "EEGNetv4", "EEGNETV4": "EEGNetv4", "EEGNET_V4": "EEGNetv4",
}
CLASSIFIER_COLORS = ["#087F8C", "#486581", "#625AB4"]
CONDITIONS = [
    "REAL_ONLY", "REAL_DUPLICATE_1X_CONTROL", "TSTR_B0_1X", "TSTR_B1_1X", "TSTR_B2_1X",
    "REAL_PLUS_B0_0P5X", "REAL_PLUS_B1_0P5X", "REAL_PLUS_B2_0P5X",
    "REAL_PLUS_B0_1X", "REAL_PLUS_B1_1X", "REAL_PLUS_B2_1X",
]
CONDITION_LABELS = {
    "REAL_ONLY": "Real only", "REAL_DUPLICATE_1X_CONTROL": "Real duplicate · 1×",
    "TSTR_B0_1X": "Synthetic B0 only", "TSTR_B1_1X": "Synthetic B1 only", "TSTR_B2_1X": "Synthetic B2 only",
    "REAL_PLUS_B0_0P5X": "Real + B0 · 0.5×", "REAL_PLUS_B1_0P5X": "Real + B1 · 0.5×", "REAL_PLUS_B2_0P5X": "Real + B2 · 0.5×",
    "REAL_PLUS_B0_1X": "Real + B0 · 1×", "REAL_PLUS_B1_1X": "Real + B1 · 1×", "REAL_PLUS_B2_1X": "Real + B2 · 1×",
}
PAGES = ["Overview & models", "Signal explorer", "Fidelity explorer", "Connectivity explorer", "Identity & diversity", "Classification utility", "Learning & methods"]
STEPS = ["Question", "Data", "Models", "Waveforms", "Spectrum & alpha", "Connectivity", "Identity", "Utility", "What the results establish", "Limitations & conclusion"]
PLOT_CONFIG = {"displaylogo": False, "responsive": True, "scrollZoom": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
