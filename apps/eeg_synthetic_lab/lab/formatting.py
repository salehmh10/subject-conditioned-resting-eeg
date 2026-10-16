"""Scientific presentation formatting with explicit fraction / percentage units."""
import math
from .config import CLASSIFIERS, CONDITION_LABELS


def percent(value, digits=2):
    return "Unavailable" if value is None or not math.isfinite(float(value)) else f"{100 * float(value):.{digits}f}%"


def percentage_points(value, digits=3, signed=True):
    if value is None or not math.isfinite(float(value)):
        return "Unavailable"
    return f"{100 * float(value):+.{digits}f} pp" if signed else f"{100 * float(value):.{digits}f} pp"


def reverse_contrast(estimate, ci_low, ci_high):
    """Reverse a directed difference and its saved confidence interval together."""
    return -estimate, -ci_high, -ci_low


def classifier_label(key):
    label = CLASSIFIERS.get(str(key))
    if label:
        return label
    upper = str(key).upper()
    if "EEGNET" in upper:
        return "EEGNetv4"
    if "RIEMANN" in upper or "OAS" in upper:
        return "OAS/Riemannian + Linear SVM"
    return str(key).replace("_", " ")


def condition_label(key):
    return CONDITION_LABELS.get(str(key), str(key).replace("_", " "))


def condition_model(key):
    return next((m for m in ("B0", "B1", "B2") if f"_{m}_" in str(key)), "Real")


def readable_metric(key):
    names={
        "acf_mae_lag1_128":"Autocorrelation MAE · lags 1–128",
        "log_psd_mae_db":"Log-PSD mean absolute error",
        "bandpower_median_mae_db":"Median bandpower mean absolute error",
        "relative_bandpower_mae":"Relative bandpower mean absolute error",
        "sd_md_pooled_uv":"SD-MD · pooled channel standard-deviation difference",
        "mean_amplitude_mae_uv":"Mean amplitude absolute error",
        "correlation_upper_mae":"Channel-correlation edge MAE",
        "correlation_upper_r":"Channel-correlation edge-pattern correlation",
        "covariance_relative_frobenius_z":"Covariance relative Frobenius error",
        "exact_same_stratum_train_copy_fraction":"Exact copies of Real Train in the same stratum",
        "within_stratum_exact_duplicate_fraction":"Exact duplicates within a synthetic stratum",
        "nearest_real_rmse_z_median":"Nearest-Real distance · median RMSE",
        "nearest_real_rmse_z_q05":"Nearest-Real distance · 5th-percentile RMSE",
        "rms_pairwise_diversity_ratio":"Pairwise RMS diversity ratio",
        "sliced_wasserstein1_z":"Sliced Wasserstein-1 distance",
    }
    if key in names: return names[key]
    for band in ("delta","theta","alpha","beta"):
        for ending,label in (("mae_db","bandpower MAE"),("topography_r","bandpower topography correlation"),("w1_db","bandpower Wasserstein-1 distance")):
            if key==f"bandpower_{band}_{ending}": return f"{band.capitalize()} {label}"
    descriptors={"mean_uv":"Mean amplitude","rms_uv":"RMS amplitude","std_uv":"Standard deviation","skewness":"Skewness","excess_kurtosis":"Excess kurtosis","hjorth_activity_uv2":"Hjorth activity","hjorth_mobility_per_sample":"Hjorth mobility","hjorth_complexity":"Hjorth complexity","permutation_entropy_m3_tau1":"Permutation entropy · m=3, τ=1"}
    for prefix,label in descriptors.items():
        if key==prefix+"_w1": return label+" · Wasserstein-1 distance"
        if key==prefix+"_w1_real_sd_units": return label+" · Wasserstein-1 in Real SD units"
    return str(key).replace("_", " ").replace("dwpli2", "dwPLI²").replace("psd", "PSD").replace("mae", "MAE").replace("swd", "SWD").replace("acf", "ACF")
