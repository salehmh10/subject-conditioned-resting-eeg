"""Transparent broadband QC; no physiological outcome computations."""
from __future__ import annotations

import math

import mne
import numpy as np
import pandas as pd
from scipy import signal


def mad_scale(x, axis=None):
    x = np.asarray(x)
    center = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - center), axis=axis)


def filter_kwargs(config):
    return {key: value for key, value in config['filter'].items()
            if key not in {'implementation', 'lengths_by_native_rate', 'pad'}}


def resample_taps(rate, config):
    target = config['resampling']['target_sfreq']
    divisor = math.gcd(int(rate), int(target))
    up, down = target // divisor, int(rate) // divisor
    max_rate = max(up, down)
    half = config['resampling']['half_length_multiplier'] * max_rate
    taps = signal.firwin(2 * half + 1, 1.0 / max_rate,
                         window=('kaiser', config['resampling']['kaiser_beta']))
    return up, down, taps


def expected_samples(n, rate, target=128):
    return (int(n) * int(target) + int(rate) - 1) // int(rate)


def filter_resample(values_v, rate, config):
    if rate not in (500, 1000, 2500):
        raise ValueError('Unvalidated native sampling rate')
    filtered = mne.filter.filter_data(values_v, sfreq=rate, copy=False,
                                     pad=config['filter']['pad'], n_jobs=1,
                                     verbose='ERROR', **filter_kwargs(config))
    up, down, taps = resample_taps(rate, config)
    result = signal.resample_poly(filtered, up, down, axis=-1, window=taps,
                                  padtype=config['resampling']['padtype'])
    if result.shape[-1] != expected_samples(values_v.shape[-1], rate):
        raise ValueError('Unexpected resampled length')
    return result


def sanity_metrics(values_v, names):
    rows = []
    for name, channel in zip(names, values_v):
        finite = np.isfinite(channel)
        x = channel[finite] * 1e6
        if len(x):
            median = float(np.median(x))
            scale = float(mad_scale(x))
            centered = x - median
            winsor = np.clip(centered, *np.quantile(centered, [0.01, 0.99]))
            diff = np.diff(x)
            rows.append(dict(channel=name, unit='uV', finite_fraction=float(finite.mean()),
                nan_count=int(np.isnan(channel).sum()), inf_count=int(np.isinf(channel).sum()),
                flat_fraction=float(np.mean(diff == 0)), exact_zero_fraction=float(np.mean(x == 0)),
                minimum=float(x.min()), maximum=float(x.max()), median=median,
                robust_rms=float(np.sqrt(np.mean(winsor ** 2))), robust_scale=scale,
                peak_to_peak=float(np.ptp(x)), difference_scale=float(mad_scale(diff)),
                difference_abs_p99=float(np.quantile(np.abs(diff), .99)),
                difference_abs_max=float(np.max(np.abs(diff))),
                extrema_repeat_fraction=float(np.mean((x == x.min()) | (x == x.max())))))
        else:
            rows.append(dict(channel=name, unit='uV', finite_fraction=0.0,
                             nan_count=int(np.isnan(channel).sum()), inf_count=int(np.isinf(channel).sum()),
                             flat_fraction=1.0, robust_scale=0.0))
    return rows


def channel_metrics(data_uv, names):
    # Trim filter support before channel statistics, preserving a state-pooled fit.
    corr = np.corrcoef(data_uv)
    np.fill_diagonal(corr, np.nan)
    return pd.DataFrame(dict(channel=names, scale_uv=mad_scale(data_uv, axis=1),
                             derivative_scale_uv=mad_scale(np.diff(data_uv, axis=1), axis=1),
                             median_abs_correlation=np.nanmedian(np.abs(corr), axis=1)))


def robust_z(x, floor):
    return (x - np.median(x)) / max(float(mad_scale(x)), floor)


def fit_bad_channels(metrics, sanity, names, target_names, config):
    require_train_fit(metrics, sanity)
    cfg = config['bad_channels']
    pooled = metrics.groupby('channel', sort=False)[['scale_uv', 'derivative_scale_uv', 'median_abs_correlation']].median().reindex(names)
    amp_z = robust_z(np.log10(np.maximum(pooled.scale_uv.to_numpy(), 1e-12)), cfg['log_scale_floor'])
    diff_z = robust_z(np.log10(np.maximum(pooled.derivative_scale_uv.to_numpy(), 1e-12)), cfg['log_scale_floor'])
    rows = []
    for i, name in enumerate(names):
        native = sanity.loc[sanity.channel == name]
        nonfinite = bool((native.finite_fraction < 1).any())
        flat = float((native.robust_scale < cfg['catastrophic_flat_scale_uv']).mean())
        votes = [abs(amp_z[i]) > cfg['log_outlier_z'],
                 abs(diff_z[i]) > cfg['log_outlier_z'],
                 pooled.loc[name, 'median_abs_correlation'] < cfg['low_median_absolute_correlation']]
        catastrophic = nonfinite or flat >= cfg['catastrophic_flat_block_fraction']
        bad = catastrophic or (sum(votes) >= cfg['minimum_ordinary_votes'] and votes[2])
        rows.append(dict(channel=name, log_scale_z=float(amp_z[i]), log_derivative_z=float(diff_z[i]),
                         median_abs_correlation=float(pooled.loc[name, 'median_abs_correlation']),
                         catastrophic=catastrophic, flat_block_fraction=flat,
                         ordinary_votes=int(sum(votes)), bad=bool(bad),
                         reason='catastrophic' if catastrophic else 'two_metric_outlier' if bad else 'within_rules',
                         fit_pool='pooled_EO_EC_Train'))
    bads = [row['channel'] for row in rows if row['bad']]
    eligible = len(bads) <= cfg['max_bad_eeg'] and len(set(bads) & set(target_names)) <= cfg['max_bad_target']
    return bads, eligible, pd.DataFrame(rows)


def reference_and_select(data_v, names, types, bads, config):
    eeg_indices = [i for i, kind in enumerate(types) if kind == 'eeg']
    if len(eeg_indices) != 61 or len(set(names)) != len(names):
        raise ValueError('CAR requires the complete 61-channel EEG montage')
    raw = mne.io.RawArray(data_v.copy(), mne.create_info(names, 128, types), verbose='ERROR')
    raw.set_montage(config['montage'], on_missing='raise', verbose='ERROR')
    raw.info['bads'] = list(bads)
    if bads:
        raw.interpolate_bads(reset_bads=True, mode='accurate', origin=(0., 0., .04), verbose='ERROR')
    # Explicit full-montage CAR before final ordering; VEOG excluded from reference.
    values = raw.get_data()
    values[eeg_indices] -= values[eeg_indices].mean(axis=0, keepdims=True)
    target = values[[names.index(name) for name in config['target_channels']]] * 1e6
    veog = values[names.index('VEOG')] * 1e6
    return target, veog


def window_starts(n_samples, config):
    w, stride = config['window']['length_samples'], config['window']['stride_samples']
    return np.arange(0, max(0, n_samples - w + 1), stride, dtype=np.int64)


def window_features(eeg_uv, veog_uv, config):
    rows = []
    for start in window_starts(eeg_uv.shape[1], config):
        stop = int(start) + config['window']['length_samples']
        x, eog = eeg_uv[:, start:stop], veog_uv[start:stop]
        finite = bool(np.isfinite(x).all() and np.isfinite(eog).all())
        if finite:
            scales = mad_scale(x, axis=1)
            dx = np.diff(x, axis=1)
            features = [float(np.max(np.ptp(x, axis=1))), float(scales.max()),
                        float(np.max(np.abs(dx))), float(np.ptp(eog))]
            flat = bool(scales.min() < config['qc']['minimum_channel_scale_uv'] or
                        mad_scale(eog) < config['qc']['minimum_veog_scale_uv'] or
                        np.max(np.mean(dx == 0, axis=1)) > config['qc']['maximum_repeated_difference_fraction'])
        else:
            features, flat = [0.0] * 4, False
        rows.append(dict(window_start_sample=int(start), finite=finite, flat=flat,
                         **dict(zip(config['qc']['features'], features))))
    return pd.DataFrame(rows)


def fit_window_thresholds(features, config, multiplier=None):
    require_train_fit(features)
    # Deliberately no state argument or state grouping: one subject-wide reference.
    cfg = config['qc']
    fit = features.loc[features.finite & ~features.flat, cfg['features']].to_numpy()
    if not len(fit):
        raise ValueError('No finite non-flat Train windows for robust QC reference')
    k = cfg['multiplier'] if multiplier is None else multiplier
    medians, scales = np.median(fit, axis=0), mad_scale(fit, axis=0)
    thresholds = np.minimum(cfg['threshold_caps'], np.maximum(cfg['threshold_floors'], medians + k * scales))
    return dict(zip(cfg['features'], map(float, thresholds)))


def apply_window_qc(features, thresholds, subject_eligible=True):
    reasons = []
    for row in features.to_dict('records'):
        reason = []
        if row.get('qc_reference_failed', False):
            reason.append('no_valid_Train_QC_reference')
        if not subject_eligible:
            reason.append('subject_bad_channel_limit')
        if row.get('native_integrity_failure', False):
            reason.append('native_signal_integrity')
        elif not row['finite']:
            reason.append('nonfinite')
        if row['flat']:
            reason.append('flat')
        for name, threshold in thresholds.items():
            if row[name] > threshold:
                reason.append(name)
        reasons.append('|'.join(reason) or 'usable')
    return np.array([reason == 'usable' for reason in reasons], dtype=bool), reasons


def synthetic_validation(config):
    rows = []
    for rate in (500, 1000, 2500):
        n = 60 * rate
        t = np.arange(n) / rate
        # Frequencies are synthetic engineering tones, outside physiological alpha.
        source = np.array([np.sin(2*np.pi*f*t) for f in (5, 20, 35, 60, 100)])
        result = filter_resample(source.copy(), rate, config)
        repeated = filter_resample(source.copy(), rate, config)
        guard = config['edge']['guard_samples_at_128']
        trimmed = result[:, guard:-guard]
        gains = np.sqrt(2) * np.sqrt(np.mean(trimmed ** 2, axis=1))
        taps = mne.filter.create_filter(None, rate, verbose='ERROR', **filter_kwargs(config))
        up, _, resampling = resample_taps(rate, config)
        support = (len(taps)-1)/(2*rate) + (len(resampling)-1)/(2*rate*up)
        rows.append(dict(native_sfreq=rate, input_samples=n, output_samples=result.shape[1],
                         expected_samples=expected_samples(n, rate), fir_length=len(taps),
                         combined_half_support_sec=support,
                         gain_5hz=float(gains[0]), gain_20hz=float(gains[1]), gain_35hz=float(gains[2]),
                         residual_60hz=float(gains[3]), residual_100hz=float(gains[4]),
                         deterministic=bool(np.array_equal(result, repeated)),
                         pass_validation=bool(np.all(np.abs(gains[:3]-1)<.02) and np.max(gains[3:])<.01
                                              and config['edge']['guard_sec'] >= support)))
    return pd.DataFrame(rows)


def require_train_fit(*frames):
    """Reject split-tagged held-out features before any reference calculation.

    Legacy synthetic utility fixtures have no split column. Production frames
    are explicitly tagged by the gated Train pipeline.
    """
    for frame in frames:
        for column in ('split', 'provisional_split'):
            if column in frame and not frame[column].eq('Train').all():
                raise ValueError('Only Train data may fit subject parameters')
