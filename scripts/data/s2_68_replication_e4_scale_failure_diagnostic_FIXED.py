from __future__ import annotations

import numpy as np

import s2_68_replication_e4_scale_failure_diagnostic as base


def _subject_sufficient_stats(
    z: np.ndarray,
    state_index: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Per-subject sufficient statistics for pooled ddof=0 variance.

    This is mathematically equivalent to flattening all selected subjects
    and calling np.std(..., ddof=0), but avoids repeatedly scanning millions
    of values inside every bootstrap replicate.
    """
    x = z[:, state_index, ...].astype(
        np.float64,
        copy=False,
    )

    if x.shape[0] != base.N_SUBJECTS:
        raise RuntimeError(
            f"Unexpected subject count in sufficient stats: {x.shape}"
        )

    per_subject_count = int(
        np.prod(
            x.shape[1:],
            dtype=np.int64,
        )
    )

    subject_sum = np.sum(
        x,
        axis=tuple(
            range(
                1,
                x.ndim,
            )
        ),
        dtype=np.float64,
    )

    subject_sumsq = np.sum(
        x * x,
        axis=tuple(
            range(
                1,
                x.ndim,
            )
        ),
        dtype=np.float64,
    )

    return (
        subject_sum,
        subject_sumsq,
        per_subject_count,
    )


def _bootstrap_pooled_std_from_stats(
    *,
    subject_sum: np.ndarray,
    subject_sumsq: np.ndarray,
    per_subject_count: int,
    subject_draws: np.ndarray,
) -> np.ndarray:
    """
    Compute pooled ddof=0 std for every bootstrap draw.

    subject_draws shape:
        [bootstrap_rep, sampled_subject_position]
    """
    total_sum = np.sum(
        subject_sum[
            subject_draws
        ],
        axis=1,
        dtype=np.float64,
    )

    total_sumsq = np.sum(
        subject_sumsq[
            subject_draws
        ],
        axis=1,
        dtype=np.float64,
    )

    total_count = float(
        subject_draws.shape[1]
        * per_subject_count
    )

    mean = (
        total_sum
        / total_count
    )

    variance = (
        total_sumsq
        / total_count
        - mean * mean
    )

    # Numerical round-off protection only.
    variance = np.maximum(
        variance,
        0.0,
    )

    return np.sqrt(
        variance
    )


def fast_bootstrap_scale(
    real_z: np.ndarray,
    b1_z: np.ndarray,
    candidate_z: np.ndarray,
    label: str,
):
    """
    Exact same scientific bootstrap as the original s2_68:
      - seed = base.BOOTSTRAP_SEED
      - reps = base.BOOTSTRAP_REPS
      - resampling unit = subject
      - same sampled subject indices applied to Real/B1/Candidate
      - ddof=0 pooled standard deviation
      - parent-relative scale = candidate_scale / B1_scale

    Only the computation is vectorized.
    """
    rng = np.random.default_rng(
        base.BOOTSTRAP_SEED
    )

    rows = []

    for state_index, state_name in enumerate(
        (
            "EO",
            "EC",
        )
    ):
        (
            real_sum,
            real_sumsq,
            real_count,
        ) = _subject_sufficient_stats(
            real_z,
            state_index,
        )

        (
            b1_sum,
            b1_sumsq,
            b1_count,
        ) = _subject_sufficient_stats(
            b1_z,
            state_index,
        )

        (
            cand_sum,
            cand_sumsq,
            cand_count,
        ) = _subject_sufficient_stats(
            candidate_z,
            state_index,
        )

        subject_draws = rng.integers(
            0,
            base.N_SUBJECTS,
            size=(
                base.BOOTSTRAP_REPS,
                base.N_SUBJECTS,
            ),
            endpoint=False,
        )

        real_std = _bootstrap_pooled_std_from_stats(
            subject_sum=real_sum,
            subject_sumsq=real_sumsq,
            per_subject_count=real_count,
            subject_draws=subject_draws,
        )

        b1_std = _bootstrap_pooled_std_from_stats(
            subject_sum=b1_sum,
            subject_sumsq=b1_sumsq,
            per_subject_count=b1_count,
            subject_draws=subject_draws,
        )

        cand_std = _bootstrap_pooled_std_from_stats(
            subject_sum=cand_sum,
            subject_sumsq=cand_sumsq,
            per_subject_count=cand_count,
            subject_draws=subject_draws,
        )

        if (
            np.any(
                real_std <= 0.0
            )
            or np.any(
                b1_std <= 0.0
            )
        ):
            raise RuntimeError(
                f"{label}/{state_name}: non-positive bootstrap std."
            )

        b1_scale = (
            b1_std
            / real_std
        )

        cand_scale = (
            cand_std
            / real_std
        )

        draws = (
            cand_scale
            / b1_scale
        )

        # Point estimate remains EXACTLY the original helper formula.
        point_real_std = base.pooled_state_std(
            real_z,
            state_index,
        )

        point_b1_scale = (
            base.pooled_state_std(
                b1_z,
                state_index,
            )
            / point_real_std
        )

        point_candidate_scale = (
            base.pooled_state_std(
                candidate_z,
                state_index,
            )
            / point_real_std
        )

        point = (
            point_candidate_scale
            / point_b1_scale
        )

        ci_low, ci_high = np.percentile(
            draws,
            [
                2.5,
                97.5,
            ],
        )

        rows.append(
            {
                "candidate": label,
                "state": state_name,
                "point_parent_relative_scale": float(
                    point
                ),
                "guard_lower": (
                    base.LOWER_SCALE_GUARD
                ),
                "guard_upper": (
                    base.UPPER_SCALE_GUARD
                ),
                "point_guard_slack": float(
                    base.scale_slack(
                        point
                    )
                ),
                "bootstrap_reps": (
                    base.BOOTSTRAP_REPS
                ),
                "bootstrap_mean": float(
                    np.mean(
                        draws
                    )
                ),
                "bootstrap_std": float(
                    np.std(
                        draws,
                        ddof=0,
                    )
                ),
                "bootstrap_ci95_low": float(
                    ci_low
                ),
                "bootstrap_ci95_high": float(
                    ci_high
                ),
                "bootstrap_fraction_inside_guard": float(
                    np.mean(
                        (
                            draws
                            >= base.LOWER_SCALE_GUARD
                        )
                        & (
                            draws
                            <= base.UPPER_SCALE_GUARD
                        )
                    )
                ),
            }
        )

        print(
            f"FAST BOOTSTRAP {label} {state_name}: "
            f"{base.BOOTSTRAP_REPS} reps complete"
        )

    return rows


# Pure technical optimization:
# preserve the original main(), paths, protocol, metrics, seed, and rules.
base.bootstrap_scale = fast_bootstrap_scale


if __name__ == "__main__":
    print(
        "=" * 80
    )
    print(
        "s2_68 FIXED — VECTORIZED SUBJECT BOOTSTRAP"
    )
    print(
        "=" * 80
    )
    print(
        "Scientific protocol unchanged; only bootstrap computation optimized."
    )

    base.main()
