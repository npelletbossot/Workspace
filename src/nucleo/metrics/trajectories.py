"""
nucleo.trajectory
------------------------
Analysis functions for analyzing results data.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

# 1.1 : Standard
import numpy as np

# 1.2 : Package
from nucleo.metrics.fitting import linear_fit


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 Reconstituting Trajectories


def reconstitute_mean_trajectory(
    t_matrix: np.ndarray,
    x_matrix: np.ndarray,
    tmax: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    bins  = np.arange(1, tmax + 1, dt, dtype=float)
    sum_x = np.zeros(tmax)
    count = np.zeros(tmax, dtype=np.int32)

    for t_row, x_row in zip(t_matrix, x_matrix):
        valid = np.isfinite(t_row) & np.isfinite(x_row) & (t_row < tmax)
        t_v = t_row[valid]
        x_v = x_row[valid]

        if t_v.size == 0:
            continue

        order = np.argsort(t_v)
        t_v   = t_v[order]
        x_v   = x_v[order]

        idx = np.searchsorted(t_v, bins, side="right") - 1
        hit = idx >= 0

        sum_x[hit] += x_v[idx[hit]]
        count[hit]  += 1

    mean_x = np.where(count > 0, sum_x / count, np.nan)
    return np.concatenate(([0.0], mean_x[:-1]))


# 2.2 Sites / Base Pairs


def clc_results(
    results: np.ndarray,
    dt: float,
    alpha_0: float,
    lb: int,
) -> tuple:
    """
    Calculate main statistics and derived results for a matrix of trajectories.

    Args:
        results (np.ndarray): A matrix containing the positions for each time step across all trajectories.
        dt (float): Time step size used in the modeling.
        alpha_0 (float): Linear scaling factor for velocity calculations (unused in trajectory definition).
        nt (int): Total number of trajectories.


    Returns:
        tuple: A tuple containing the following main results:
            - mean_results (np.ndarray): The mean trajectory calculated across all trajectories.
            - v_mean (float): The velocity derived from the mean trajectory, scaled by alpha_0.
            - err_v_mean (float): Bootstrapped error of the mean velocity.
            - med_results (np.ndarray): The median trajectory calculated across all trajectories.
            - v_med (float): The velocity derived from the median trajectory, scaled by alpha_0.
            - err_v_med (float): Error associated with the median velocity (currently set to 0).
            - std_results (np.ndarray): Standard deviation of the trajectories at each time step.

    Notes:
        - This function assumes that `results` contains no invalid data (e.g., NaNs), or they are handled correctly with `np.nanmean` and `np.nanstd`.
        - The velocity calculations use a linear fit applied to the mean and median trajectories.
        - Bootstrapping is used to estimate the error of the mean velocity.
    Accepte :
      - results 2D (nt, tmax) : calcule mean/median/std puis fit
      - results 1D (tmax,)    : déjà la trajectoire moyenne, fit direct
    """
    if results.ndim == 2:
        mean_results = np.nanmean(results, axis=0)
        med_results  = np.nanmedian(results, axis=0)
        std_results  = np.nanstd(results, axis=0)
    elif results.ndim == 1:
        mean_results = results
        med_results  = results       # même courbe, pas d'info supplémentaire
        std_results  = np.full_like(results, np.nan)  # indisponible
    else:
        raise ValueError(f"results must be 1D or 2D, got shape {results.shape}")

    v_mean = linear_fit(mean_results[lb:], dt, offset=lb) * alpha_0
    v_med  = linear_fit(med_results[lb:],  dt, offset=lb) * alpha_0

    return mean_results, med_results, std_results, v_mean, v_med
