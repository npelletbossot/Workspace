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


# 2.1 Reconsituting Trajectories




# 2.2 Sites / Base Pairs


def clc_results(results: np.ndarray, dt: float, alpha_0: float, lb: int) -> tuple:
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
    """

    mean_results = np.nanmean(results, axis=0)      # Calculate mean trajectory across all trajectories
    med_results = np.nanmedian(results, axis=0)     # Calculate median trajectory across all trajectories
    std_results = np.nanstd(results, axis=0)        # Calculate the standard deviation of the trajectories

    v_mean = linear_fit(mean_results[lb:], dt, offset=lb) * alpha_0    # Calculate the velocity for the mean trajectory
    v_med = linear_fit(med_results[lb:], dt, offset=lb) * alpha_0      # Calculate the velocity for the median trajectory

    return mean_results, med_results, std_results, v_mean, v_med
