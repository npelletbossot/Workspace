"""
nucleo.fitting_functions
------------------------
Fitting functions for analyzing processivities, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np
from scipy.stats import linregress


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


def linear_fit(array: np.ndarray, step: float) -> float:
    """
    Linear regression without forcing intercept = 0.
    Ignores NaN values in the array.

    Args:
        array (np.ndarray): Input array of values.
        step (float): Step size.
        offset (int): Offset added to indices before scaling.

    Returns:
        float: Slope of the regression line, or np.nan if not enough valid data.
    """

    valid_mask = ~np.isnan(array)
    y = array[valid_mask]

    if len(y) < 2:
        return np.nan

    idx = np.arange(len(array))[valid_mask]
    x = idx
    x = x * step
    x = x[:, np.newaxis]  # shape (N, 1)

    # Add column of 1s for intercept
    X = np.hstack([x, np.ones_like(x)])

    # Least squares → returns [slope, intercept]
    slope, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return slope[0]



def filtering_before_fit(
    time: np.ndarray, 
    data: np.ndarray, 
    std: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters the input arrays to remove invalid or problematic data points before fitting.

    Args:
        time (np.ndarray): Array of time values.
        data (np.ndarray): Array of data values.
        std (np.ndarray): Array of standard deviation values.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered time, data, and standard deviation arrays.
    """

    # Condition on lenghts
    if len(time) == len(data) == len(std) :

        # Convert inputs to NumPy arrays if they are not already
        time = np.array(time)
        data = np.array(data)
        std = np.array(std)

        # Filter out NaN, infinite values, and invalid data points
        valid_idx = ~np.isnan(data) & ~np.isnan(std) & ~np.isinf(data) & ~np.isinf(std)
        time = time[valid_idx]
        data = data[valid_idx]
        std = std[valid_idx]

        # Replace standard deviations of 0 with a small positive value
        std = np.where(std == 0, 1e-10, std)

        return time, data, std

    else :
        print("Problem with arrays : not the same lenghts")
        return None


def fitting_in_two_steps(times, positions, deviations, bound_low, bound_mid, bound_high):
    """
    Perform a two-step fit on trajectory data: 
    - a linear fit on x(t)/t for early-time behavior,
    - a log-log fit for power-law behavior at later times.

    This function automatically handles edge cases where the dataset is too short 
    to perform the second fit, returning `None` for those values.

    Args:
        times (np.ndarray): Array of time values.
        positions (np.ndarray): Array of average positions (x(t)).
        deviations (np.ndarray): Array of standard deviations.
        bound_low (int): Number of initial points to use for the linear average.
        bound_high (int): Starting index for the power-law log-log fit.
        rf (int): Rounding factor for the returned values.

    Returns:
        tuple: Rounded values for:
            - vf (float or None): Average x(t)/t in early phase.
            - Cf (float or None): Prefactor from log-log fit.
            - wf (float or None): Exponent from log-log fit.
            - vf_std (float or None): Standard deviation of vf.
            - Cf_std (float or None): Error estimate on Cf.
            - wf_std (float or None): Error estimate on wf.
    """

    # Filter data before fitting
    times, positions, deviations = filtering_before_fit(times, positions, deviations)

    # Getting the filter points in %
    n_pts = len(times)
    lb_idx = int(bound_low / 100 * n_pts)
    mb_idx = int(bound_mid / 100 * n_pts)
    ub_idx = int(bound_high / 100 * n_pts)

    # Remove the first point to avoid (0, 0)
    times = times[1:]
    positions = positions[1:]

    # Check if there's enough data for both bounds
    if len(positions) < max(ub_idx, lb_idx + 2):
        return None, None, None, None, None, None, None, None, None, None

    # Step 1: linear average of x(t)/t over early time
    xt_over_t = np.divide(positions, times)
    array_low = xt_over_t[lb_idx:mb_idx]
    vf = np.mean(array_low)
    vf_std = np.std(array_low)

    # Step 2: logarithmic Derivative (G) to observe where the bound_high is - helps plots
    dlogx = np.diff(np.log(positions))
    dlogt = np.diff(np.log(times))
    G = np.divide(dlogx, dlogt)

    # Step 3: check if there are enough points for log-log fit
    if len(times) <= ub_idx + 1:
        return None, None, None, None, None, None, None, None, None, None

    # Step 4: log-log fit of x(t) = Cf * t^wf on the right side
    log_t_high = np.log(times[ub_idx:])
    log_x_high = np.log(positions[ub_idx:])
    coeffs, cov = np.polyfit(log_t_high, log_x_high, 1, cov=True)
    wf = coeffs[0]
    Cf = np.exp(coeffs[1])
    wf_std = np.sqrt(cov[0, 0])
    Cf_std = Cf * np.sqrt(cov[1, 1])

    return (
        vf, Cf, wf,
        vf_std, Cf_std, wf_std,
        xt_over_t, G
    )