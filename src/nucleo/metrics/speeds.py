"""
nucleo.speeds
------------------------
Analysis functions for analyzing speed data.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

# 1.1 : Standard
import numpy as np

# 1.2 : Package
from nucleo.simulation.chromatin import clc_alpha_mean
from nucleo.metrics.utils import clc_distrib


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


def clc_th_speed(
    algo: str,
    s: int, l: int,
    mu: float,
    alphaf: float, alphao: float, 
    rcapt: float, rrest: float,
    alphac: float,
    alphar: float, Kp: float
    ) -> float:
    """
    Calculate the theoretical average speed.
    Loop Extrusion related.
    """
    alpha_mean = clc_alpha_mean(s, l, alphaf, alphao, alphar, Kp)

    if algo == "one_step":
        return mu * alpha_mean
    
    elif algo == "two_steps":
        rates_mean = (1 / (rcapt)) + (1 / (rrest))
        return mu * alphac / rates_mean * alpha_mean


def clc_inst_speeds(
    t_matrix: np.ndarray, 
    x_matrix: np.ndarray, 
    first_bin: float = 0,
    last_bin: float = 1e5,
    bin_width: float = 1.0,
) -> tuple[
    np.ndarray, np.ndarray, float, float, float, 
    np.ndarray, np.ndarray, float, float, float, 
    np.ndarray, np.ndarray, float, float, float
]:
    """
    Calculate statistics for instantaneous speeds across multiple trajectories.

    Args:
        t_matrix (np.ndarray): Times for all trajectories.
        x_matrix (np.ndarray): Positions for all trajectories.
        n_t (int): Total number of trajectories.
        first_bin (float, optional): Lower bound for the histogram bins. Defaults to 0.
        last_bin (float, optional): Upper bound for the histogram bins. Defaults to 1e6.
        bin_width (float, optional): Width of bins for the speed distribution. Defaults to 1.0.

    Returns:
        tuple: 
            - dx_points (np.ndarray): Points (bin centers) for the displacement distribution (Δx).
            - dx_distrib (np.ndarray): Normalized displacement distribution (Δx).
            - dx_mean (float): Mean displacement (Δx).
            - dx_med (float): Median displacement (Δx).
            - dx_mp (float): Most probable displacement (Δx).
            - dt_points (np.ndarray): Points (bin centers) for the time interval distribution (Δt).
            - dt_distrib (np.ndarray): Normalized time interval distribution (Δt).
            - dt_mean (float): Mean time interval (Δt).
            - dt_med (float): Median time interval (Δt).
            - dt_mp (float): Most probable time interval (Δt).
            - v_points (np.ndarray): Points (bin centers) for the speed distribution.
            - v_distrib (np.ndarray): Normalized speed distribution.
            - v_mean (float): Mean of the instantaneous speeds.
            - v_med (float): Median of the instantaneous speeds.
            - v_mp (float): Most probable instantaneous speed.
    """
    
    # Loop through each trajectory
    n_traj = x_matrix.shape[0]

    # Initialize arrays for displacements, time intervals, and speeds
    dx_array = np.array([None] * n_traj, dtype=object)
    dt_array = np.array([None] * n_traj, dtype=object)
    vi_array = np.array([None] * n_traj, dtype=object)

    for i in range(n_traj):
        x = np.array(x_matrix[i])
        t = np.array(t_matrix[i])


        # Skip NaN-only lines
        if np.all(np.isnan(x)) or np.all(np.isnan(t)):
            continue

        # Calculate displacements (Δx) and time intervals (Δt)
        dx = x[1:] - x[:-1]
        dt = t[1:] - t[:-1]

        # Avoid division by zero or invalid intervals
        valid = (~np.isnan(dx)) & (~np.isnan(dt)) & (dt != 0)

        # Calculate instantaneous speeds (Δx / Δt)
        dx = dx[valid]
        dt = dt[valid]
        dv = dx / dt

        # Filter out non-finite speeds
        valid_speed = np.isfinite(dv)
        dx_array[i] = dx[valid_speed]
        dt_array[i] = dt[valid_speed]
        vi_array[i] = dv[valid_speed]

    # Datas
    dx_list = [a for a in dx_array if a is not None and len(a) > 0]
    dt_list = [a for a in dt_array if a is not None and len(a) > 0]
    vi_list = [a for a in vi_array if a is not None and len(a) > 0]

    # Float
    dx_array = np.concatenate(dx_list).astype(np.float64) if dx_list else np.array([], dtype=np.float64)
    dt_array = np.concatenate(dt_list).astype(np.float64) if dt_list else np.array([], dtype=np.float64)
    vi_array = np.concatenate(vi_list).astype(np.float64) if vi_list else np.array([], dtype=np.float64)

    # Calculate distributions for Δx, Δt, and speeds
    dx_points, dx_distrib = clc_distrib(dx_array, first_bin, last_bin, bin_width)
    dt_points, dt_distrib = clc_distrib(dt_array, first_bin, last_bin, bin_width)
    vi_points, vi_distrib = clc_distrib(vi_array, first_bin, last_bin, bin_width)

    # Compute statistics (mean, median, most probable values)
    if vi_distrib.size > 0:
        dx_mean = float(np.mean(dx_array)) if dx_array.size else 0.0
        dx_med = float(np.median(dx_array)) if dx_array.size else 0.0
        dx_mp  = float(dx_points[np.argmax(dx_distrib)]) if dx_distrib.size else 0.0

        dt_mean = float(np.mean(dt_array)) if dt_array.size else 0.0
        dt_med  = float(np.median(dt_array)) if dt_array.size else 0.0
        dt_mp   = float(dt_points[np.argmax(dt_distrib)]) if dt_distrib.size else 0.0

        vi_mean = float(np.mean(vi_array)) if vi_array.size else 0.0
        vi_med  = float(np.median(vi_array)) if vi_array.size else 0.0
        vi_mp   = float(vi_points[np.argmax(vi_distrib)]) if vi_distrib.size else 0.0

    # Default values if distributions are empty
    else:
        dx_mean = dx_med = dx_mp = 0.0
        dt_mean = dt_med = dt_mp = 0.0
        vi_mean = vi_med = vi_mp = 0.0

    # Return results
    return (
        dx_points, dx_distrib, dx_mean, dx_med, dx_mp,
        dt_points, dt_distrib, dt_mean, dt_med, dt_mp,
        vi_points, vi_distrib, vi_mean, vi_med, vi_mp
    )