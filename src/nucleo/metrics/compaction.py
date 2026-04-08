"""
nucleo.compaction
------------------------
Analysis functions for analyzing speed data.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

# 1.1 : Standard
import numpy as np

# 1.2 : Package
from nucleo.metrics.utils import clc_distrib


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 First Method


def clc_bp(segment, alphaf, alphao, c_linker, c_nucleo):
    n_alphaf = np.count_nonzero(segment == alphaf)
    n_alphao = np.count_nonzero(segment == alphao)
    n_tot = n_alphaf + n_alphao

    if n_tot == 0:
        return np.nan

    return ((c_linker * n_alphaf) + (c_nucleo * n_alphao)) / n_tot

    
def clc_bp_speeds(
    algorithm: str, alphaf: float, alphao: float, c_linker: float, c_nucleo: float,
    alpha_matrix: np.ndarray, t_matrix: np.ndarray, x_matrix: np.ndarray
):
    """
    Compute compaction-corrected velocities (in base pairs per unit time)
    from position and time trajectories over a chromatin landscape.

    For each trajectory, the function computes the instantaneous velocity
    between successive positions and renormalizes it by a local compaction
    factor derived from the underlying chromatin landscape. The compaction
    factor is computed as a weighted average of linker and nucleosomal
    contributions over the genomic segment crossed during each jump.

    The chromatin landscape is encoded in `alpha_matrix`, where values
    `alphaf` and `alphao` represent linker and nucleosomal regions,
    respectively.

    NaN values in the input position arrays are ignored. Segments of zero
    length are allowed but may yield NaN values if no landscape information
    is available.

    Parameters
    ----------
    alpha_matrix : np.ndarray
        Array of chromatin landscapes. Each row corresponds to a trajectory
        and encodes the chromatin state along the genome.
    t_matrix : np.ndarray
        Array of time points for each trajectory.
    x_matrix : np.ndarray
        Array of genomic positions corresponding to `t_matrix`.
    alphaf : float
        Value representing linker regions in `alpha_matrix`.
    alphao : float
        Value representing nucleosomal regions in `alpha_matrix`.
    c_linker : float
        Compaction factor associated with linker DNA.
    c_nucleo : float
        Compaction factor associated with nucleosomal DNA.

    Returns
    -------
    np.ndarray
        One-dimensional array containing all compaction-corrected velocities
        (in base pairs per unit time) for all trajectories, with NaN values
        removed.
    """
  
    n = len(x_matrix)
    bp_matrix = np.full_like(x_matrix, np.nan, dtype=float)
    
    for i in range(n):
        t_list = t_matrix[i]
        x_list = x_matrix[i]
        alpha_list = alpha_matrix[i]
        
        # filtrering non NaN
        valid = ~np.isnan(x_list)
        x_list_valid = x_list[valid]
        t_list_valid = t_list[valid]
        
        if len(x_list_valid) < 2:
            continue
        
        delta_x = x_list_valid[1:] - x_list_valid[:-1]
        delta_t = t_list_valid[1:] - t_list_valid[:-1]
        delta_v = delta_x / delta_t
        delta_bp = np.zeros_like(delta_v, dtype=float)
        
        for j in range(len(delta_v)):
            start = int(x_list_valid[j])
            end   = int(x_list_valid[j+1])
            segment = alpha_list[start:end]
            c = clc_bp(segment, alphaf, alphao, c_linker, c_nucleo)
            delta_bp[j] = delta_v[j] * c
            
        bp_matrix[i, :len(delta_bp)] = delta_bp
        vi_bp_array = bp_matrix[~np.isnan(bp_matrix)]
    
    if algorithm == "one_step":
        return vi_bp_array
    elif algorithm == "two_steps":
        return vi_bp_array[vi_bp_array > 0]
    

# 2.2 Second Method


def clc_compaction_landscape(alpha_matrix: np.ndarray) -> np.ndarray:
    """
    Cumulative sum of alpha along the genome axis.
    Used to compute sum(alpha, x0→x1) = C[x1] - C[x0] in O(1).
    """
    return np.cumsum(alpha_matrix, axis=1)


def clc_compaction_speeds(
    alpha_matrix_c: np.ndarray,
    t_matrix: np.ndarray,
    x_matrix: np.ndarray,
    c_linker: float = 1.0,
    c_nucleo: float = 150 / 35,
) -> np.ndarray:
    """
    delta_bp = c_nucleo * delta_x + (c_linker - c_nucleo) * (C[x1] - C[x0])
    v_bp     = delta_bp / delta_t
    """
    n_i, n_j = x_matrix.shape
    vc_array = np.full((n_i, n_j - 1), np.nan)

    for i in range(n_i):
        for j in range(n_j - 1):
            xf_float = x_matrix[i, j + 1]
            if np.isnan(xf_float):
                continue

            xi = int(x_matrix[i, j])
            xf = int(xf_float)
            dt = t_matrix[i, j + 1] - t_matrix[i, j]
            if dt <= 0:
                continue

            delta_x  = xf - xi
            sum_alpha = alpha_matrix_c[i, xf] - alpha_matrix_c[i, xi]

            delta_bp = c_nucleo * delta_x + (c_linker - c_nucleo) * sum_alpha
            vc_array[i, j] = delta_bp / dt

    return vc_array


def clc_compaction_statistics(
        alpha_matrix: np.ndarray, t_matrix: np.ndarray, x_matrix: np.ndarray,
        c_linker: float, c_nucleo: float
    ):

    alpha_matrix_c = clc_compaction_landscape(alpha_matrix)
    vc_array = clc_compaction_speeds(alpha_matrix_c, t_matrix, x_matrix, c_linker, c_nucleo)

    vc_mean = np.nanmean(vc_array)
    vc_med  = np.nanmedian(vc_array)

    vc_points, vc_distrib = clc_distrib(data=vc_array, first_bin=0, last_bin=1000, bin_width=1)
    vc_mp   = vc_points[np.where(vc_distrib == np.max(vc_distrib))]

    return vc_points, vc_distrib, vc_mean, vc_med, vc_mp
