"""
nucleo.metrics_landscape_functions
------------------------
Analysis functions for analyzing landscape data.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

# 1.1 : Standard 
import numpy as np


# 1.1 : Package
from nucleo.simulation.chromatin import find_blocks
from nucleo.metrics.utils import clc_distrib


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


def find_interval_containing_value(
    intervals: list[tuple[int, int]], value: int
) -> tuple[int, int]:
    """
    Return the first interval (start, end) that contains the specified value.

    Parameters
    ----------
    intervals : list[tuple[int, int]]
        A list of intervals (start, end) sorted or unsorted.
    
    value : int
        The index or position to locate within the intervals.

    Returns
    -------
    Optional[tuple[int, int]]
        The interval that contains the value, or None if not found.
    """
    intervals_array = np.array(intervals)
    mask = (intervals_array[:, 0] <= value) & (value < intervals_array[:, 1])

    
    if np.any(mask):
        return tuple(intervals_array[mask][0])


def clc_link_view(
    alpha_matrix: np.ndarray, 
    landscape:str, alphaf: float, Lmin: int, Lmax: int,
    nt: int,
    view_size=1_000, threshold=1_000):
    """
    Calculate the average landscape around linker regions for multiple trajectories.

    This function processes a matrix of alpha arrays (trajectories) to extract the 
    regions around linker blocks and computes the average local landscape. 
    It filters out linkers too close to the edges (controlled by `threshold`)
    and focuses only on a fixed window size (`view_size`) around the linker start points.

    Parameters
    ----------
    data : np.ndarray
        A 2D array of shape (nt, Lmax) containing alpha values for each trajectory.
        Each row corresponds to the landscape of one trajectory.
    landscape : str
        The scenario.
    nt : int
        Number of trajectories to process. Must match the number of rows in `data`.
    alphaf : float
        Alpha value defining the identy of a linker in order to find regions with `find_blocks`.
    Lmin : int
        First point of chromatin.
    Lmax : int
        Last point of chromatin.
    view_size : int, optional
        Size of the window around each linker start point to extract. Default is 10_000.
    threshold : int, optional
        Margin to exclude linkers too close to the start or end of the array. Default is 10_000.

    Returns
    -------
    view_mean : np.ndarray
        A 1D array of length `view_size` representing the average landscape around 
        linker regions across all trajectories.

    Raises
    ------
    ValueError
        If 'homogeneous' linker does not really exist. 
        If `threshold` is larger than half of Lmax.
        If `view_size` is larger than 10,000.
        If `data` contains only one trajectory or is not a matrix.
        If `len(data)` is different from `nt`.

    Notes
    -----
    - `find_blocks()` is assumed to return pairs of linker regions based on `alpha_value`.
    - Only the first position of each pair is used.
    - Linkers too close to the boundaries are excluded to ensure full window extraction.
    - Averages are computed first for each trajectory, then globally across all.
    """

    # Conditions on inputs
    if landscape == "homogeneous":
        view_mean = np.array(alpha_matrix[0][threshold:threshold+view_size], dtype=float)
        return view_mean
    if threshold > Lmax // 2:
        raise ValueError("You set the threshold too big !")
    if view_size > Lmax // 2:
        raise ValueError("You set the view_size superior to Lmax // 2!")
    if Lmax == 50_000:
        view_size, threshold = 10_000, 10_000

    # Calculation
    view_datas = np.empty((nt, view_size), dtype=float)

    # Main loop                   
    for _ in range(0,nt):

        # Extracting values
        alpha_array = alpha_matrix[_]                                           # Array data for one trajectory
        pairs_of_linkers = find_blocks(array=alpha_array, alpha_value=alphaf)   # All pairs of linker zones
        pairs_of_linkers = np.array(pairs_of_linkers, dtype=int)                # Conversion in array to get only the first values
        column_of_linkers = pairs_of_linkers[:, 0]                              # Extracting only the first values of couples : first point

        # Filtering to stay within limits
        filter_bounds = (column_of_linkers >= Lmin + threshold) & \
                        (column_of_linkers <= Lmax - threshold - view_size)
        column_of_linkers = column_of_linkers[filter_bounds]

        # Initialisation of a numpy matrix for each personal linker view
        n_linker = len(column_of_linkers)
        view_matrix = np.empty((n_linker, view_size), dtype=float)

        # Line-by-line filling
        for rank, o_link in enumerate(column_of_linkers):
            portion_of_alpha = alpha_array[o_link : o_link + view_size]
            view_matrix[rank, :] = portion_of_alpha  # On suppose que portion_of_alpha a bien la bonne taille

        # Getting results of one trajectory for every linkers
        view_array = np.mean(view_matrix, axis=0)   # Average per column
        view_datas[_] = view_array                  # Filling the all datas matrix

    # Mean and return
    view_mean = np.mean(view_datas, axis=0)
    if np.isnan(view_mean).all():
        return np.zeros(len(view_mean), dtype=float)
    else:
        return view_mean


def clc_obs_and_link_distrib(
    alpha_scenario: str, s: int, l: int,
    alpha_array: np.ndarray, alphaf: float, alphao: float,
    step: int
) -> tuple[float, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Process a 1D alpha array to calculate lengths of linker and obstacle sequences
    and their distributions.

    Args:
        alpha_array (np.ndarray): 1D array representing linkers (alphaf) and obstacles (alphao).
        alphaf (float): Value representing the linkers.
        alphao (float): Value representing the obstacles.
        step (int): Step size for the distribution calculation (default is 1).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - points_o (np.ndarray): Centers of bins for obstacle lengths.
            - distrib_o (np.ndarray): Normalized distribution of obstacle lengths.
            - points_l (np.ndarray): Centers of bins for linker lengths.
            - distrib_l (np.ndarray): Normalized distribution of linker lengths.
    """
    
    # Concerning flat landscape
    if alpha_scenario == "homogeneous":
        return (
            np.float64(s), np.array([0.0]), np.array([0.0]), 
            np.float64(l), np.array([0.0]), np.array([0.0])
        )
    
    # Masks for obstacles and linkers
    mask_o = alpha_array == alphao
    mask_l = alpha_array == alphaf

    # Obstacles
    diffs_o = np.diff(np.concatenate(([0], mask_o.astype(int), [0])))
    starts_o = np.where(diffs_o == 1)[0]
    ends_o = np.where(diffs_o == -1)[0]
    counts_o = ends_o - starts_o

    if counts_o.size > 0:
        mean_o = float(np.mean(counts_o))
        points_o, distrib_o = clc_distrib(
            data=counts_o,
            first_bin=0,
            last_bin=np.max(counts_o) + step,
            bin_width=step
        )
    else:
        mean_o = 0.0
        points_o, distrib_o = np.array([0.0]), np.array([0.0])

    # Linkers
    diffs_l = np.diff(np.concatenate(([0], mask_l.astype(int), [0])))
    starts_l = np.where(diffs_l == 1)[0]
    ends_l = np.where(diffs_l == -1)[0]
    counts_l = ends_l - starts_l


    # Adding blocks of roadblocks to linkers of size 0 into the distribution
    roadblocks = counts_o // s
    values, counts = np.unique(roadblocks, return_counts=True)

    zeros_to_add = []
    for i in range(1, len(values)):
        zeros_to_add.extend([0] * (counts[i] * (values[i] - 1)))

    counts_l = np.concatenate((counts_l, np.array(zeros_to_add)))
    # print(values, counts, counts_l)
            
    # Linker data
    if counts_l.size > 0:
        mean_l = float(np.mean(counts_l))
        points_l, distrib_l = clc_distrib(
            data=counts_l,
            first_bin=0,
            last_bin=np.max(counts_l) + step,
            bin_width=step
        )
    else:
        mean_l = 0.0
        points_l, distrib_l = np.array([0.0]), np.array([0.0])

    return mean_o, points_o, distrib_o, mean_l, points_l, distrib_l