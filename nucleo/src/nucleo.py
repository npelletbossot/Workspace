#!/usr/bin/env python
# coding: utf-8


# ================================================
# Part 1 : Imports
# ================================================


# 1.1 : Standard library imports
import os
import gc
import time
import math
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from itertools import groupby, product
from collections import Counter
from typing import Callable, Tuple, List, Dict, Optional
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# 1.2 Third-party library imports
import numpy as np
from scipy.stats import gamma
from scipy.stats import linregress
from scipy.optimize import curve_fit
import pyarrow as pa
import pyarrow.parquet as pq


# 1.3 Matplotlib if required
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ================================================
# Part 2.1 : General functions
# ================================================


# 2.1.1 : Dictionaries


def add_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge two dictionaries by summing their values for common keys.

    Args:
        dict1 (dict): First dictionary with numeric values.
        dict2 (dict): Second dictionary with numeric values.

    Returns:
        dict: A new dictionary with keys from both dictionaries, 
              where the values are the sum of the values from dict1 and dict2.
    """
    merged_dict = defaultdict(int)

    # Add values from dict1 to merged_dict
    for key, value in dict1.items():
        merged_dict[key] += value

    # Add values from dict2 to merged_dict
    for key, value in dict2.items():
        merged_dict[key] += value

    return dict(merged_dict)


def compute_mean_from_dict(input_dict: dict) -> dict:
    """
    Compute the mean of all list values in a dictionary.

    Args:
        input_dict (dict): Dictionary where keys map to lists of numeric values.

    Returns:
        dict: A new dictionary with the same keys, where each value is the mean
              of the corresponding list from the input dictionary.
    """
    mean_dict = {}
    for key, values in input_dict.items():
        if isinstance(values, list):
            mean_dict[key] = np.mean(np.array(values))
    return mean_dict


# 2.1.2 : Calculations


def calculate_distribution(
    data: np.ndarray, 
    first_bin: float, 
    last_bin: float, 
    bin_width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the normalized distribution of data using a histogram.

    Args:
        data (np.ndarray): Array of data values to compute the distribution for.
        first_bin (float): Lower bound of the first bin.
        last_bin (float): Upper bound of the last bin.
        bin_width (float): Width of each bin.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - points (np.ndarray): Array of bin centers.
            - distrib (np.ndarray): Normalized distribution (sum equals 1).
    """

    # # Remove NaN values
    # data = data[~np.isnan(data)]

    # Handle empty data array
    if data.size == 0: 
        return np.array([]), np.array([])

    # Points and not bins
    bins_array = np.arange(first_bin, int(last_bin) + bin_width, bin_width)
    distrib, bins_edges = np.histogram(data, bins=bins_array)

    # Normalizing without generating NaNs
    if np.sum(distrib) > 0:
        distrib = distrib / np.sum(distrib)
    else:
        distrib = np.zeros_like(distrib)

    points = (bins_edges[:-1] + bins_edges[1:]) / 2

    # Return the bin centers and the normalized distribution
    return points, distrib


def linear_fit(array: np.ndarray, step: float) -> float:
    """
    Calculate the slope of a linear regression constrained to pass through the origin (0, 0),
    ignoring NaN values in the input array.

    Args:
        array (np.ndarray): Input array of values.
        step (float): Step size for the analysis.

    Returns:
        float: Slope of the linear regression, or np.nan if pas assez de données valides.
    """

    valid_mask = ~np.isnan(array)
    valid_array = array[valid_mask]
    
    if len(valid_array) < 2:
        return np.nan

    x = np.arange(0, len(array)) * step
    x = x[valid_mask]
    x = x[:, np.newaxis]

    slope, _, _, _ = np.linalg.lstsq(x, valid_array, rcond=None)
    return slope[0]


def exp_decay(t, y0, tau):
    return y0 * np.exp(-t / tau)


# 2.1.3 : Datas


def listoflist_into_matrix(listoflist: list) -> np.ndarray:
    """
    Converts a list of lists with varying lengths into a 2D NumPy array,
    padding shorter rows with np.nan so that all rows have equal length.
    """
    len_max = max(len(row) for row in listoflist)
    matrix = np.full((len(listoflist), len_max), np.nan)
    for i, row in enumerate(listoflist):
        matrix[i, :len(row)] = row
    return matrix


# ================================================
# Part 2.2 : Probability functions
# ================================================


def proba_tataki(R: float, L: float, lp: float) -> float:
    """
    Compute the probability density function (PDF) for a given model.

    Args:
        R (float): Radial distance or observed value.
        L (float): Characteristic length scale of the system.
        lp (float): Persistence length of the system.

    Returns:
        float: Probability density function value for the given parameters.

    Notes:
        - This function models a probability based on a mathematical expression involving
          parameters `R`, `L`, and `lp`.
        - The calculation includes terms like the persistence length (`lp`) and 
          radial distance to length ratio (`R/L`).
    """
    # Compute auxiliary parameters
    t = (3 * L) / (2 * lp)
    alpha = (3 * t) / 4

    # Compute the normalization factor (N)
    N = (4 * (alpha ** (3 / 2))) / (((math.pi) ** (3 / 2)) * (4 + (12 * alpha ** (-1)) + 15 * (alpha ** (-2))))

    # Set constant A (scaling factor)
    A = 1

    # Compute the probability density function
    PLR = A * ((4 * math.pi * N) * ((R / L) ** 2)) / (L * ((1 - (R / L) ** 2)) ** (9 / 2)) \
          * np.exp(alpha - (3 * t) / (4 * (1 - (R / L) ** 2)))

    return PLR


def proba_gamma(mu: float, theta: float, L: float) -> float:
    """
    Compute the probability density function (PDF) of a Gamma distribution.

    Args:
        mu (float): Mean of the Gamma distribution.
        theta (float): Standard deviation of the Gamma distribution.
        L (float): Value at which to evaluate the probability density.

    Returns:
        float: Probability density at the given value L for the Gamma distribution.
    """
    alpha_gamma = mu**2 / theta**2                              # Calculate the shape parameter (alpha) of the Gamma distribution
    beta_gamma = theta**2 / mu                                  # Calculate the scale parameter (beta) of the Gamma distribution
    p_gamma = gamma.pdf(L, a=alpha_gamma, scale=beta_gamma)     # Compute the probability density for the value L

    p_gamma = p_gamma / np.sum(p_gamma)

    return p_gamma


# ================================================
# Part 2.3 : Landscape functions
# ================================================


def alpha_random(s:int, l:int, alphao:float, alphaf:float, L_min:int, Lmax:int, bps:int) -> np.ndarray:
    """
    Generates one random landscape


    Args:
        s (int): Value of s, nucleosome size.
        l (int): Value of l, linker length.
        alphao (float): Probability of beeing accepted on nucleosome sites.
        alphaf (float): Probability of beeing accepted on linker sites.
        Lmin (int): First point of chromatin.
        Lmax (int): Last point of chromatin.
        bps (int): Number of base pairs per site.

    Returns:
        np.ndarray: Landscape corresponding to one trajectory.
    """
    np.random.seed()

    alpha_array = np.full(int((Lmax-L_min)/bps), alphaf)    # creates a NumPy array of length Lmax filled with the value alphaf
    T = int(Lmax / (l + s))                                 # how many blocks of size l + s can fit into the total length Lmax ; represents how many obstacle blocks can be inserted into the list
    max_pos = Lmax - (T * s)                                # determines the maximum possible position for obstacle blocks, taking into account the fact that each block occupies s positions 

    # random generation of position to insert obstacles
    alpha_random = np.random.randint(0, max_pos + 1, T)         # creates an array of random indices to place the obstacle blocks, ranging from 0 to _max_position_ inclusive, with a total of _T_ positions
    alpha_random_sorted = np.sort(alpha_random)                 # sorts random positions to avoid overlapping and ensure orderly placement of obstacle blocks
    alpha_random_modified = alpha_random_sorted + np.arange(len(alpha_random_sorted)) * (s)     # each point is shifted ([2 + 0, 3 + s, 6 + 2*s ....] etc) to prevent overlapping
    
    # filling with obstacles
    for pos in alpha_random_modified:
        alpha_array[pos:pos + s] = alphao                                                       # for each position modified in _alpha_random_modified_, place an obstacle block (alphao) of length s in the _alpha_array

    return alpha_array


def alpha_periodic(s:int, l:int, alphao:float, alphaf:float, Lmin:int, Lmax:int, bps:int) -> np.ndarray:
    """Generates one periodic pattern

    Args:
        s (int): Value of s, nucleosome size.
        l (int): Value of l, linker length.
        alphao (float): Probability of beeing accepted on nucleosome sites.
        alphaf (float): Probability of beeing accepted on linker sites.
        Lmin (int): First point of chromatin.
        Lmax (int): Last point of chromatin.
        bps (int): Number of base pairs per site.

    Returns:
        np.ndarray: Landscape corresponding to one trajectory.
    """

    N = int(int((Lmax-Lmin)/bps) // (l + s))
    residue = Lmax - N * (l + s)
    pattern = np.concatenate((np.full(l, alphaf), np.full(s, alphao)))
    alpha_array = np.concatenate((np.tile(pattern, N), np.full(residue, alphaf)))
    
    return alpha_array


def alpha_constant(s:int, l:int, alphao:float, alphaf:float, Lmin:int, Lmax:int, bps:int) -> np.ndarray:
    """Generates one flat pattern

    Args:
        s (int): Value of s, nucleosome size.
        l (int): Value of l, linker length.
        alphao (float): Probability of beeing accepted on nucleosome sites.
        alphaf (float): Probability of beeing accepted on linker sites.
        Lmin (int): First point of chromatin.
        Lmax (int): Last point of chromatin.
        bps (int): Number of base pairs per site.

    Returns:
        np.ndarray: Landscape corresponding to one trajectory.
    """

    value = (alphao * s + alphaf * l) / (l + s)
    size = int((Lmax - Lmin) / bps)
    alpha_array = np.full(size, value)

    return alpha_array


def alpha_matrix_calculation(alpha_choice:str, s:int, l:int, bpmin:int, alphao:float, alphaf:float, Lmin:int, Lmax:int, bps:int, nt:int) -> np.ndarray:
    """
    Calculation of the matrix of obstacles, each line corresponding to a trajectory

    Args:
        alpha_choice (str): Choice of the alpha configuration ('ntrandom', 'periodic', 'constantmean').
        s (int): Value of s, nucleosome size.
        l (int): Value of l, linker length.
        bpmin (int): Minimum base pair threshold.
        alphao (float): Probability of beeing accepted on linker sites.
        alphaf (float): Probability of beeing accepted on nucleosome sites.
        Lmin (int): First point of chromatin.
        Lmax (int): Last point of chromatin.
        bps (int): Number of base pairs per site.
        nt (int): Number of trajectories for the simulation.


    Raises:
        ValueError: In case the choice is not aligned with the possibilities

    Returns:
        np.ndarray: Matrix of each landscape corresponding to a trajectory
    """

    alpha_functions = {'periodic', 'one_random', 'ntrandom', 'constantmean'}
    
    if alpha_choice not in alpha_functions:
        raise ValueError(f"Unknown alpha_choice: {alpha_choice}")
    
    elif alpha_choice == 'periodic' :
        alpha_array = alpha_periodic(s, l, alphao, alphaf, Lmin, Lmax, bps)
        alpha_matrix = np.tile(alpha_array, (nt,1))

    elif alpha_choice == 'one_random' :
        alpha_array = alpha_random(s, l, alphao, alphaf, Lmin, Lmax, bps)
        alpha_matrix = np.tile(alpha_array, (nt,1))
    
    elif alpha_choice == 'constantmean' :
        alpha_array = alpha_constant(s, l, alphao, alphaf, Lmin, Lmax, bps)
        alpha_matrix = np.tile(alpha_array, (nt,1))

    elif alpha_choice == 'ntrandom':
        alpha_matrix = np.empty((nt, int((Lmax - Lmin) / bps)))
        for i in range(nt):
            alpha_matrix[i] = alpha_random(s, l, alphao, alphaf, Lmin, Lmax, bps)

    # Values
    alpha_matrix = np.array([binding_length(alpha, alphao, alphaf, bpmin) for alpha in alpha_matrix], dtype=float)
    mean_alpha = np.mean(alpha_matrix, axis=0)
    
    return alpha_matrix, mean_alpha


def binding_length(alpha_list: np.ndarray, alphao: float, alphaf: float, bpmin: int) -> np.ndarray:
    """
    Modifies sequences of consecutive `alphaf` values in an array if their length is less than `bpmin`.

    This function takes an input array `alpha_list` and checks for sequences of consecutive
    elements equal to `alphaf`. If the length of any such sequence is less than `bpmin`,
    all values in that sequence are replaced with `alphao`.

    Parameters:
    -----------
    alpha_list : np.ndarray
        The input array of numerical values to process.
    alphao : float
        The value to replace sequences with if their length is less than `bpmin`.
    alphaf : float
        The value representing sequences of interest in the array.
    bpmin : int
        The minimum length of a sequence of `alphaf` required to remain unchanged.

    Returns:
    --------
    np.ndarray
        A new array where sequences of `alphaf` with a length smaller than `bpmin`
        have been replaced by `alphao`.

    Example:
    --------
    >>> alpha_list = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    >>> alphao = 0
    >>> alphaf = 1
    >>> bpmin = 2
    >>> binding_length(alpha_list, alphao, alphaf, bpmin)
    array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    """
    alpha_array = alpha_list.copy()     # Avoid modifying the original input array
    mask = alpha_array == alphaf        # Identify indices where the values are equal to `alphaf`

    # Find start and end indices of consecutive sequences of `alphaf`
    diffs = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(diffs == 1)[0]    # Start of sequences
    ends = np.where(diffs == -1)[0]     # End of sequences

    # Iterate over sequences and replace if the length is less than `bpmin`
    for start, end in zip(starts, ends):
        length = end - start
        if length < bpmin:
            alpha_array[start:end] = alphao

    return alpha_array


def find_blocks(array: np.ndarray, alpha_value: float) -> List[Tuple[int, int]]:
    """
    Identify contiguous regions in the array where values are equal (or close) to a given value.
    Can be used to find obstacles and linkers !

    Parameters
    ----------
    array : np.ndarray
        The array representing the full environment.
    
    value : float
        The value considered as an obstacle (using approximate comparison).

    Returns
    -------
    List[Tuple[int, int]]
        A list of intervals (start_index, end_index) for each contiguous obstacle block.
    """
    array = np.asarray(array)
    is_block = np.isclose(array, alpha_value, atol=1e-8)
    diff = np.diff(is_block.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if is_block[0]:
        starts = np.insert(starts, 0, 0)
    if is_block[-1]:
        ends = np.append(ends, len(array))

    return list(zip(starts, ends))


def find_interval_containing_value(
    intervals: List[Tuple[int, int]], value: int
) -> Optional[Tuple[int, int]]:
    """
    Return the first interval (start, end) that contains the specified value.

    Parameters
    ----------
    intervals : List[Tuple[int, int]]
        A list of intervals (start, end) sorted or unsorted.
    
    value : int
        The index or position to locate within the intervals.

    Returns
    -------
    Optional[Tuple[int, int]]
        The interval that contains the value, or None if not found.
    """
    intervals_array = np.array(intervals)
    mask = (intervals_array[:, 0] <= value) & (value < intervals_array[:, 1])

    
    if np.any(mask):
        return tuple(intervals_array[mask][0])
    return None


def calculate_linker_landscape(data, alpha_choice ,nt, alphaf, Lmin, Lmax, view_size=10_000, threshold=10_000):
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
    alpha_choice : str
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
        If 'constantmean' linker does not really exist. 
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
    if alpha_choice == "constantmean":
        view_mean = np.array(data[0][threshold:threshold+view_size], dtype=float)
        return view_mean
    if threshold > Lmax // 2:
        raise ValueError("You set the threshold too big !")
    if view_size > 10_000:
        raise ValueError("You set the view_size superior to 10_000!")
    if len(data) == 1:
        raise ValueError("You set data as an array and not as a matrix")
    if len(data) != nt:
        raise ValueError("You set nt not equal to len(data)")

    # Calculation
    view_datas = np.empty((nt, view_size), dtype=float)                         # Futur return

    # Main loop                   
    for _ in range(0,nt):

        # Extracting values
        alpha_array = data[_]                                                   # Array data for one trajectory
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

    # Last result and return
    view_mean = np.mean(view_datas, axis=0)
    return view_mean


# ================================================
# Part 2.4 : Modeling functions
# ================================================


def jump(o: int, probabilities: list) -> int:
    """
    Simulate a jump based on cumulative probabilities.

    Args:
        offset (int): Starting position or initial offset (o).
        probabilities (list): List of cumulative probabilities (p), where each value
                              represents the probability threshold for a jump to occur.

    Returns:
        int: The resulting position after the jump.

    """
    
    r = np.random.rand()    # Generate a random number in [0, 1)
    j = 0                   # Initialize the jump counter

    # Increment until the random number is less than the cumulative probability
    while j < len(probabilities) and r >= probabilities[j]:
        j += 1

    return o + j


def attempt(alpha: float) -> bool:
    """
    Perform a validation or refutation attempt based on a given probability threshold.

    Args:
        alpha (float): Probability threshold (between 0 and 1). 
            If a randomly generated number is less than `alpha`, the attempt is successful.

    Returns:
        bool: 
            - `True` if the attempt is successful (random number < alpha).
            - `False` otherwise.
    """
    random_value = np.random.rand()     # Generate a random number in [0, 1)

    if random_value < alpha:
        return True
    else:
        return False
 

def unhooking(beta: float) -> bool:
    """
    Simulate the unhooking (stalling) process based on a probability threshold.

    Args:
        beta (float): Probability threshold (between 0 and 1). 
            If a randomly generated number is less than `beta`, unhooking occurs.

    Returns:
        bool: 
            - `True` if unhooking (stalling) occurs (random number < beta).
            - `False` otherwise.
    """
    random_value = np.random.rand()     # Generate a random number in [0, 1)

    if random_value < beta:
        return True
    else:
        return False


def order() -> bool:
    """
    Determine the priority of execution randomly.

    Returns:
        bool: 
            - `True` if priority is assigned to the first order (randomly chosen as 1).
            - `False` if priority is assigned to the second order (randomly chosen as 2).

    Notes:
        - The function randomly selects between two possible choices (1 or 2).
        - This can be used to simulate a probabilistic decision for execution order.

    """
    chosen_order = np.random.choice(np.array([1, 2]))   # Randomly choose between 1 and 2

    if chosen_order == 1:
        return True
    else:
        return False


def gillespie(r_tot: float) -> float:
    """
    Perform a random draw from a decreasing exponential distribution 
    for use in stochastic simulations (e.g., Gillespie algorithm).

    Args:
        r_tot (float): Total reaction rate or propensity. 
            Must be a positive value.

    Returns:
        float: A random time interval (`delta_t`) sampled from an exponential distribution.

    Notes:
        - The time interval is computed as `delta_t = -log(U) / r_tot`, 
          where `U` is a random number uniformly distributed in [0, 1).
        - This function is commonly used in stochastic simulation algorithms 
          such as the Gillespie algorithm.
    """

    delta_t = -np.log(np.random.rand()) / r_tot     # Generate a random time interval using an exponential distribution
    return delta_t


def folding(landscape:np.ndarray, first_origin:int) -> int:
    """
    Jumping on a random place around the origin, for the first position of the simulation.

    Args:
        landscape (np.ndarray): landscape with the minimum size for condensin to bind.
        origin (int): first point on which condensin arrives.

    Returns:
        int: The real origin of the simulation
    """

    # In order to test but normally we'll never begin any simulation on 0
    if first_origin == 0 :
        true_origin = 0

    else :
        # Constant scenario : forcing the origin -> Might provoc a problem if alpha_f and alpha_o are not 0 or 1 anymore !
        if landscape[first_origin] != 1 and landscape[first_origin] != 0 :
            true_origin = first_origin        

        # Falling on a 1 : Validated
        if landscape[first_origin] == 1 :
            true_origin = first_origin

        # Falling on a 0 : Refuted
        if landscape[first_origin] == 0 :
            back_on_obstacle = 1
            while landscape[first_origin-back_on_obstacle] != 1 :
                back_on_obstacle += 1
            pos = first_origin - back_on_obstacle
            back_on_linker = 1
            while landscape[pos-back_on_linker] != 0 :
                back_on_linker += 1
            true_origin = np.random.randint(first_origin-(back_on_obstacle+back_on_linker), first_origin-back_on_obstacle)+1

    return(true_origin)


# --- One step --- #
def gillespie_algorithm_one_step(
    nt: int, tmax: float, dt: float,
    alpha_matrix: np.ndarray, beta: float, 
    Lmax: int, lenght: int, origin: int, 
    p: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The algorithm on which the values are generated. One step !

    Args:
        nt (int): Number of trajectories for the simulation.
        tmax (float): Maximum time for the simulation.
        dt (float): Time step increment.
        alpha_matrix (np.ndarray): Matrix of acceptance probability.
        beta (float): Unfolding probability.
        Lmax (int): Last point of chromatin.
        lenght (int): Total length of chromatin.
        origin (int): Starting position for the simulation.
        p (np.ndarray): Input probability.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: matrix of results, all time, all positions
    """

    # --- Starting values --- #
    beta_matrix = np.tile(np.full(lenght, beta), (nt, 1))

    results = np.empty((nt, int(tmax/dt)))
    results.fill(np.nan)

    t_matrix = np.empty(nt, dtype=object)
    x_matrix = np.empty(nt, dtype=object)

    # --- Loop on trajectories --- #
    for _ in range(0,nt) :

        # Initialization of starting values
        t = 0
        # x = 0
        x = folding(alpha_matrix[_], origin)    # Initial calculation
        prev_x = np.copy(x)                     # Copy for later use (filling the matrix)
        ox = np.copy(x)                         # Initial point on the chromatin (used to reset trajectories to start at zero)
        i0 = 0                                  # Initial index
        i = 0                                   # Current index

        # Initial calibration
        results[_][0] = t                       # Store the initial time
        t_list = [t]                            # List to track time points
        x_list = [x-ox]                         # List to track recalibrated positions

        # --- Loop on times --- #
        while (t<tmax) :

            # Gillespie values : scanning the all genome
            r_tot = beta_matrix[_][x] + np.nansum(p[1:(Lmax-x)] * alpha_matrix[_][(x+1):Lmax])

            # # Jumping or unbinding : condition on inf times (other version)
            # if np.isinf(t_jump):                # Possibility of generating an infinite time because of no accessible positions
            #     results[_][i0:] = x-ox          # Filling with the last value
            #     break                           # Breaking the loop

            # Next time and rate of reaction
            t = t - np.log(np.random.rand())/r_tot
            r0 = np.random.rand()

            # Condition on time (and not on rtot) in order to capture the last jump and have weight on blocked events
            if np.isinf(t) == True:
                t = 1e308

            # Unhooking or not
            if r0<(beta_matrix[_][x]/r_tot) :
                i = int(np.floor(t/dt))                                     # Last time
                results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = prev_x     # Last value
                break

            # Not beeing in a disturbed area
            if x >= (Lmax - origin) :
                i = int(np.floor(t/dt))                                     # Last time
                results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = np.nan     # No value
                # print('Loop extrusion arrival at the end of the chain.')
                break

            # Choosing the reaction
            else :
                di = 1                                                      # ! di begins to 1 : p(0)=0 !
                rp = beta_matrix[_][x] + p[di] * alpha_matrix[_][x+di]      # Gillespie reaction rate

                while ((rp/r_tot)<r0) and (di<Lmax-1-x) :                   # Sum on all possible states
                    di += 1                                                 # Determining the rank of jump
                    rp += p[di] * alpha_matrix[_][x+di]                     # Sum : element per element

            # Updated parameters
            x += di

            # Acquisition of data
            t_list.append(t)
            x_list.append(x-ox)

            # Filling 
            i = int(np.floor(t/dt))
            results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = int(prev_x-ox)
            i0 = i+1
            prev_x = np.copy(x)

        # All datas
        t_matrix[_] = t_list
        x_matrix[_] = x_list

    return results, t_matrix, x_matrix


# --- Two steps --- #
def gillespie_algorithm_two_steps(
    alpha_matrix: np.ndarray,
    p: np.ndarray,
    beta: float, 
    lmbda: float,
    rtot_bind: float,
    rtot_rest: float,
    nt: int, 
    tmax: float, 
    dt: float,
    L: np.ndarray, 
    origin: int, 
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates stochastic transitions using a two-step Gillespie algorithm.

    Args:
        alpha_matrix (np.ndarray): Matrix of acceptance probabilities.
        p (np.ndarray): Probability array for transitions.
        beta (float): Unfolding probability.
        lmbda (float): Probability to perform a reverse jump after a forward move.
        rtot_bind (float): Reaction rate for binding events.
        rtot_rest (float): Reaction rate for resting events.
        nt (int): Number of trajectories to simulate.
        tmax (float): Maximum simulation time.
        dt (float): Time step increment.
        L (np.ndarray): Chromatin structure array.
        origin (int): Initial position in the simulation.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - A matrix containing the simulation results.
            - An array of all recorded time steps.
            - An array of all recorded positions.
    """


    # --- Starting values --- #
    # beta_matrix = np.tile(np.full(len(L)*bps, beta), (nt, 1))

    results = np.empty((nt, int(tmax/dt)))
    results.fill(np.nan)

    t_matrix = np.empty(nt, dtype=object)
    x_matrix = np.empty(nt, dtype=object)

    # --- Loop on trajectories --- #
    for _ in range(0,nt) :

        # Initialization of starting values
        t, t_bind, t_rest = 0, 0, 0           # First times
        x = folding(alpha_matrix[_], origin)  # Initial calculation
        prev_x = np.copy(x)                   # Copy for later use (filling the matrix)
        ox = np.copy(x)                       # Initial point on the chromatin (used to reset trajectories to start at zero)

        # Model 
        i0, i = 0, 0

        # Initial calibration
        results[_][0] = t                     # Store the initial time
        t_list = [t]                          # List to track time points
        x_list = [x-ox]                       # List to track recalibrated positions

        # --- Loop on times --- #
        while (t<tmax) :

            # --- Unbinding or not --- #

            # # Not needed for the moment
            # r0_unbind = np.random.rand()
            # if r0_unbind<(beta_matrix[_][x]):
            #     i = int(np.floor(t/dt))                                         # Last time
            #     results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = prev_x-ox      # Last value
            #     break

            # --- Jumping : mandatory --- #

            # Almost instantaneous jumps (approx. 20 ms)
            x_jump = np.random.choice(L, p=p)       # Gives the x position
            x += x_jump                             # Whatever happens loop extrusion spends time trying to extrude

            # --- Jumping : edge conditions  --- #
            if x >= (np.max(L) - origin) :
                i = int(np.floor(t/dt))                                         # Last time
                results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = np.nan         # Last value
                break

            # --- Binding or Abortion --- #

            # Binding : values
            r_bind = alpha_matrix[_][x]
            t_bind = - np.log(np.random.rand())/rtot_bind       # Random time of bind or abortion
            r0_bind = np.random.rand()                          # Random event of bind or abortion

            # Condition on time for tentative
            if np.isinf(t_bind) == True:
                t = 1e308

            # Binding : whatever happens loop extrusion spends time trying to bind event if it fails  
            t += t_bind

            # Acquisition 1
            t_list.append(t)
            x_list.append(x-ox)
      
            # Binding : Loop Extrusion does occur - it will have to rest
            if r0_bind < r_bind * (1-lmbda):
                LE = True
                t_rest = - np.log(np.random.rand())/rtot_rest

                if np.isinf(t_rest) == True:
                    t_rest = 1e308

                t += t_rest

            # Binding : Loop Extrusion does not occur - it will not have to rest
            else : 
                LE = False
                x = prev_x

            # Acquisition 2
            t_list.append(t)
            x_list.append(x-ox)

            # Filling
            i = int(np.floor(t/dt))                                   
            results[_][i0:int(min(np.floor(tmax/dt),i)+1)] = int(prev_x-ox)
            i0 = i+1
            prev_x = np.copy(x)

        # All datas
        t_matrix[_] = t_list
        x_matrix[_] = x_list

    return results, t_matrix, x_matrix


def theoretical_speed(alphaf, alphao, s, l, mu, lmbda, rtot_bind, rtot_rest):
        p_alpha = (s*alphao + l*alphaf) / (l+s) * (1-lmbda)
        t_alpha = (1 / rtot_bind) + (1 / rtot_rest)
        x_alpha = mu
        return p_alpha / t_alpha * x_alpha


# ================================================
# Part 2.5 : Analysis calculation functions
# ================================================


def calculate_obs_and_linker_distribution(
    alpha_array: np.ndarray, alphao: float, alphaf: float, step: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a 1D alpha array to calculate lengths of linker and obstacle sequences
    and their distributions.

    Args:
        alpha_array (np.ndarray): 1D array representing linkers (alphaf) and obstacles (alphao).
        alphao (float): Value representing the obstacles.
        alphaf (float): Value representing the linkers.
        step (int): Step size for the distribution calculation (default is 1).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - points_o (np.ndarray): Centers of bins for obstacle lengths.
            - distrib_o (np.ndarray): Normalized distribution of obstacle lengths.
            - points_l (np.ndarray): Centers of bins for linker lengths.
            - distrib_l (np.ndarray): Normalized distribution of linker lengths.
    """
    # Masks for obstacles and linkers
    mask_o = alpha_array == alphao
    mask_l = alpha_array == alphaf

    # Find lengths of obstacle sequences
    diffs_o = np.diff(np.concatenate(([0], mask_o.astype(int), [0])))
    starts_o = np.where(diffs_o == 1)[0]
    ends_o = np.where(diffs_o == -1)[0]
    counts_o = ends_o - starts_o

    # Find lengths of linker sequences
    diffs_l = np.diff(np.concatenate(([0], mask_l.astype(int), [0])))
    starts_l = np.where(diffs_l == 1)[0]
    ends_l = np.where(diffs_l == -1)[0]
    counts_l = ends_l - starts_l

    # Handle empty counts
    if counts_o.size == 0:
        points_o, distrib_o = np.array([]), np.array([])
    else:
        points_o, distrib_o = calculate_distribution(data=counts_o, first_bin=0, last_bin=np.max(counts_o)+step, bin_width=step)

    if counts_l.size == 0:
        points_l, distrib_l = np.array([]), np.array([])
    else:
        points_l, distrib_l = calculate_distribution(data=counts_l, first_bin=0, last_bin=np.max(counts_l)+step, bin_width=step)

    # Returns the distribution on one array !
    return points_o, distrib_o, points_l, distrib_l


def calculate_main_results(results: np.ndarray, dt: float, alpha_0: float, nt: int) -> tuple:
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

    mean_results = np.nanmean(results, axis=0)                    # Calculate mean trajectory across all trajectories
    med_results = np.nanmedian(results, axis=0)                   # Calculate median trajectory across all trajectories
    std_results = np.nanstd(results, axis=0)                      # Calculate the standard deviation of the trajectories

    v_mean = linear_fit(mean_results, dt) * alpha_0            # Calculate the velocity for the mean trajectory
    v_med = linear_fit(med_results, dt) * alpha_0              # Calculate the velocity for the median trajectory

    return mean_results, med_results, std_results, v_mean, v_med


def calculate_position_histogram(results: list, Lmax: int, origin: int, tmax: int, time_step: int = 1) -> np.ndarray:
    """
    Calculate the position histogram for a set of results over time.

    Args:
        results (list): A list of trajectories, where each trajectory is a list of positions over time.
        Lmax (int): Maximum length of the domain.
        origin (int): Offset or starting position for the domain.
        tmax (int): Maximum time for the simulation.
        time_step (int, optional): Time step interval for calculating histograms. Defaults to 1.

    Returns:
        np.ndarray: A 2D array representing the normalized histogram of positions over time.
            Rows correspond to bins, and columns correspond to time steps.

    Notes:
        - The domain is divided into bins ranging from 0 to `Lmax - 2 * origin`.
        - Histograms are calculated for each time step, and the resulting counts are normalized to probabilities.
        - If no data exists for a time step, the corresponding histogram is filled with zeros.
    """

    results_transposed = np.array(results).T                # Transpose the results to process positions at each time step
    num_bins = np.arange(0, Lmax - (2 * origin) + 1, 1)     # Define the bins for the histogram
    histograms = [None] * (tmax // time_step)               # Initialize the list for histograms

    # Calculate the histogram for each time step
    for t in range(0, tmax, time_step):
        bin_counts, _ = np.histogram(results_transposed[t], bins=num_bins)
        histograms[t] = bin_counts

    # Normalize the histograms to probabilities
    histograms_list = [arr.tolist() for arr in histograms]
    for t in range(0, tmax, time_step):
        total_count = np.sum(histograms_list[t])
        if total_count != 0:
            histograms_list[t] = np.divide(histograms_list[t], total_count)
        else:
            histograms_list[t] = np.zeros_like(histograms_list[t])

    # Convert the list of histograms to a NumPy array and transpose it
    histograms_array = np.copy(histograms_list).T

    return histograms_array


def calculate_jumpsize_distribution(x_matrix: np.ndarray, first_bin: int, last_bin: int, bin_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the distribution of jump sizes from position data.

    Args:
        x_matrix: 2D array of positions.
        first_bin: Lower bound of the histogram.
        last_bin: Upper bound of the histogram. If None, set to max value.
        bin_width: Width of histogram bins.

    Returns:
        Tuple of bin centers and corresponding distribution values.
    """

    data = np.diff(x_matrix, axis=1)
    data = data[~np.isnan(data)]
    points, distribution = calculate_distribution(data, first_bin, last_bin, bin_width)

    return points, distribution


def calculate_timejump_distribution(t_matrix : np.ndarray, last_bin: float = 1e5):
    """Calculate the distribution of times between jumps : tbj

    Args:
        t_matrix (list of lists): List of time steps for all trajectories.
        tmax (int): Maximum time value for the simulation.


    Returns:
        tuple: 
            - tbj_bins (np.ndarray): Array of bin edges for the time intervals.
            - tbj_distribution (np.ndarray): Normalized histogram of times between jumps.

    Notes:
        - The function computes time differences between consecutive jumps across all trajectories.
        - It returns a normalized histogram representing the distribution of these time differences.
        - If no data exists, the distribution is filled with zeros.
    """
    # Define bins
    tbj_bins = np.arange(0, last_bin + 1, 1)

    # Flatten t_matrix and compute time differences
    tbj_list = np.diff(np.concatenate(t_matrix))               # Differences between jumps

    # Create histogram
    tbj_distrib, _ = np.histogram(tbj_list, bins=tbj_bins)     # Compute histogram

    # Normalize the distribution
    if np.sum(tbj_distrib) != 0:
        tbj_distrib = tbj_distrib / np.sum(tbj_distrib)  
    else:
        tbj_distrib = np.zeros_like(tbj_distrib)

    # Return bin edges (excluding the last) and normalized distribution
    return tbj_bins[:-1], tbj_distrib


def calculate_fpt_matrix(t_matrix: np.ndarray, x_matrix: np.ndarray, tmax: int, t_bin: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the first passage time (FPT) density using binning.
    Uses a 2D ndarray with NaNs for padding to handle non-uniform trajectories.

    Args:
        t_matrix (np.ndarray): 2D array (nt, steps) of time values (with np.nan for missing values).
        x_matrix (np.ndarray): 2D array (nt, steps) of position values (same shape as t_matrix).
        tmax (int): Maximum time considered (clipped at this value).
        t_bin (int): Bin size for the position space.

    Returns:
        tuple:
            - fpt_results (np.ndarray): Matrix of normalized FPT densities per bin and time.
            - fpt_number (np.ndarray): Number of trajectories reaching each bin at least once.
    """

    # Replace all values beyond tmax or nan with np.nan (just in case)
    t_matrix = np.where(t_matrix > tmax, np.nan, t_matrix)

    # Determine binning on x
    valid_x = x_matrix[~np.isnan(x_matrix)]
    x_max = np.max(valid_x)
    n_bins = math.ceil(x_max / t_bin)
    fpt_matrix = np.zeros((tmax + 1, n_bins))
    nt = x_matrix.shape[0]

    # Translate all trajectories so they start at 0
    start_positions = x_matrix[:, 0][:, np.newaxis]
    translated_x = x_matrix - start_positions

    # Loop over trajectories
    for traj_x, traj_t in zip(translated_x, t_matrix):
        valid = ~np.isnan(traj_x) & ~np.isnan(traj_t)
        x_vals = traj_x[valid]
        t_vals = np.floor(traj_t[valid]).astype(int)
        t_vals = np.clip(t_vals, 0, tmax)

        for i in range(1, len(x_vals)):
            x_prev_bin = int(x_vals[i - 1] // t_bin)
            x_curr_bin = int(x_vals[i] // t_bin)
            t_idx = t_vals[i]
            if x_vals[i] != 0:
                fpt_matrix[t_idx, x_prev_bin:x_curr_bin] += 1

    # Count trajectories that never reached each bin
    fpt_number = np.sum(fpt_matrix, axis=0)
    not_fpt_number = nt - fpt_number
    fpt_matrix = np.vstack((fpt_matrix, not_fpt_number))

    # Normalize per bin (avoid division by 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        fpt_results = fpt_matrix / np.sum(fpt_matrix, axis=0, keepdims=True)
        fpt_results[:, np.sum(fpt_matrix, axis=0) == 0] = 0  # fill 0 if no trajectory reached

    return fpt_results, fpt_number


# def calculate_fpt_matrix(t_matrix: np.ndarray, x_matrix: np.ndarray, tmax: int, t_bin: int) -> tuple[np.ndarray, np.ndarray] :
#     """
#     Calculate the first passage time (FPT) density using bins to reduce memory usage.
#     Positions are grouped into bins of 'bin_size'.

#     Args:
#         t_matrix (np.ndarray): All times.
#         x_matrix (np.ndarray): All positions.
#         tmax (int): Time maximum.
#         bin_size (int): Bin size of time.
#         rf (int): Rounding Factor.


#     Returns:
#         tuple: 
#             - fpt_results (np.ndarray): Matrix of density of first pass times.
#             - fpt_number (np.ndarray): Number of trajectories that reached the positions.
#     """
    
#     x_max = np.max(np.concatenate(x_matrix))
#     n_bins = math.ceil(x_max / t_bin)
#     fpt_matrix = np.zeros((tmax + 1, n_bins))
#     nt = len(t_matrix)

#     translated_all_x = [[x - sublist[0] for x in sublist] for sublist in x_matrix]

#     # I : matrix
#     couples = np.concatenate([list(zip(x, [min(math.floor(ti), tmax) for ti in t])) for x, t in zip(translated_all_x, t_matrix)])

#     for _ in range(len(couples)):
#         if couples[_][0] != 0:
#             bin_index_start = couples[_ - 1][0] // t_bin
#             bin_index_end = couples[_][0] // t_bin
#             fpt_matrix[couples[_][1]][bin_index_start:bin_index_end ] += 1

#     # II : curve
#     fpt_number = np.sum(fpt_matrix, axis=0)
#     not_fpt_number = nt - fpt_number
#     fpt_matrix = np.vstack((fpt_matrix, not_fpt_number))
#     fpt_results = fpt_matrix / np.sum(fpt_matrix, axis=0)  # normalizing with the line of absent trajectories

#     return fpt_results, fpt_number


def calculate_instantaneous_statistics(
    t_matrix: np.ndarray, 
    x_matrix: np.ndarray, 
    nt: int, 
    first_bin: float = 0,
    last_bin: float = 1e5,
    bin_width: float = 1.0,
) -> Tuple[
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

    # Initialize arrays for displacements, time intervals, and speeds
    dx_array = np.array([None] * nt, dtype=object)
    dt_array = np.array([None] * nt, dtype=object)
    vi_array = np.array([None] * nt, dtype=object)

    # Loop through each trajectory
    for i in range(nt):
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

    # # Concatenate arrays for all trajectories
    # dx_array = np.concatenate(dx_array)
    # dt_array = np.concatenate(dt_array)
    # vi_array = np.concatenate(vi_array)

    # Concatenate all valid segments
    dx_array = np.concatenate([arr for arr in dx_array if arr is not None and len(arr) > 0])
    dt_array = np.concatenate([arr for arr in dt_array if arr is not None and len(arr) > 0])
    vi_array = np.concatenate([arr for arr in vi_array if arr is not None and len(arr) > 0])

    # Calculate distributions for Δx, Δt, and speeds
    dx_points, dx_distrib = calculate_distribution(dx_array, first_bin, last_bin, bin_width)
    dt_points, dt_distrib = calculate_distribution(dt_array, first_bin, last_bin, bin_width)
    vi_points, vi_distrib = calculate_distribution(vi_array, first_bin, last_bin, bin_width)

    # Compute statistics (mean, median, most probable values)
    if vi_distrib.size > 0:
        dx_mean = np.mean(dx_array)
        dx_med = np.median(dx_array)
        dx_mp = dx_points[np.argmax(dx_distrib)]

        dt_mean = np.mean(dt_array)
        dt_med = np.median(dt_array)
        dt_mp = dt_points[np.argmax(dt_distrib)]

        vi_mean = np.mean(vi_array)
        vi_med = np.median(vi_array)
        vi_mp = vi_points[np.argmax(vi_distrib)]

    # Default values if distributions are empty
    else:
        dx_mean, dx_med, dx_mp = 0.0, 0.0, 0.0
        dt_mean, dt_med, dt_mp = 0.0, 0.0, 0.0
        vi_mean, vi_med, vi_mp = 0.0, 0.0, 0.0

    # Return results
    return (
        dx_points, dx_distrib, dx_mean, dx_med, dx_mp,
        dt_points, dt_distrib, dt_mean, dt_med, dt_mp,
        vi_points, vi_distrib, vi_mean, vi_med, vi_mp
    )


# ================================================
# Part 2.6 : Fitting functions
# ================================================


def filtering_before_fit(
    time: np.ndarray, 
    data: np.ndarray, 
    std: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters the input arrays to remove invalid or problematic data points before fitting.

    Args:
        time (np.ndarray): Array of time values.
        data (np.ndarray): Array of data values.
        std (np.ndarray): Array of standard deviation values.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered time, data, and standard deviation arrays.
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


def fitting_in_two_steps(times, positions, deviations, bound_low=5, bound_high=80, epsilon=1e-10, rf=3):
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
        epsilon (float): Small value to avoid log(0).
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

    # Check if there's enough data for both bounds
    if len(positions) < max(bound_high, bound_low + 2):
        return None, None, None, None, None, None, None, None, None, None

    # Remove the first point to avoid (0, 0)
    times = times[1:]
    positions = positions[1:]

    # Step 1: linear average of x(t)/t over early time
    xt_over_t = np.divide(positions, times)
    array_low = xt_over_t[:bound_low]
    vf = np.mean(array_low)
    vf_std = np.std(array_low)

    # Step 2: logarithmic Derivative (G) to observe where the bound_high is - helps plots
    dlogx = np.diff(np.log(positions))
    dlogt = np.diff(np.log(times))
    G = np.divide(dlogx, dlogt)

    # Step 3: check if there are enough points for log-log fit
    if len(times) <= bound_high + 1:
        return np.round(vf, rf), None, None, np.round(vf_std, rf), None, None, None, None, None, None

    # Step 4: log-log fit of x(t) = Cf * t^wf on the right side
    log_t_high = np.log(times[bound_high:])
    log_x_high = np.log(np.maximum(positions[bound_high:], epsilon))
    slope, intercept, r_value, p_value, std_err_slope = linregress(log_t_high, log_x_high)

    # Fit results
    Cf = np.exp(intercept)
    wf = slope

    # Error estimates
    n = len(log_t_high)
    std_err_intercept = std_err_slope * np.sqrt(np.sum(log_t_high**2) / n)
    Cf_std = Cf * std_err_intercept
    wf_std = std_err_slope

    return (
        np.round(vf, rf), np.round(Cf, rf), np.round(wf, rf), 
        np.round(vf_std, rf), np.round(Cf_std, rf), np.round(wf_std, rf),
        xt_over_t, G, bound_low, bound_high
    )


# ================================================
# Part 2.7 : Writing functions
# ================================================


def set_working_environment(base_dir: str = Path.home() / "Documents" / "PhD" / "Workspace" / "nucleo" / "outputs", subfolder: str = "") -> None:
    """
    Ensure the specified folder exists and change the current working directory to it.
        Check if the folder exists; if not, create it
        Change the current working directory to the specified folder

    Args:
        folder_path (str): Path to the folder where the working environment should be set.

    Returns:
        None.
    """
    root = os.getcwd()
    full_path = os.path.join(root, base_dir, subfolder)
    
    os.makedirs(full_path, exist_ok=True)
    os.chdir(full_path)

    return full_path


def prepare_value(value):
    """
    Convert various data types to Parquet-compatible formats, including deep handling of NaNs.
    Do not write in scientific number because it would become string and use more memory.

    Args:
        value: The value to be converted.

    Returns:
        The converted value in a compatible format.

    Raises:
        ValueError: If the data type is unsupported.
    """
    # Convert NumPy matrix or array to list
    if isinstance(value, (np.ndarray, np.matrix)):
        return [prepare_value(v) for v in np.array(value).tolist()]

    # Convert NumPy scalars to native scalars
    elif isinstance(value, (np.integer, np.floating)):
        if np.isnan(value):
            return None
        return value.item()

    # Handle float NaN explicitly
    elif isinstance(value, float) and np.isnan(value):
        return None

    # Convert list recursively
    elif isinstance(value, list):
        return [prepare_value(v) for v in value]

    # Scalars and strings
    elif isinstance(value, (int, float, str)):
        return value

    # Optional: allow None to pass through
    elif value is None:
        return None

    else:
        raise ValueError(f"Unsupported data type: {type(value)}")


def writing_parquet(file:str, title: str, data_result: dict, data_info = False) -> None:
    """
    Write a dictionary directly into a Parquet file using PyArrow.
    Ensures that all numerical values, arrays, and lists are properly handled.

    Note:
        - Each key in the Parquet file must correspond to a list or an array.
        - Compatible only with native Python types.
        - Even a number like 1.797e308 only takes up 8 bytes (64 bits) in the Parquet file.

    Args:
        title (str): The base name for the Parquet file and folder.
        data_result (dict): Dictionary containing the data to write. 
            Supported types for values:
                - NumPy arrays (converted to lists).
                - NumPy matrices (converted to lists).
                - NumPy scalars (converted to native Python types).
                - Python scalars (int, float).
                - Lists (unchanged).
                - Strings (unchanged).

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If a value in the dictionary has an unsupported data type.
        Exception: If writing to the Parquet file fails for any reason.
    """

    # If you need to see the details od the data registered
    if data_info :
        inspect_data_types(data_result)

    # Define the Parquet file path
    data_file_name = os.path.join(title, f'{file}_{title}.parquet')
    os.makedirs(title, exist_ok=True)

    # Prepare the data for Parquet
    prepared_data = {key: [prepare_value(value)] if not isinstance(value, list) else prepare_value(value)
                     for key, value in data_result.items()}

    try:        
        table = pa.table(prepared_data)                                         # Create a PyArrow Table from the dictionary
        pq.write_table(table, data_file_name, compression='gzip')               # Write the table to a Parquet file

    except Exception as e:
        print(f"Failed to write Parquet file due to: {e}")
    
    return None


def inspect_data_types(data: dict, launch = True) -> None:
    """
    Inspect and print the types and dimensions of values in a dictionary.

    Args:
        data (dict): Dictionary containing the data to inspect. 
            Keys are expected to be strings, and values can be of various types, such as:
            - NumPy arrays (prints dimensions).
            - Lists (prints length).
            - Other types (prints the type of the value).

    Returns:
        None
    """
    if launch:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"Key: {key}, Dimensions: {value.shape}")     # Check if the value is a NumPy array
            elif isinstance(value, list):
                print(f"Key: {key}, Length: {len(value)}")          # Check if the value is a list
            else:
                print(f"Key: {key}, Type: {type(value)}")           # Other types
    return None


# ================================================
# Part 2.8 : Test functions
# ================================================


def HiC_map_generation(data: np.ndarray) -> np.ndarray:
    """
    Generates a normalized Hi-C contact map with a fixed size of 10,000 x 10,000.
    Positions greater than or equal to 10,000 are excluded before processing.

    Parameters:
    -----------
    data : np.ndarray
        A 2D NumPy array where each row represents a sequence of genomic positions.

    Returns:
    --------
    np.ndarray
        A normalized 2D NumPy array (Hi-C contact map) of size (10,000 x 10,000), 
        where values sum to 1, representing interaction densities.

    Notes:
    ------
    - The function first filters out values that are greater than or equal to 10,000.
    - It creates a zero-initialized matrix of shape (10,000 x 10,000).
    - It iterates over each row of `data` to extract consecutive position pairs.
    - Each valid (x1, x2) pair is used to increment the corresponding position in `HiC_map`.
    - Finally, the Hi-C map is normalized so that its sum equals 1.
    """

    size = 10000  # Fixed size of the Hi-C map
    HiC_map = np.zeros((size, size), dtype=np.float32)  # Use float32 for memory efficiency

    for array_x in data:
        # Filter out values >= 10000 before processing
        array_x = np.array(array_x)
        array_x = array_x[array_x < size]

        if len(array_x) > 1:  # Ensure there are at least two values to form a contact
            contacts = np.stack((array_x[:-1], array_x[1:]), axis=1)
            x1, x2 = contacts.T.astype(int)  # Convert to integer indices
            HiC_map[x1, x2] += 1  # Increment contact counts

    # Normalize to ensure the sum equals 1
    HiC_map_normed = HiC_map / HiC_map.sum() if HiC_map.sum() > 0 else HiC_map

    return HiC_map_normed


# ================================================
# Part 2.9 : Dynamic analysis functions
# ================================================


def find_jumps(x_matrix: np.ndarray, t_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Identifies forward and reverse jump times from position and time matrices.

    Args:
        x_matrix: 2D array of positions.
        t_matrix: 2D array of cumulative times.

    Returns:
        Tuple containing flattened arrays of:
            - forward bind times
            - forward rest times
            - reverse bind times
            - reverse rest times
    """
    
    # Getting all the times (non cumulated) from the t_matrix
    time = np.diff(t_matrix, axis=1)

    # Initilializaition and filtering where x[i][j] == x[i][j+1]
    frwd_mask = np.zeros_like(x_matrix, dtype=bool)
    equal_next = (x_matrix[:, :-1] == x_matrix[:, 1:])

    # Transmit the True from bind_time to corresponding rest_time
    frwd_mask[:, :-1] |= equal_next
    frwd_mask[:,  1:] |= equal_next
    frwd_mask = np.copy(frwd_mask[:, 1:])
    rvrs_mask = ~ frwd_mask
    x_matrix = np.copy(x_matrix[:, 1:])    
    
    # Forward : select the columns corresponding to bind (odd) and rest (even)
    frwd_time = frwd_mask * time
    frwd_time[frwd_time==0] = np.nan  
    frwd_bind = frwd_time[:, 0::2]
    frwd_rest = frwd_time[:, 1::2]

    # Reverse : select the columns corresponding to bind (odd) and rest (even)
    rvrs_time = rvrs_mask * time
    rvrs_time[rvrs_time==0] = np.nan  
    rvrs_bind = rvrs_time[:, 0::2]
    rvrs_rest = rvrs_time[:, 1::2]
    
    # print(x_matrix, "\n\n", time, "\n\n", frwd_bind, "\n\n", frwd_rest, "\n\n", rvrs_bind, "\n\n", rvrs_rest)
    return (np.concatenate(frwd_bind),
            np.concatenate(frwd_rest),
            np.concatenate(rvrs_bind), 
            np.concatenate(rvrs_rest))
    
    
def calculate_nature_jump_distribution(t_matrix: np.ndarray,
                                       x_matrix: np.ndarray,
                                       first_bin: int, 
                                       last_bin: int,
                                       bin_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the binned distributions of forward and reverse bind/rest times.

    Args:
        t_matrix: 2D array of cumulative times.
        x_matrix: 2D array of positions.
        first_bin: Lower bound of histogram bins.
        last_bin: Upper bound of histogram bins.
        bin_width: Width of histogram bins.

    Returns:
        Tuple of distributions:
            - fb : forward bind times
            - fr : forward rest times
            - rb : reverse bind times
            - rr : reverse rest times
   
    """

    # Get the datas
    fb_array, fr_array, rb_array, rr_array = find_jumps(x_matrix, t_matrix)
    
    # Get the distributions of datas
    _, fb_y = calculate_distribution(data=fb_array, first_bin=first_bin, last_bin=last_bin, bin_width=bin_width)
    _, fr_y = calculate_distribution(data=fr_array, first_bin=first_bin, last_bin=last_bin, bin_width=bin_width)
    _, rb_y = calculate_distribution(data=rb_array, first_bin=first_bin, last_bin=last_bin, bin_width=bin_width)
    _, rr_y = calculate_distribution(data=rr_array, first_bin=first_bin, last_bin=last_bin, bin_width=bin_width)
    
    return fb_y, fr_y, rb_y, rr_y


def extracting_taus(
    fb_y: np.ndarray, 
    fr_y: np.ndarray, 
    rb_y: np.ndarray, 
    rr_y: np.ndarray, 
    array: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Fits exponential decay to the given distributions and extracts decay constants and initial values.

    Args:
        fb_y: Forward bind distribution.
        fr_y: Forward rest distribution.
        rb_y: Reverse bind distribution.
        rr_y: Reverse rest distribution.
        array: Bin centers or time points.

    Returns:
        Tuple of decay constants and initial values for all four distributions.
    """

    y0_fb, tau_fb = curve_fit(exp_decay, array, fb_y, p0=(fb_y[0], 1.0))[0]
    y0_fr, tau_fr = curve_fit(exp_decay, array, fr_y, p0=(fr_y[0], 1.0))[0]
    y0_rb, tau_rb = curve_fit(exp_decay, array, rb_y, p0=(rb_y[0], 1.0))[0]
    y0_rr, tau_rr = curve_fit(exp_decay, array, rr_y, p0=(rr_y[0], 1.0))[0]

    return tau_fb, tau_fr, tau_rb, tau_rr


def calculating_rates(tau_fb, tau_fr, tau_rb, tau_rr):
    """
    Calculate fitted binding and resting rates based on times.
    So not on dweel times !

    Parameters:
        tau_fb (float): Mean forward binding time.
        tau_fr (float): Mean forward resting time.
        tau_rb (float): Mean reverse binding time.
        tau_rr (float): Mean reverse resting time.

    Returns:
        tuple: 
            rtot_bind_fit (float): Fitted total binding rate.
            rtot_rest_fit (float): Fitted total resting rate.
    """
    rtot_bind_fit = ((tau_fb + tau_rb) / 2) ** -1
    rtot_rest_fit = ((tau_fr + tau_rr) / 2) ** -1

    return rtot_bind_fit, rtot_rest_fit


def getting_forwards(
    t_matrix: np.ndarray, 
    x_matrix: np.ndarray, 
    first_bin: int, 
    last_bin: int, 
    bin_width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the distribution of forward times based on position and time matrices.

    Args:
        t_matrix: 2D array of cumulative times.
        x_matrix: 2D array of positions.
        first_bin: Lower bound of histogram bins.
        last_bin: Upper bound of histogram bins.
        bin_width: Width of histogram bins.

    Returns:
        Tuple of bin centers and forward time distribution.
    """

    # Get the datas
    mask = np.zeros_like(x_matrix, dtype=bool)
    matches = (x_matrix[:, :-1] == x_matrix[:, 1:])
    mask[:, 1:] = matches

    array = mask * t_matrix
    result = np.concatenate([
        np.insert(row[(row != 0 ) & ~np.isnan(row)], 0, 0)
        for row in array
    ])

    diff = np.diff(result)
    frwd_times = diff[diff > 0]

    points, distrib_forwards = calculate_distribution(frwd_times, first_bin, last_bin, bin_width)
    return points, distrib_forwards


def getting_reverses(
    t_matrix: np.ndarray, 
    x_matrix: np.ndarray, 
    first_bin: int, 
    last_bin: int, 
    bin_width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the distribution of reverse dwell times from position and time matrices.

    Args:
        t_matrix: 2D array of cumulative times.
        x_matrix: 2D array of positions.
        first_bin: Lower bound of histogram bins.
        last_bin: Upper bound of histogram bins.
        bin_width: Width of histogram bins.

    Returns:
        Tuple of bin centers and reverse dwell time distribution.
    """

    # Proper times
    times = t_matrix[:, 0::2]

    mask = np.zeros_like(x_matrix, dtype=bool)
    matches = (x_matrix[:, :-1] == x_matrix[:, 1:])
    mask[:, 1:] = matches
    filter = mask[:, 0::2]

    dwell = []

    for i in range (len(filter)):
        for j in range(len(filter[0])):
            if filter[i][j] == False:
                false_value = times[i][j]
            if filter[i][j] == True:
                dwell.append(times[i][j] - false_value)

    points, distrib_reverses = calculate_distribution(np.array(dwell), first_bin, last_bin, bin_width)
    return points, distrib_reverses


def calculate_dwell_distribution(t_matrix: list, x_matrix: list, first_bin: float, last_bin: float, bin_width: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the dwell time distributions for forward and reverse events based on time and position matrices.

    Args:
        t_matrix (list of list): Time values. Each sublist corresponds to a trajectory.
        x_matrix (list of list): Position values. Each sublist corresponds to a trajectory.
        first_bin: Lower bound of histogram bins.
        last_bin: Upper bound of histogram bins.
        bin_width: Width of histogram bins.

    Returns:
        Tuple of bin centers and forward time distribution.
            - t_points (np.ndarray): points of the distributions.
            - forward_result (np.ndarray): forward dwell time distribution.
            - reverse_result (np.ndarray): reverse dwell time distribution.
    
    Notes
    -----
    - A "dwell" corresponds to a time interval between two steps.
    - Forward dwells are detected when two consecutive jumps go forward.
    - Reverse dwells are detected when a forward jump is followed by a reverse.
    - Nan values in the input are safely masked and ignored.
    - Zero-duration dwell times are excluded from the final distributions.

    Notations
    -----
    - e for event
    - d for dwell
    - e_forwards : True = Forward & False = Reverse & -- = nan
    """

    # Getting the datas in the proper format    
    t = np.diff(t_matrix, axis=1)
    x = x_matrix

    # Filtering on the x positions : did it progress along chromatin or not ?
    x_pair = x[:, 0::2]
    x_mask = np.ma.masked_invalid(x_pair)
    e_forwards = x_mask[:, :-1] < x_mask[:, 1:]

    # Filtering on the events to get the dwells : 
    d_forwards = (e_forwards[:, :-1] == True) & (e_forwards[:, 1:] == True)     # was there a forward jump then a forward jump ?
    d_reverses = (e_forwards[:, :-1] == True) & (e_forwards[:, 1:] == False)    # was there a forward jump then a reverse jump ?

    # Calculating time associated by grouping them per 2 because of our formalism : bind + rest
    t_event = np.add(t[:, ::2], t[:, 1::2])
    t_forwards = d_forwards * t_event[:, :-1]
    t_reverses = d_reverses * t_event[:, :-1]

    # Filtering the results to remove the 0.0 and --
    t_forwards_filtered = t_forwards[t_forwards != 0.0].compressed()
    t_reverses_filtered = t_reverses[t_reverses != 0.0].compressed()

    # Calculating the distributions of all extracted times
    dwell_points = np.arange(first_bin, last_bin, bin_width)
    _, forward_result = calculate_distribution(t_forwards_filtered, first_bin, last_bin, bin_width)
    _, reverse_result = calculate_distribution(t_reverses_filtered, first_bin, last_bin, bin_width)

    return dwell_points, forward_result, reverse_result


def calculate_dwell_times(
    points: np.ndarray, 
    distrib_forwards: np.ndarray, 
    distrib_reverses: np.ndarray,
    xmax: float = None
):
    """
    Fits exponential decay models separately to the forward and reverse distributions,
    automatically choosing the region beyond the distribution maximum.

    Args:
        points: Bin centers or time points.
        distrib_forwards: Forward time distribution.
        distrib_reverses: Reverse time distribution.
        xmax: Optional maximum bound for fitting.

    Returns:
        Decay constants and initial values for forward and reverse fits.
    """

    # Condition on empty arrays
    if len(distrib_forwards) == 0 or len(distrib_reverses) == 0:
        tau_forwards, tau_reverses = np.nan, np.nan
        return tau_forwards, tau_reverses
    
    # Determine automatic xmin for each distribution (after its peak)
    else:
        xmin_forward = points[np.argmax(distrib_forwards)]
        xmin_reverse = points[np.argmax(distrib_reverses)]

    # Apply filtering per distribution
    mask_forward = (points >= xmin_forward)
    mask_reverse = (points >= xmin_reverse)

    if xmax is not None:
        mask_forward &= (points <= xmax)
        mask_reverse &= (points <= xmax)

    # Filtered data
    x_fit_fwd = points[mask_forward]
    y_fit_fwd = distrib_forwards[mask_forward]

    x_fit_rev = points[mask_reverse]
    y_fit_rev = distrib_reverses[mask_reverse]

    # Check for too few points
    if len(x_fit_fwd) < 2 or len(x_fit_rev) < 2:
        raise ValueError("Not enough data points in fitting range. Adjust bins or range.")

    # p0 guess: amplitude ~ first value, tau ~ 10
    p0_fwd = (y_fit_fwd[0], 10.0)
    p0_rev = (y_fit_rev[0], 10.0)

    # Fitting
    def safe_fit(x, y, p0):
        try:
            return curve_fit(exp_decay, x, y, p0=p0)[0]
        except:
            return np.nan, np.nan

    # Call
    y0_forwards, tau_forwards = safe_fit(x_fit_fwd, y_fit_fwd, p0_fwd)
    y0_reverses, tau_reverses = safe_fit(x_fit_rev, y_fit_rev, p0_rev)

    return tau_forwards, tau_reverses 


# ================================================
# Part 3.1 : Main function
# ================================================


def sw_nucleo(
    alpha_choice: str, s: int, l: int, bpmin: int, 
    mu: float, theta: float, lmbda: float, alphao: float, alphaf: float, beta: float, 
    rtot_bind: float, rtot_rest: float,
    nt: int, path: str,
    Lmin: int, Lmax: int, bps: int, origin: int,
    tmax: float, dt: float, 
    algorithm_choice = "two_steps",
    saving = "data"
    ) -> None:
    """
    Simulates condensin dynamics along chromatin with specified parameters.

    Args:
        alpha_choice (str): Choice of the alpha configuration ('ntrandom', 'periodic', 'constantmean').
        s (int): Nucleosome size.
        l (int): Linker length.
        bpmin (int): Minimum base pair threshold.
        mu (float): Mean value for the distribution used in the simulation.
        theta (float): Standard deviation for the distribution used in the simulation.
        lmbda (float): Lambda parameter for the simulation.
        alphao (float): Acceptance probability on nucleosome sites.
        alphaf (float): Acceptance probability on linker sites.
        beta (float): Unfolding probability.
        rtot_bind (float): Reaction rate for binding (inverse of characteristic time).
        rtot_rest (float): Reaction rate for resting (inverse of characteristic time).
        nt (int): Number of trajectories to simulate.
        path (str): Output path for saving results.
        Lmin (int): First chromatin position.
        Lmax (int): Last chromatin position.
        bps (int): Base pairs per site.
        origin (int): Starting position for the simulation.
        tmax (float): Maximum simulation time.
        dt (float): Time step increment.
        algorithm_choice (str): Choice of algorithm for the modeling.
        saving (bool): Whether to save the results and in which kind.
    Returns:
        None: This function does not return any value. It performs a simulation and saves results in a file.

    Note:
        - The function assumes that all inputs are valid and within the expected range.
        - This function is a core part of the nucleosome simulation pipeline.
    """

    # ------------------- Initialization ------------------- #

    # Title & Folder    
    title = (
            f"alphachoice={alpha_choice}_s={s}_l={l}_bpmin={bpmin}_"
            f"mu={mu}_theta={theta}_"
            # f"lmbda={lmbda:.2e}_rtotbind={rtot_bind:.2e}_rtotrest={rtot_rest:.2e}_"
            f"nt={nt}"
            )

    # Chromatin
    L = np.arange(Lmin, Lmax, bps)
    lenght = (Lmax-Lmin) // bps

    # Time 
    times = np.arange(0,tmax,dt)    # Discretisation of all times
    bin_fpt = int(1e+1)             # Bins on times during the all analysis

    # Linear factor
    alpha_0 = int(1e+0)             # Calibration on linear speed in order to multiplicate speeds by a linear number

    # Bins for Positions and Times : fb (firstbin) - lb (lastbin) - bw (binwidth)
    x_fb, x_lb, x_bw = 0, 10_000, 1
    t_fb, t_lb, t_bw = 0, 100, 0.20
    x_bins = np.arange(x_fb, x_lb, x_bw)
    t_bins = np.arange(t_fb, t_lb, t_bw)

    # ------------------- Simulation ------------------- #

    # Chromatin : Landscape + Obstacles and Linkers
    alpha_matrix, alpha_mean = alpha_matrix_calculation(
        alpha_choice, s, l, bpmin, alphao, alphaf, Lmin, Lmax, bps, nt
    )
    obs_points, obs_distrib, link_points, link_distrib = calculate_obs_and_linker_distribution(
        alpha_matrix[0], alphao, alphaf
    )
    link_view = calculate_linker_landscape(
        alpha_matrix, alpha_choice, nt, alphaf, Lmin, Lmax
    )

    # Probabilities
    p = proba_gamma(mu, theta, L)

    # Gillespie simulation
    if algorithm_choice == "one_step":
        results, t_matrix, x_matrix = gillespie_algorithm_one_step(
            nt, tmax, dt, alpha_matrix, beta, Lmax, lenght, origin, p
        )
    elif algorithm_choice == "two_steps":
        results, t_matrix, x_matrix = gillespie_algorithm_two_steps(
            alpha_matrix, p, beta, lmbda, rtot_bind, rtot_rest, nt, tmax, dt, L, origin
        )
    else:
        raise ValueError("Invalid algorithm choice")   

    # Clean datas
    x_matrix = listoflist_into_matrix(x_matrix)
    t_matrix = listoflist_into_matrix(t_matrix)

    # ------------------- Analysis 1 : General results + Jump size + Time size + First pass times ------------------- #

    # General results
    results_mean, results_med, results_std, v_mean, v_med = calculate_main_results(
        results, dt, alpha_0, nt
    )
    vf, Cf, wf, vf_std, Cf_std, wf_std, xt_over_t, G, bound_low, bound_high = fitting_in_two_steps(
        times, results_mean, results_std
    )

    # Jump size distribution
    xbj_points, xbj_distrib = calculate_jumpsize_distribution(
        x_matrix, x_fb, x_lb, x_bw
    )

    # Time size distribution
    tbj_points, tbj_distrib = calculate_timejump_distribution(t_matrix)

    # First pass times
    fpt_distrib_2D, fpt_number = calculate_fpt_matrix(t_matrix, x_matrix, tmax, bin_fpt)

    # ------------------- Analysis 2 : Speeds + Rates ------------------- #

    # Instantaneous speeds
    dx_points, dx_distrib, dx_mean, dx_med, dx_mp, \
    dt_points, dt_distrib, dt_mean, dt_med, dt_mp, \
    vi_points, vi_distrib, vi_mean, vi_med, vi_mp = calculate_instantaneous_statistics(
        t_matrix, x_matrix, nt
    )

    # Rates and Taus
    dwell_points, forward_result, reverse_result = calculate_dwell_distribution(
        t_matrix, x_matrix, t_fb, t_lb, t_bw
    )
    tau_forwards, tau_reverses = calculate_dwell_times(
        dwell_points, distrib_forwards=forward_result, distrib_reverses=reverse_result, xmax=100
    )

    # ------------------- Working area ------------------- #

    # Currently studying speeds in order to verify  
    # The definition is maybe not good + we are not using here what we found eralier on dweel times
    v_th = theoretical_speed(alphaf, alphao, s, l, mu, lmbda, rtot_bind, rtot_rest)
    fb_y, fr_y, rb_y, rr_y = calculate_nature_jump_distribution(t_matrix, x_matrix, t_fb, t_lb, t_bw)
    tau_fb, tau_fr, tau_rb, tau_rr = extracting_taus(fb_y, fr_y, rb_y, rr_y, t_bins)
    rtot_bind_fit, rtot_rest_fit = calculating_rates(tau_fb, tau_fr, tau_rb, tau_rr)
    v_fit = theoretical_speed(alphaf, alphao, s, l, mu, lmbda, rtot_bind_fit, rtot_rest_fit)

    # # Nucleosomic profile close to : "Determinants of nucleosome organization in primary human cells"
    # plt.figure(figsize=(8,6))
    # plt.plot(link_view, label="link_view")
    # plt.grid(True, which="both")
    # plt.legend()
    # plt.show()


    # ------------------- Writing ------------------- #

    if saving == "data":
        data_result = {
            # --- Principal Parameters --- #
            'alpha_choice'   : alpha_choice,
            's'              : s,
            'l'              : l,
            'bpmin'          : bpmin,
            'mu'             : mu,
            'theta'          : theta,
            'alphao'         : alphao,
            'alphaf'         : alphaf,
            'beta'           : beta,
            'lmbda'          : lmbda,
            'rtot_bind'      : rtot_bind,
            'rtot_rest'      : rtot_rest,

            # --- Chromatin Parameters --- #
            'Lmin'           : Lmin,
            'Lmax'           : Lmax,
            'bps'            : bps,
            'origin'         : origin,

            # --- Time Parameters --- #
            'tmax'           : tmax,
            'dt'             : dt,
            'nt'             : nt,

            # --- Chromatin --- #
            'alpha_mean'     : alpha_mean,
            'obs_points'     : obs_points,
            'obs_distrib'    : obs_distrib,
            'link_points'    : link_points,
            'link_distrib'   : link_distrib,
            'link_view'      : link_view,

            # --- Results --- #
            'results'        : results,
            'results_mean'   : results_mean,
            'results_med'    : results_med,
            'results_std'    : results_std,
            'v_mean'         : v_mean,
            'v_med'          : v_med,
            'vf'             : vf,
            'Cf'             : Cf,
            'wf'             : wf,
            'vf_std'         : vf_std,
            'Cf_std'         : Cf_std,
            'wf_std'         : wf_std,

            # --- Between Jumps --- #
            'xbj_points'     : xbj_points,
            'xbj_distrib'    : xbj_distrib,
            'tbj_points'     : tbj_points,
            'tbj_distrib'    : tbj_distrib,

            # --- First Passage Time --- #
            'bin_fpt'        : bin_fpt,
            'fpt_distrib_2D' : fpt_distrib_2D,
            'fpt_number'     : fpt_number,

            # --- Instantaneous statistics --- #
            'dx_points'      : dx_points,
            'dx_distrib'     : dx_distrib,
            'dx_mean'        : dx_mean,
            'dx_med'         : dx_med,
            'dx_mp'          : dx_mp,
            'dt_points'      : dt_points,
            'dt_distrib'     : dt_distrib,
            'dt_mean'        : dt_mean,
            'dt_med'         : dt_med,
            'dt_mp'          : dt_mp,
            'vi_points'      : vi_points,
            'vi_distrib'     : vi_distrib,
            'vi_mean'        : vi_mean,
            'vi_med'         : vi_med,
            'vi_mp'          : vi_mp,

            # --- Fits --- #
            'alpha_0'        : alpha_0,
            'xt_over_t'      : xt_over_t,
            'G'              : G,
            'bound_low'      : bound_low,
            'bound_high'     : bound_high,
        }

    elif saving == "map":
        data_result = {
            # --- Principal Parameters --- #
            'alpha_choice'   : alpha_choice,
            's'              : s,
            'l'              : l,
            'bpmin'          : bpmin,
            'mu'             : mu,
            'theta'          : theta,
            'alphao'         : alphao,
            'alphaf'         : alphaf,
            'beta'           : beta,
            'lmbda'          : lmbda,
            'rtot_bind'      : rtot_bind,
            'rtot_rest'      : rtot_rest,

            # --- Chromatin Parameters --- #
            'Lmin'           : Lmin,
            'Lmax'           : Lmax,
            'bps'            : bps,
            'origin'         : origin,

            # --- Time Parameters --- #
            'tmax'           : tmax,
            'dt'             : dt,
            'nt'             : nt,

            # --- Speeds and Taus --- #
            'v_mean'         : v_mean,
            'v_th'           : v_th,
            'v_fit'          : v_fit,
            'tau_forwards'   : tau_forwards,
            'tau_reverses'   : tau_reverses,
        }


    # Types of data registered if needed
    inspect_data_types(data_result, launch=False)

# Writing data
    writing_parquet(file=path, title=title, data_result=data_result)

    # Clean raw datas
    del alpha_matrix
    del data_result
    gc.collect()

    return None


# ================================================
# Part 3.2 : Launching functions
# ================================================


def checking_inputs(
    alpha_choice, s, l, bpmin, 
    mu, theta, lmbda, alphao, alphaf, beta,
    nt,
    Lmin, Lmax, bps, origin,
    tmax, dt
):
    """
    Checks the validity of input parameters for the simulation.

    Parameters:
    - s (int): Nucleosome lenght (must be 150).
    - l (int): Linker DNA length (must be ≤ s).
    - bpmin (int): Minimum base pair value to bind (must be ≤ 10).
    - alphao (float): Obstacle alpha parameter (must be in [0, 1]).
    - alphaf (float): Linker alpha parameter (must be in [0, 1]).
    - alphar (float): FACT alpha parameter (must be in [0, 1]).
    - Lmin (int): Minimum condensin position (must be 0).
    - Lmax (int): Maximum condensin position (must be > Lmin).
    - bps (int): Base pair spacing step (must be > 0).
    - L (np.ndarray): 1D array of condensin positions from Lmin to Lmax.
    - nt (int): Number of trajectories (must be > 0).
    - mu (float): Mean jump length (must be > 0).
    - theta (float): Spread or jump lenght (must be ≥ 0).
    - tmax (int): Maximum simulation time (must be > 0).
    - dt (float): Time resolution step (must be > 0).
    - origin (int): Starting index of the simulation (must be in [0, Lmax)).
    - alpha_choice (str): Mode for alpha distribution (must be one of {"constantmean", "periodic", "ntrandom"}).

    Raises:
    - ValueError: If any of the parameter constraints are violated.
    """

    # Obstacles
    if alpha_choice not in {"constantmean", "periodic", "ntrandom"}:
        raise ValueError(f"Invalid alpha_choice: {alpha_choice}. Must be 'constantmean', 'periodic', or 'ntrandom'.")
    for name, value in [("s", s), ("l", l), ("bpmin", bpmin)]:
        if not isinstance(value, np.integer) or value < 0:
            raise ValueError(f"Invalid value for {name}: must be an int >= 0. Got {value}.")

    # Probabilities
    if not isinstance(mu, np.integer) or mu < 0:
        raise ValueError(f"Invalid value for mu: must be an int >= 0. Got {mu}.")
    if not isinstance(theta, np.integer) or theta < 0:
        raise ValueError(f"Invalid value for theta: must be an int >= 0. Got {theta}.")
    for name, value in zip(["lmbda", "alphao", "alphaf", "beta"], [lmbda, alphao, alphaf, beta]):
        if not (0 <= value <= 1):
            raise ValueError(f"{name} must be between 0 and 1. Got {value}.")

    # Chromatin
    if Lmin != 0:
        raise ValueError(f"Lmin must be 0. Got {Lmin}.")
    if Lmax <= Lmin:
        raise ValueError(f"Lmax must be greater than Lmin. Got Lmax={Lmax}, Lmin={Lmin}.")
    if not isinstance(bps, int) or bps < 0:
        raise ValueError(f"Invalid value for bps: must be an int >= 0. Got {bps}.")
    if not (0 <= origin < Lmax):
        raise ValueError(f"origin must be within [0, Lmax). Got origin={origin}, Lmax={Lmax}.")
    
    # Trajectories
    if not isinstance(nt, int) or nt < 0:
        raise ValueError(f"Invalid value for nt: must be an int >= 0. Got {nt}.")

    # Times
    if not isinstance(tmax, int) or tmax < 0:
        raise ValueError(f"Invalid value for tmax: must be an int >= 0. Got {tmax}.")
    if dt <= 0:
        raise ValueError(f"dt must be positive. Got {dt}.")
    

def process_function(params: dict, chromatin: dict, time: dict) -> None:
    """
    Executes one simulation with the given parameters and shared constants.
    
    Args:
        params (dict): One combination of geometry + probas + rates + meta parameters.
        chromatin (dict): Dict with Lmin, Lmax, bps, origin.
        time (dict): Dict with tmax, dt.
    """
    checking_inputs(
        alpha_choice=params['alpha_choice'],
        s=params['s'],
        l=params['l'],
        bpmin=params['bpmin'],
        mu=params['mu'],
        theta=params['theta'],
        lmbda=params['lmbda'],
        alphao=params['alphao'],
        alphaf=params['alphaf'],
        beta=params['beta'],
        nt=params['nt'],
        Lmin=chromatin["Lmin"],
        Lmax=chromatin["Lmax"],
        bps=chromatin["bps"],
        origin=chromatin["origin"],
        tmax=time["tmax"],
        dt=time["dt"]
    )

    sw_nucleo(
        params['alpha_choice'],
        params['s'], params['l'], params['bpmin'],
        params['mu'], params['theta'],
        params['lmbda'], params['alphao'], params['alphaf'], params['beta'],
        params['rtot_bind'], params['rtot_rest'],
        params['nt'], params['path'],
        chromatin["Lmin"], chromatin["Lmax"], chromatin["bps"], chromatin["origin"],
        time["tmax"], time["dt"]
    )


# ================================================
# Part 3.3 : Multiprocessing functions
# ================================================


def choose_configuration(config: str) -> dict:
    """
    Returns a dictionary of study parameters organized in logical blocks.
    All list-like parameters are converted to np.array.
    """

    # ──────────────────────────────────
    # Shared constants (used everywhere)
    # ──────────────────────────────────

    CHROMATIN = {
        "Lmin": 0,          # First point of chromatin (included !)
        "Lmax": 50_000,     # Last point of chromatin (excluded !)
        "bps": 1,           # Based pair step 1 per 1
        "origin": 10_000    # Falling point of condensin on chromatin 
    }

    TIME = {
        "tmax": 100,        # Total time of modeling : 0 is taken into account
        "dt": 1             # Step of time
    }

    PROBAS = {
        "lmbda": 0.40,      # Probability of in vitro condensin to reverse
        "alphao": 0.00,     # Probability of binding if obstacle
        "alphaf": 1.00,     # Probability of binding if linker
        "beta": 0.00,       # Probability of in vitro condensin to undinb
    }

    RATES = {
        "rtot_bind": 1/6,   # Rate of binding
        "rtot_rest": 1/6    # Rate of resting
    }

    # ──────────────────────────────────
    # Presets for study configurations
    # ──────────────────────────────────

    presets = {

        "NU": {
            "geometry": {
                "alpha_choice": np.array(['ntrandom', 'periodic', 'constantmean']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.arange(100, 605, 5),
                "theta": np.arange(1, 101, 1),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": np.array([RATES["rtot_bind"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float)
            },
            "meta": {
                "nt": 100,
                "path": "ncl_nu_test"
            }
        },

        "BP": {
            "geometry": {
                "alpha_choice": np.array(['ntrandom']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([5, 10, 15], dtype=int)
            },
            "probas": {
                "mu": np.arange(100, 605, 5),
                "theta": np.arange(1, 101, 1),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": np.array([RATES["rtot_bind"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": "ncl_bp"
            }
        },

        "LSLOW": {
            "geometry": {
                "alpha_choice": np.array(['ntrandom']),
                "s": np.array([150], dtype=int),
                "l": np.array([5, 15, 20, 25], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.arange(100, 605, 5),
                "theta": np.arange(1, 101, 1),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": np.array([RATES["rtot_bind"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": "ncl_lslow"
            }
        },

        "LSHIGH": {
            "geometry": {
                "alpha_choice": np.array(['ntrandom']),
                "s": np.array([150], dtype=int),
                "l": np.array([50, 100, 150], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.arange(100, 605, 5),
                "theta": np.arange(1, 101, 1),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": np.array([RATES["rtot_bind"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float)
            },
            "meta": {
                "nt": 10_000,
                "path": "ncl_lshigh"
            }
        },

        "TEST": {
            "geometry": {
                "alpha_choice": np.array(['ntrandom']),
                "s": np.array([150], dtype=int),
                "l": np.array([10, 30, 50], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.array([300]),
                "theta": np.array([50]),
                "lmbda": np.array([PROBAS["lmbda"]], dtype=float),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": np.array([RATES["rtot_bind"]], dtype=float),
                "rtot_rest": np.array([RATES["rtot_rest"]], dtype=float)
            },
            "meta": {
                "nt": 1_000,
                "path": "ncl_test"
            }
        },


        "MAP": {
            "geometry": {
                "alpha_choice": np.array(['constantmean']),
                "s": np.array([0], dtype=int),
                "l": np.array([150], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                "mu": np.array([300]),
                "theta": np.array([50]),
                "lmbda": np.arange(0.10, 0.90, 0.20),
                "alphao": np.array([PROBAS["alphao"]], dtype=float),
                "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
                "beta": np.array([PROBAS["beta"]], dtype=float)
            },
            "rates": {
                "rtot_bind": 1 / np.linspace(0.10, 20, 100),
                "rtot_rest": 1 / np.linspace(0.10, 20, 100)
            },
            "meta": {
                "nt": 1_000,
                "path": "ncl_map"
            }
        }
        
    }

    if config not in presets:
        raise ValueError(f"Unknown configuration: {config}")

    return {
        **presets[config],
        "chromatin": CHROMATIN,
        "time": TIME
    }


def generate_param_combinations(cfg: dict) -> list[dict]:
    """
    Generates the list of parameter combinations from the configuration.
    """
    geometry = cfg['geometry']
    probas = cfg['probas']
    rates = cfg['rates']
    meta = cfg['meta']

    keys = ['alpha_choice', 's', 'l', 'bpmin', 'mu', 'theta', 'lmbda', 'alphao', 'alphaf', 'beta', 'rtot_bind', 'rtot_rest']
    values = product(
        geometry['alpha_choice'], geometry['s'], geometry['l'], geometry['bpmin'],
        probas['mu'], probas['theta'], probas['lmbda'],
        probas['alphao'], probas['alphaf'], probas['beta'],
        rates['rtot_bind'], rates['rtot_rest']
    )

    return [
        dict(zip(keys, vals)) | {"nt": meta['nt'], "path": meta['path']}
        for vals in values
    ]


def run_parallel(params: list[dict], chromatin: dict, time: dict, num_workers: int, use_tqdm: bool = False) -> None:
    """
    Exécute les fonctions en parallèle avec ou sans barre de progression.
    """
    process = partial(process_function, chromatin=chromatin, time=time)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process, p) for p in params]
        iterator = tqdm(as_completed(futures), total=len(futures), desc="Processing") if use_tqdm else as_completed(futures)

        for future in iterator:
            try:
                future.result()
            except Exception as e:
                print(f"Process failed with exception: {e}")


def run_sequential(params: list[dict], chromatin: dict, time: dict, folder_path: str) -> None:
    """
    Exécute les fonctions séquentiellement (utile pour profiling ou debug).
    """
    process = partial(process_function, chromatin=chromatin, time=time)

    for p in tqdm(params, desc="Processing sequentially"):
        try:
            process(p)
        except Exception as e:
            print(f"Process failed with exception: {e}")


def execute_in_parallel(config: str, execution_mode: str, slurm_params: dict) -> None:
    """
    Launches multiple processes based on selected configuration and execution mode.
    """
    cfg = choose_configuration(config)
    chromatin = cfg["chromatin"]
    time = cfg["time"]

    all_params = generate_param_combinations(cfg)

    # Split tasks by SLURM
    task_id = slurm_params['task_id']
    num_tasks = slurm_params['num_tasks']
    task_params = np.array_split(all_params, num_tasks)[task_id]

    # Create working dir
    folder_name = f"{cfg['meta']['path']}_{task_id}"
    set_working_environment(subfolder=folder_name)

    # Execution modes
    if execution_mode == 'PSMN':
        run_parallel(task_params, chromatin, time, num_workers=slurm_params['num_cores_used'])

    elif execution_mode == 'PC':
        run_parallel(all_params, chromatin, time, num_workers=3, use_tqdm=True)

    elif execution_mode == 'SNAKEVIZ':
        folder_path = f"/home/nicolas/tests/{folder_name}"
        run_sequential(all_params, chromatin, time, folder_path)

    else:
        raise ValueError(f"Unknown execution mode: {execution_mode}")


# ================================================
# Part 4 : Main
# ================================================


# ─────────────────────────────────────────────
# 4.1. SLURM environment parsing
# ─────────────────────────────────────────────

def get_slurm_params():
    return {
        'num_cores_used': int(os.getenv('SLURM_CPUS_PER_TASK', '1')),
        'num_tasks': int(os.getenv('SLURM_NTASKS', '1')),
        'task_id': int(os.getenv('SLURM_PROCID', '0'))
    }

# ─────────────────────────────────────────────
# 4.2. Execution parameters
# ─────────────────────────────────────────────

# Options: PSMN / PC / SNAKEVIZ
EXE_MODE = "PC"

# Options: NU / BP / LSLOW / LSHIGH / MAP / TEST
CONFIG = "TEST"

# ─────────────────────────────────────────────
# 4.3. Main function
# ─────────────────────────────────────────────

def main():
    print('\n#- Launched -#\n')
    start_time = time.time()
    initial_address = Path.cwd()

    slurm_env = get_slurm_params()
    print(f"SLURM ENV → {slurm_env}")

    try:
        execute_in_parallel(CONFIG, EXE_MODE, slurm_env)
    except Exception as e:
        print(f"[ERROR] Process failed: {e}")

    os.chdir(initial_address)
    elapsed = time.time() - start_time
    print(f'\n#- Finished in {int(elapsed // 60)}m at {initial_address} -#\n')

# ─────────────────────────────────────────────
# 4.4 Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    main()


# ================================================
# .
# ================================================