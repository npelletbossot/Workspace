"""
nucleo.utility_functions
------------------------
Utility functions for dictionary merging, math helpers, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

from collections import defaultdict
import numpy as np


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 : Dictionaries


def add_dicts(dict1: dict, dict2: dict) -> dict:
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


# 2.2 : Calculations


def clc_distrib(
    data: np.ndarray, 
    first_bin: float, 
    last_bin: float, 
    bin_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the normalized distribution of data using a histogram.

    Args:
        data (np.ndarray): Array of data values to compute the distribution for.
        first_bin (float): Lower bound of the first bin.
        last_bin (float): Upper bound of the last bin.
        bin_width (float): Width of each bin.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - points (np.ndarray): Array of bin centers.
            - distrib (np.ndarray): Normalized distribution (sum equals 1).
    """

    # Everything in float
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    # Handle empty data array
    if arr.size == 0: 
        return np.array([]), np.array([])

    # Points and not bins
    bins = np.arange(first_bin, last_bin + bin_width, bin_width, dtype=np.float64)
    density, edges = np.histogram(arr, bins=bins, density=True)
    points = (edges[:-1] + edges[1:]) / 2.0

    return points, density


def listoflist_into_matrix(listoflist: list) -> np.ndarray:
    """
    Converts a list of lists with varying lengths into a 2D NumPy array,
    padding shorter rows with np.nan so that all rows have equal length.
    """
    len_max = max(len(row) for row in listoflist)
    matrix = np.full((len(listoflist), len_max), np.nan, dtype=np.float32)
    for i, row in enumerate(listoflist):
        matrix[i, :len(row)] = row
    return matrix


# 2.3 : Mathematics


def exp_decay(t, y0, tau):
    return y0 * np.exp(-t / tau)