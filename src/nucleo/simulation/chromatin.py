"""
nucleo.landscape_functions
------------------------
Landscape functions for generating chromatin, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 : Patterns


def alpha_random(s:int, l:int, alphaf:float, alphao:float, Lmin:int, Lmax:int, bps:int) -> np.ndarray:
    """
    Generates one random landscape


    Args:
        s (int): Value of s, nucleosome size.
        l (int): Value of l, linker length.
        alphaf (float): Probability of beeing accepted on linker sites.
        alphao (float): Probability of beeing accepted on nucleosome sites.
        Lmin (int): First point of chromatin.
        Lmax (int): Last point of chromatin.
        bps (int): Number of base pairs per site.

    Returns:
        np.ndarray: Landscape corresponding to one trajectory.
    """
    np.random.seed()

    alpha_array = np.full(int((Lmax-Lmin)/bps), alphaf)    # creates a NumPy array of length Lmax filled with the value alphaf
    T = int(Lmax / (l + s))                                 # how many blocks of size l + s can fit into the total length Lmax ; represents how many obstacle blocks can be inserted into the list
    max_pos = Lmax - (T * s)                                # determines the maximum possible position for obstacle blocks, taking into account the fact that each block occupies s positions 

    # random generation of position to insert obstacles
    alpha_random = np.random.randint(0, max_pos + 1, T)         # creates an array of random indices to place the obstacle blocks, ranging from 0 to _max_position_ inclusive, with a total of _T_ positions
    alpha_random_sorted = np.sort(alpha_random)                 # sorts random positions to avoid overlapping and ensure orderly placement of obstacle blocks
    alpha_random_modified = alpha_random_sorted + np.arange(len(alpha_random_sorted)) * (s)     # each point is shifted ([2 + 0, 3 + s, 6 + 2*s ....] etc) to prevent overlapping
    
    # filling with obstacles
    for pos in alpha_random_modified:
        alpha_array[pos:pos + s] = alphao                                                       # for each position modified in _alpha_random_modified_, place an obstacle block (alphao) of length s in the _alpha_array

    return np.array(alpha_array, dtype=float)


def alpha_periodic(s:int, l:int, alphaf:float, alphao:float, Lmin:int, Lmax:int, bps:int) -> np.ndarray:
    """Generates one periodic pattern

    Args:
        s (int): Value of s, nucleosome size.
        l (int): Value of l, linker length.
        alphaf (float): Probability of beeing accepted on nucleosome sites.
        alphao (float): Probability of beeing accepted on linker sites.
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
    
    return np.array(alpha_array, dtype=float)


def clc_alpha_mean(
    s: int, l: int, alphaf: float, alphao: float,
    alphar: float, Kp: float,
) -> float:
    """Computes the mean acceptance probability for a homogeneous landscape.

    Args:
        s (int): Nucleosome size (in sites).
        l (int): Linker length (in sites).
        alphaf (float): Acceptance probability on linker sites.
        alphao (float): Acceptance probability on nucleosome sites (off state).
        alphar (float): Acceptance probability on nucleosome sites (on state).
        Kp (float): Fraction of time nucleosome is in 'on' (remodeled) state.

    Returns:
        float: Weighted mean alpha value over one nucleosome + linker unit.
    """
    return (l * alphaf + s * (Kp * alphar + (1 - Kp) * alphao)) / (l + s)


def alpha_homogeneous(
    s: int, l: int, alphaf: float, alphao: float,
    alphar: float, Kp: float,
    Lmin: int, Lmax: int, bps: int,
) -> np.ndarray:
    """Generates a uniform alpha landscape over the chromatin region.

    Args:
        s (int): Nucleosome size (in sites).
        l (int): Linker length (in sites).
        alphaf (float): Acceptance probability on linker sites.
        alphao (float): Acceptance probability on nucleosome sites (off state).
        alphar (float): Acceptance probability on nucleosome sites (on state).
        Kp (float): Fraction of time nucleosome is in 'on' (remodeled) state.
        Lmin (int): First site of chromatin.
        Lmax (int): Last site of chromatin.
        bps (int): Number of base pairs per site.

    Returns:
        np.ndarray: Uniform alpha landscape for one trajectory.
    """
    value = clc_alpha_mean(s, l, alphaf, alphao, alphar, Kp)
    size = int((Lmax - Lmin) / bps)
    return np.full(size, value)


# 2.2 Generation


def clc_alpha_matrix(
    landscape: str, s: int, l: int, bpmin: int, 
    alphaf: float, alphao: float, 
    alphar: float, Kp: float,
    Lmin: int, Lmax: int, bps: int, nt: int
) -> np.ndarray:
    """
    Calculation of the matrix of obstacles, each line corresponding to a trajectory

    Args:
        landscape (str): Choice of the alpha configuration ('random', 'periodic', 'homogeneous').
        s (int): Value of s, nucleosome size.
        l (int): Value of l, linker length.
        bpmin (int): Minimum base pair threshold.
        alphaf (float): Probability of beeing accepted on linker sites.
        alphao (float): Probability of beeing accepted on nucleosome sites.
        Lmin (int): First point of chromatin.
        Lmax (int): Last point of chromatin.
        bps (int): Number of base pairs per site.
        nt (int): Number of trajectories for the simulation.

    Raises:
        ValueError: In case the choice is not aligned with the possibilities

    Returns:
        np.ndarray: Matrix of each landscape corresponding to a trajectory
    """

    alpha_functions = {'periodic', 'one_random', 'random', 'homogen'}
    
    if landscape not in alpha_functions:
        raise ValueError(f"Unknown landscape: {landscape}")
    
    elif landscape == 'periodic' :
        alpha_array = alpha_periodic(s, l, alphaf, alphao, Lmin, Lmax, bps)
        alpha_matrix = np.tile(alpha_array, (nt,1))

    elif landscape == 'one_random' :
        alpha_array = alpha_random(s, l, alphaf, alphao, Lmin, Lmax, bps)
        alpha_matrix = np.tile(alpha_array, (nt,1))
    
    elif landscape == 'homogen' :
        alpha_array = alpha_homogeneous(s, l, alphaf, alphao, alphar, Kp, Lmin, Lmax, bps)
        alpha_matrix = np.tile(alpha_array, (nt,1))

    elif landscape == 'random':
        alpha_matrix = np.empty((nt, int((Lmax - Lmin) / bps)))
        for i in range(nt):
            alpha_matrix[i] = alpha_random(s, l, alphaf, alphao, Lmin, Lmax, bps)

    # Values
    alpha_matrix = np.array([binding_length(alpha_list, alphaf, alphao, bpmin) for alpha_list in alpha_matrix], dtype=np.float32)
    
    return alpha_matrix


# 2.3 Binding minimal size


def binding_length(alpha_list: np.ndarray, alphaf: float, alphao: float, bpmin: int) -> np.ndarray:
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
    
    # Trying because of the case bpmin covering everything
    try:
        # Iterate over sequences and replace if the length is less than `bpmin`
        for start, end in zip(starts, ends):
            length = end - start
            if length < bpmin:
                alpha_array[start:end] = alphao

    # If nothing accessible
    except Exception:
        # Only zeros
        alpha_array = np.zeros_like(alpha_array, dtype=int)
        
    # Return in both cases
    return alpha_array


# 2.4 : Tools


def find_blocks(array: np.ndarray, alpha_value: float) -> list[tuple[int, int]]:
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
    list[tuple[int, int]]
        A list of intervals (start_index, end_index) for each contiguous obstacle block.
    """

    if np.all(np.isclose(array, array[0])):
        return np.array([[0, 0]])

    array = np.asarray(array)
    
    is_block = np.isclose(array, alpha_value, atol=1e-8)
    diff = np.diff(is_block.astype(int))
    
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if is_block[0]:
        starts = np.insert(starts, 0, 0)
    if is_block[-1]:
        ends = np.append(ends, len(array))

    return np.array(list(zip(starts, ends)))


def destroy_obstacles(array: np.ndarray, ratio: float, alphaf: float, alphao: float, begin: int, end: int) -> np.ndarray:
    """
    Randomly destroys a fraction of obstacles in a given 1D landscape array
    within a specified region [begin:end].

    Args:
        array (np.ndarray): 1D array representing the landscape.
        ratio (float): Fraction of obstacles to destroy (between 0 and 1).
        alphaf (float): Value representing the linkers (after destruction).
        alphao (float): Value representing the obstacles (before destruction).
        begin (int): First point (inclusive) to consider for destruction.
        end (int): Last point (exclusive) to consider for destruction.

    Returns:
        np.ndarray: A copy of `array` with some obstacles destroyed in [begin:end].
    """
    if not 0 <= ratio <= 1:
        raise ValueError(f"Ratio must be between 0 and 1, got {ratio}")
    
    # Looking for obstacles in the propre region
    region = array[begin:end].copy()
    couples = find_blocks(region, alphao)
    if couples.size == 0:
        return array

    # Number of obstacles to destroy
    n_obs_destroyed = int(np.floor(ratio * len(couples)))
    if n_obs_destroyed == 0:
        return array

    # Random choice of the obstacles destroyed
    places = np.random.choice(len(couples), size=n_obs_destroyed, replace=False)
    
    # Destroy the selected obstacles
    for idx in places:
        start, stop = couples[idx]
        region[start:stop] = alphaf

    array[begin:end] = region
    return array