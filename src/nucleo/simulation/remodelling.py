"""
nucleo.remodelling_functions
------------------------
Remoodelling functions for chromatin, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

# 1.1 : Standard 
import numpy as np


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 : Modes


def fact_passive(Kp: float) -> bool:
    """
    Determine whether FACT-mediated chromatin remodelling occurs
    using a passive (time-independent) model.

    In this approximation, FACT binding dynamics are not modelled explicitly.
    FACT is assumed to be present on the nucleosome with a constant probability:

        P(F) = K

    where K represents the equilibrium occupancy of the FACT-bound state.
    Each remodelling attempt is treated as an independent Bernoulli trial,
    with no temporal correlations or internal dynamics.

    This model corresponds to a mean-field, static description of FACT activity,
    valid when FACT binding/unbinding is either much faster than all other
    processes or when temporal fluctuations are intentionally neglected.

    Parameters
    ----------
    K : float
        Equilibrium probability that FACT is bound to the nucleosome.

    Returns
    -------
    bool
        True if chromatin remodelling occurs, False otherwise.
    """
    r_fact = np.random.rand()
    PF = Kp
    return (r_fact < PF)


# def fact_active(kU: float, kBp: float, Kz: float, Kp: float, t_rest: float) -> bool:
#     """
#     Determine whether FACT-mediated chromatin remodelling occurs
#     during a rest period using an active mean-field approximation.

#     This function models FACT as a two-state binding process (bound/unbound),
#     but integrates the binding dynamics analytically over the rest time t_rest
#     rather than simulating individual stochastic trajectories.

#     The probability of remodelling during t_rest is given by:

#         PF = Kz * exp(-(kBp + kU) * t_rest)
#            + Kp * [1 - exp(-(kBp + kU) * t_rest)]

#     where the exponential term describes relaxation toward the steady-state
#     FACT occupancy with characteristic time (kBp + kU)^(-1).

#     This approach neglects intra-rest temporal fluctuations and memory effects,
#     and is equivalent to averaging over all possible FACT trajectories during
#     the rest interval.

#     Parameters
#     ----------
#     kU : float
#         FACT unbinding rate (F → NF).
#     kBp : float
#         FACT binding rate (NF → F).
#     Kz : float
#         Initial probability of FACT being bound at the start of the rest period.
#     Kp : float
#         Equilibrium probability of FACT being bound.
#     t_rest : float
#         Duration of the rest period during which remodelling may occur.

#     Returns
#     -------
#     bool
#         True if chromatin remodelling occurs during the rest period,
#         False otherwise.
#     """
#     r_fact = np.random.rand()
#     PF = Kz * np.exp(-(kBp + kU) * t_rest) + Kp * (1 - np.exp(-(kBp + kU) * t_rest))        
#     return (r_fact < PF)


def fact_active(Ktot: float, Kp: float, Kz: float, t_rest: float) -> bool:
    """
    Determine whether FACT-mediated chromatin remodelling occurs
    during a rest period using an active mean-field approximation.

    The remodelling probability during t_rest is:
        PF = Kz * exp(-k_rel * t_rest) + Kp * [1 - exp(-k_rel * t_rest)]

    where k_rel = kBp + kU is the total FACT relaxation rate toward
    steady-state occupancy Kp.

    Parameters
    ----------
    k_rel : float
        Total FACT relaxation rate (kBp + kU), i.e. inverse of the
        characteristic binding/unbinding timescale.
    Kz : float
        Initial probability of FACT being bound at the start of the rest period.
    Kp : float
        Equilibrium probability of FACT being bound.
    t_rest : float
        Duration of the rest period.

    Returns
    -------
    bool
        True if chromatin remodelling occurs, False otherwise.
    """
    r_fact = np.random.rand()
    k_rel = 1 / Ktot
    exp_term = np.exp(-k_rel * t_rest)
    PF = Kz * exp_term + Kp * (1 - exp_term)
    return r_fact < PF


# def fact_pheno(kB: float, kU: float, K: float, t_rest: float) -> bool:
#     """
#     Determine whether FACT-mediated chromatin remodelling occurs during a rest period tR.

#     This function implements a two-state stochastic model for FACT binding dynamics:
#     - State F  : FACT is bound to the nucleosome.
#     - State NF : FACT is unbound.
    
#     At the beginning of the rest interval tR, the system is assigned a state according to
#     the equilibrium probability:
#         P(F) = K = kB / (kB + kU)
#         P(NF) = 1 - K
    
#     Depending on the state, dwell times are drawn using exponential distributions:
#         T_F   ~ Exp(kU)
#         T_NF  ~ Exp(kB)
    
#     A random internal time T is drawn uniformly within the dwell interval 
#     (i.e. T = TF * U or T = TNF * U with U ~ Uniform[0,1]).
    
#     During the rest period tR, remodelling may occur either:
#     1. Immediately: if tR < (T_state - T)
#     2. After a delayed event with probability:
    
#         PF  = K * [1 - exp(-(kB + kU) * (tR - (TF  - T)))]    for state F
#         PNF = exp(-(kB + kU) * (tR - (TNF - T))) 
#               + K * [1 - exp(-(kB + kU) * (tR - (TNF - T)))]  for state NF
    
#     Parameters
#     ----------
#     kB : float
#         FACT binding rate (NF → F).
#     kU : float
#         FACT unbinding rate (F → NF).
#     K : float
#         Equilibrium occupancy of the FACT-bound state (kB / (kB + kU)).
#     t_rest : float
#         Rest period duration during which remodelling may occur.
    
#     Returns
#     -------
#     bool
#         True if chromatin remodelling occurs during tR, False otherwise.
#     """

#     r_fact = np.random.rand()
    
#     # --- FACT state (F) --- #
#     if r_fact < K:
#         TF = -1 / kU * np.log(np.random.rand())
#         rF1 = np.random.rand()
#         T = TF * rF1
        
#         # Immediate remodelling success
#         if t_rest < (TF - T):
#             return True
        
#         # Delayed remodelling
#         rF2 = np.random.rand()
#         PF = K * (1 - np.exp(-(kB + kU) * (t_rest - (TF - T))))
#         return (rF2 < PF)
            
#     # --- Non-FACT state (NF) --- #       
#     else:
#         TNF = -1 / kB * np.log(np.random.rand())
#         rF1 = np.random.rand() 
#         T = TNF * rF1
        
#         # Immediate remodelling failure
#         if t_rest < (TNF - T):
#             return False
            
#         # Delayed remodelling
#         rF2 = np.random.rand()
#         PNF = (
#             np.exp(-(kB + kU) * (t_rest - (TNF - T)))
#             + K * (1 - np.exp(-(kB + kU) * (t_rest - (TNF - T))))
#         )
#         return (rF2 < PNF)
    

# 2.2 Remodelling


def remodelling_obstacle(
    s: int, alpha_array: np.ndarray, x: int,
    pos_obs: np.ndarray, start_obs: np.ndarray, end_obs: np.ndarray,
    alphar: float
):
    mask = (start_obs <= x) & (x <= end_obs)
    hit_obs = mask.any()
    if hit_obs:
        start, end = pos_obs[mask][0]
        if x == end:
            alpha_array[end-s:end] = alphar
        else:
            rank_obs = int((x - start) / s)
            alpha_array[start + rank_obs * s : start + (rank_obs + 1) * s] = alphar
                
    return np.array(alpha_array)
    
    
def remodelling(
    factmode,
    alpha_array, s,
    x, r_capt,
    pos_obs, start_obs, end_obs,
    alphar,
    Ktot, Kp, Kz, t_rest, 
):
    
    factmodes = ["passfull", "passmemo", "actifull", "actimemo"]
    if factmode not in factmodes:
        raise ValueError(f"You set factmode={factmode} which is not in {factmodes}")
                
    if factmode == "passfull":
        if fact_passive(Kp):
            r_capt = alphar
        
    elif factmode == "passmemo":
        if fact_passive(Kp):
            alpha_array = remodelling_obstacle(s, alpha_array, x, pos_obs, start_obs, end_obs, alphar)
            r_capt = alpha_array[x]

    elif factmode == "actifull":
        if fact_active(Ktot, Kp, Kz, t_rest):
            r_capt = alphar

    elif factmode == "actimemo":
        if fact_active(Ktot, Kp, Kz, t_rest):
            alpha_array = remodelling_obstacle(s, alpha_array, x, pos_obs, start_obs, end_obs, alphar)
            r_capt = alpha_array[x]    
            
    return(r_capt, alpha_array)