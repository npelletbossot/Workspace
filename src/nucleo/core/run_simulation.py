"""
nucleo.run_functions
------------------------
Running functions for one simulation.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

# 1.1 : Standard 
from __future__ import annotations
import numpy as np
import gc


# 1.2 : Package

# 1.2.1 : Simulation
from nucleo.simulation.models import gillespie_algo_one_step, gillespie_algo_two_steps
from nucleo.simulation.chromatin import (
    clc_alpha_mean,
    clc_alpha_matrix,
    destroy_obstacles,
    find_blocks
)
from nucleo.simulation.probabilities import proba_gamma

# 1.2.2 : Tools
from nucleo.metrics.utils import listoflist_into_matrix
from nucleo.metrics.fitting import fitting_in_two_steps

# 1.2.3 : Metrics
from nucleo.metrics.landscape import (
    clc_link_view, 
    clc_obs_and_link_distrib
)
from nucleo.metrics.trajectories import clc_results

from nucleo.metrics.jumps import (
    clc_pos_hist,
    clc_jumpsize_distrib,
    clc_jumptime_distrib,
    clc_fpt_matrix
)
from nucleo.metrics.speeds import (
    clc_th_speed,
    clc_inst_speeds
)

from nucleo.metrics.compaction import clc_compaction_positions

from nucleo.metrics.twosteps import get_jump_nature

# 1.2.4 : Writing
from nucleo.io.writing import inspect_data_types, writing_parquet


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 : Inputs verification


def checking_inputs(
    algo, fact, mode,
    land, s, l, bpmin, 
    mu, theta, 
    alphaf, alphao, beta,
    rcapt, rrest,
    alphac, alphad, alphar, 
    Ktot, Kp, Kz,
    Lmin, Lmax, bps, origin,
    tmax, dt,
    nt, path, data_return, total_return
):
    """
    Validate all input parameters for the simulation before execution.

    This function ensures that all provided parameters related to chromatin
    structure, obstacle configuration, probabilities, remodeling rates,
    trajectory counts, and temporal parameters meet the required constraints.
    It raises detailed error messages to help identify invalid inputs early,
    preventing inconsistencies or undefined behaviors in downstream simulation
    routines.

    Parameters
    ----------
    land : str
        Chromatin landscape model. Must be one of:
        {"homogen", "periodic", "random"}.
    s : np.integer
        Nucleosome size (must be >= 0).
    l : np.integer
        Accessible linker size (must be >= 0 and nonzero).
    bpmin : np.integer
        Minimum number of accessible base pairs (>= 0).
    mu : np.integer
        Parameter controlling random obstacle density (>= 0).
    theta : np.integer
        Parameter controlling mean alpha values (>= 0).
    alphac : np.ndarray
        Probability modifier array. Must satisfy 0 ≤ alphac ≤ 1.
    alphaf : np.ndarray
        FACT-induced capture rate modifier (0 ≤ alphaf ≤ 1).
    alphao : np.ndarray
        Baseline capture probability (0 ≤ alphao ≤ 1).
    beta : np.ndarray
        Unbinding rate (normalized), must satisfy 0 ≤ beta ≤ 1.
    alphar : np.ndarray
        Capture rate in remodeled nucleosomes (0 ≤ alphar ≤ 1).
    kB : np.integer
        FACT binding rate (>= 0).
    kU : np.integer
        FACT unbinding rate (>= 0). The sum kB + kU must be nonzero.
    nt : int
        Number of trajectories to simulate (>= 0).
    Lmin : int
        Minimum lattice coordinate. Must be 0.
    Lmax : int
        Maximum lattice coordinate. Must be > Lmin.
    bps : int
        Number of base pairs per lattice site (>= 0).
    origin : int
        Starting position of loop extrusion. Must satisfy 0 ≤ origin < Lmax.
    tmax : int
        Maximum time for simulation (>= 0).
    dt : float
        Temporal resolution of the simulation (must be > 0).

    Raises
    ------
    ValueError
        If any parameter violates expected constraints, the function raises 
        a ValueError with a precise explanation of the issue.

    Notes
    -----
    - The function captures all ValueErrors inside a try-except block and prints 
      a unified message indicating that the issue originates from `checking_inputs()`,
      followed by the specific error message.
    - This function does not return anything; its purpose is purely validation.
    """
    
    try:
                 
        # Formalism
        if algo not in ["1S", "2S"]:
            raise ValueError(f"Invalid value for algorithm you set : algorithm={algo}")
        if (algo == "one_step") and ((fact != False) or (mode != "none")):
            raise ValueError(f"Error with algorithm and fact you set : algorithm={algo} - fact={fact} - mode={mode}")
        
        if mode not in ["none", "passfull", "passmemo", "actifull","actimemo"]:
            raise ValueError(f"You set factmode={mode} for remodelling which is not a valid mode")  

        # Obstacles
        if land not in {"homogen", "periodic", "random"}:
            raise ValueError(f"Invalid land: {land}. Must be 'homogen', 'periodic', or 'random'.")
        for name, value in [("s", s), ("l", l), ("bpmin", bpmin)]:
            if not isinstance(value, np.integer) or value < 0:
                raise ValueError(f"Invalid value for {name}: must be an int >= 0. Got {value}.")
        if l == 0:
            raise ValueError("You cannot set l=0, there is absolutly accessible places.")

        # Probabilities and Rates
        if not isinstance(mu, np.integer) or mu < 0:
            raise ValueError(f"Invalid value for mu: must be an int >= 0. Got {mu}.")
        if not isinstance(theta, np.integer) or theta < 0:
            raise ValueError(f"Invalid value for theta: must be an int >= 0. Got {theta}.")
        for name, value in zip(["alphaf", "alphao", "beta", "alphac", "alphad", "alphar"], 
                            [alphaf, alphao, beta, alphac, alphad, alphar]):
            if not ((0 <= value).all() and (value <= 1).all()):
                raise ValueError(
                    f"{name} must be between 0 and 1. "
                    f"Got array with min={value.min()}, max={value.max()}."
                )
            
        # Rates
        for name, val in [("rcapt", rcapt), ("rrest", rrest)]:
            if not isinstance(val, (float, int)) or val <= 0:
                raise ValueError(
                    f"Invalid {name}={val}: must be a float strictly > 0."
                )
    
        # --- Ktot (kB + kU) ---
        if Ktot < 0:
            raise ValueError(
                f"Invalid Ktot={Ktot}: must be a float > or = to 0."
            )
        # --- Kp and Kz (probabilities) ---
        kcheck = np.asarray([Kp, Kz])
        if not np.all((0.0 <= kcheck) & (kcheck <= 1.0)):
            raise ValueError(
                f"Invalid Kp/Kz values: must be in [0, 1]. Got {kcheck}."
            )

        # Chromatin
        if Lmin != 0:
            raise ValueError(f"Lmin must be 0. Got {Lmin}.")
        if Lmax <= Lmin:
            raise ValueError(f"Lmax must be greater than Lmin. Got Lmax={Lmax}, Lmin={Lmin}.")
        if not isinstance(bps, int) or bps < 0:
            raise ValueError(f"Invalid value for bps: must be an int >= 0. Got {bps}.")
        if not (0 <= origin < Lmax):
            raise ValueError(f"origin must be within [0, Lmax). Got origin={origin}, Lmax={Lmax}.")
        
        # Density of pattern to avoid boundary effects
        if (s > 50) and (Lmax - Lmin) < 10_000:
            raise ValueError(
                f"You cannot give this values of s={s} and  Lmax-Lmin={Lmax-Lmin}"
                "because it will cause boundary effects."
            )

        # Times
        if not isinstance(tmax, int) or tmax < 0:
            raise ValueError(f"Invalid value for tmax: must be an int >= 0. Got {tmax}.")
        if dt <= 0:
            raise ValueError(f"dt must be positive. Got {dt}.")
        
        # Meta
        if not isinstance(nt, int) or nt < 0:
            raise ValueError(f"Invalid value for nt: must be an int >= 0. Got {nt}.")
        if not isinstance(path, str):
            raise ValueError(f"Invalid format for path: must be an str. Got {type(path)}.")
        if not isinstance(data_return, bool):
            raise ValueError(f"Invalid format for data_return: must be a bool. Got {type(data_return)}.")
        if not isinstance(total_return, bool):
            raise ValueError(f"Invalid format for total_return: must be a bool. Got {type(total_return)}.")
        
    except Exception as e:
        print(f"The error is in the checking_inputs() function and is : {e}")


# 2.2 : Stochastic Walker


def sw_nucleo(
    algo: str, fact: str, mode: str, dstr: bool,
    land: str, s: int, l: int, bpmin: int,
    mu: float, theta: float, 
    alphaf: float, alphao: float, beta: float, 
    rcapt: float, rrest: float,
    alphac: float, alphad: float, alphar: float, 
    Ktot: float, Kp: float, Kz: float,
    Lmin: int, Lmax: int, bps: int, origin: int,
    tmax: float, dt: float,
    nt: int, path: str,
    data_return: bool = False, total_return: bool = False
    ) -> None:
    """
    Simulates condensin dynamics along chromatin with specified parameters.

    Args:
        land (str): Choice of the alpha configuration ('ntrandom', 'periodic', 'constantmean').
        s (int): Nucleosome size.
        l (int): Linker length.
        bpmin (int): Minimum base pair threshold.
        mu (float): Mean value for the distribution used in the simulation.
        theta (float): Standard deviation for the distribution used in the simulation.
        alphaf (float): Acceptance probability on linker sites.
        alphao (float): Acceptance probability on nucleosome sites.
        beta (float): Unfolding probability.
        alphac (float) : Acceptance probability of in vitro condensin.
        rtot_capt (float): Reaction rate for capturing (inverse of characteristic time).
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
    
    # Compactions
    c_linker = 10 / 10
    c_nucleo = 150 / 35

    # Title & Folder    
    title = (
            f"algo={algo}__fact={fact}__mode={mode}__dstr={dstr}__"
            f"land={land}__s={s}__l={l}__bpmin={bpmin}__"
            f"mu={mu}__theta={theta}__"
            f"rcapt={rcapt:.1e}__rrest={rrest:.1e}__"
            f"alphac={alphac:.1e}__alphad={alphad:.1e}__alphar={alphar:.1e}__"
            f"Ktot={Ktot:.1e}__Kp={Kp:.1e}__Kz={Kz:.1e}__"
            f"nt={nt:.1e}__"
    )
    
    # Chromatin
    security_step = 1e-6
    L = np.arange(Lmin + security_step, Lmax + security_step, bps)
    lenght = (Lmax-Lmin) // bps

    # Time 
    times = np.arange(0,tmax,dt)

    # Linear factor
    alpha0 = int(1e+0)

    # Bins for Positions and Times : fb (firstbin) - lb (lastbin) - bw (binwidth)
    x_fb, x_lb, x_bw = 0, 10_000, 1
    t_fb, t_lb, t_bw = 0, 100, 0.20
    x_bins = np.arange(x_fb, x_lb, x_bw)
    t_bins = np.arange(t_fb, t_lb, t_bw)
    binx = int(1e0)
    bint = int(1e+1)


    # ------------------- Input 1 : Chromatin ------------------- #
    
    try:

        # Chromatin Generation : Landscape
        alpha_matrix = clc_alpha_matrix(
            land, s, l, bpmin, 
            alphaf, alphao, 
            alphar, Kp, 
            Lmin, Lmax, bps, nt
        )
            
        # Chromatin Generation : Destroying Obstacles
        if dstr and not np.isclose(alphad, 0.0, atol=1e-8):
            first_point = Lmin
            last_point = Lmax
            for i in range(len(alpha_matrix)):
                alpha_matrix[i] = destroy_obstacles(alpha_matrix[i], alphad, alphaf, alphao, first_point, last_point)
                
    except Exception as e:
        print(f"Error in Input 1 - Chromatin : {e}")
            

    # ------------------- Input 2 : Probability ------------------- #

    try:
        
        # Probabilities
        p = proba_gamma(mu, theta, L)
    
    except Exception as e:
        print(f"Error in Input 2 - Probability : {e}")
    
    
    # ------------------- Simulations ------------------- #

    try:
        
        # Gillespie One-Step
        if algo == "1S":
            results, t_matrix, x_matrix = gillespie_algo_one_step(
                nt, tmax, dt, alpha_matrix, beta, Lmax, lenght, origin, p
            )
            
        # Gillespie Two-Steps
        elif algo == "2S":
            results, t_matrix, x_matrix = gillespie_algo_two_steps(
                fact, mode,
                alpha_matrix, p,
                s, 
                alphao, beta,
                rcapt, rrest, 
                alphac, alphar, 
                Ktot, Kp, Kz, 
                L, origin, bps,
                tmax, dt, 
                nt
            )  

        # Clean datas
        x_matrix = listoflist_into_matrix(x_matrix)
        t_matrix = listoflist_into_matrix(t_matrix)
        
    except Exception as e:
        print(f"Error in Simulations: {e} in {title}")
        
        
    # ------------------- Analysis 1 : Landscape ------------------- #

    try:
        
    # Chromatin Analysis : Obstacles Linkers Distribution
        s_mean, s_points, s_distrib, l_mean, l_points, l_distrib = clc_obs_and_link_distrib(
            land, s, l, alpha_matrix[0], alphaf, alphao, binx
        )

        # Chromatin Analysis : Linker Profile
        l_view = clc_link_view(
            alpha_matrix, land, alphaf, Lmin, Lmax, nt
        )
        
        # Chromatin Analysis : Mean Landscape - Array / Value / Calculated
        alpha_mean_a = np.mean(alpha_matrix, axis=0)
        alpha_mean_v = np.mean(alpha_mean_a)
        alpha_mean_c = clc_alpha_mean(alphaf, alphao, s_mean, l_mean, alphar, Kp)
        
        # Chromatin Remodelling : Obstacles Positions
        obstacles = find_blocks(alpha_matrix[0], alphao)


    except Exception as e:
        print(f"Error in Analysis 1 - Landscape : {e}")
        

    # ------------------- Analysis 2 : Trajectories ------------------- #
    
    try:

        # Main Results
        results_mean, results_med, results_std, v_mean, v_med = clc_results(
            results, dt, alpha0, lb=20
        )
        
        # Fits
        vf, Cf, wf, vf_std, Cf_std, wf_std, xt_over_t, G, bound_low, bound_high = fitting_in_two_steps(
            times, results_mean, results_std
        )
                
        # Theoretical
        v_mean_th = clc_th_speed(algo, s, l, mu, alphaf, alphao, rcapt, rrest, alphac, alphar, Kp)
        v_mean_th_eff = clc_th_speed(algo, s_mean, l_mean, mu, alphaf, alphao, rcapt, rrest, alphac, alphar, Kp)
    
    except Exception as e:
        print(f"Error in Analysis 2 - Trajectories: {e}")
        
    
    # ------------------- Analysis 2 : Jump size + Jump time + First pass times ------------------- #
    
    if total_return:
        
        try:
            
            # Histogram Arrays
            pos_hist = clc_pos_hist(
                results, Lmax, origin, tmax
            )

            # Jump Size Distribution
            xbj_points, xbj_distrib = clc_jumpsize_distrib(
                x_matrix, x_fb, x_lb, x_bw
            )

            # Time Size Distribution
            tbj_points, tbj_distrib = clc_jumptime_distrib(t_matrix)

            # First Pass Times
            fpt_distrib, fpt_number = clc_fpt_matrix(t_matrix, x_matrix, tmax, bint) 
            
        except Exception as e:
            print(f"Error in Analysis 3 - Jump size + Time size + First pass times : {e}")


    # ------------------- Analysis 4 : Speeds ------------------- #
    
    try:
        
        # All Jumps
        if algo == "1S":
            pass
            t_analysis = t_matrix   # Does not use memory
            x_analysis = x_matrix   # Does not use memory
        
        # Forward Jumps
        elif algo == "2S":
            t_forward, x_forward, t_reverse, x_reverse = get_jump_nature(t_matrix, x_matrix)
            t_analysis = np.cumsum(t_forward, axis=1)
            del t_forward
            gc.collect()
            x_analysis = x_forward
            del x_forward
            gc.collect()
        
        # Instantaneous Speeds [Sites][vi_*]
        dx_points, dx_distrib, dx_mean, dx_med, dx_mp, \
        dt_points, dt_distrib, dt_mean, dt_med, dt_mp, \
        vi_points, vi_distrib, vi_mean, vi_med, vi_mp = clc_inst_speeds(
            t_analysis, x_analysis
        )            
        
        x_matrix_c = clc_compaction_positions()
        # Instantaneous Speeds [Base Pairs][vb_*]
        vc_points, vc_distrib, vc_mean, vc_med, vc_mp = clc_compaction_statistics(
            alpha_matrix, t_matrix, x_matrix, c_linker, c_nucleo
        )
                    
    except Exception as e:
        print(f"Error in Analysis 4 - Speeds : {e}")
         

    # ------------------- Analysis 4 : Rates and Taus ------------------- #
    
    # try:
    
    #     if (FORMALISM == "2") or (FORMALISM == "3"):
            
            # # Dwell times
            # dwell_points, forward_result, reverse_result = calculate_dwell_distribution(
            #     t_matrix, x_matrix, t_fb, t_lb, t_bw
            # )
            # tau_forwards, tau_reverses = calculate_dwell_times(
            #     dwell_points, distrib_forwards=forward_result, distrib_reverses=reverse_result, xmax=100
            # )

            # # Rates and Taus
            # fb_y, fr_y, rb_y, rr_y = calculate_nature_jump_distribution(t_matrix, x_matrix, t_fb, t_lb, t_bw)
            # tau_fb, tau_fr, tau_rb, tau_rr = extracting_taus(fb_y, fr_y, rb_y, rr_y, t_bins)
            # rtot_capt_fit, rtot_rest_fit = calculating_rates(tau_fb, tau_fr, tau_rb, tau_rr)
            # v_th_fit = calculate_theoretical_speed(alphaf, alphao, s, l, mu, lmbda, rtot_capt_fit, rtot_rest_fit, alphar, kB, kU, FORMALISM)
            
    # except Exception as e:
    #     print(f"Error in Analysis 4 - Rates and Taus : {e} for {title}")


    # ------------------- Tests ------------------- #

    # print(type(vc_mean), vc_mean)
    
    
    # ------------------- Data ------------------- #
    
    try:

        # Cleaning data for memory
        del alpha_matrix
        gc.collect()

        
        data_result = {

            # --- Algorithm --- #
            'algo'      : algo,
            'fact'      : fact,
            'mode'      : mode,
            'dstr'      : dstr,

            # --- Principal Parameters --- #
            'land'      : land,
            's'         : s,
            'l'         : l,
            'bpmin'     : bpmin,
            'mu'        : mu,
            'theta'     : theta,
            'alphaf'    : alphaf,
            'alphao'    : alphao,
            'beta'      : beta,
            'rcapt'     : rcapt,
            'rrest'     : rrest,
            'alphac'    : alphac,
            'alphad'    : alphad,
            'alphar'    : alphar,
            'Ktot'      : Ktot,
            'Kp'        : Kp,
            'Kz'        : Kz,
            'c_linker'  : c_linker,
            'c_nucleo'  : c_nucleo,

            # --- Chromatin Parameters --- #
            'Lmin'      : Lmin,
            'Lmax'      : Lmax,
            'bps'       : bps,
            'origin'    : origin,

            # --- Time Parameters --- #
            'tmax'      : tmax,
            'dt'        : dt,
            'times'     : times,

            # --- Bins --- #
            'binx'      : binx,
            'bint'      : bint,

            # --- Simulation --- #
            'nt'        : nt,
        }
        
        data_result.update({

        # --- Means --- #
        's_mean'        : s_mean,
        'l_mean'        : l_mean,
        'alpha_mean_a'  : alpha_mean_a,
        'alpha_mean_v'  : alpha_mean_v,
        'alpha_mean_c'  : alpha_mean_c,

        # --- Linear Speeds --- #
        'v_mean'        : v_mean,
        'v_med'         : v_med,
        'v_mean_th'     : v_mean_th,
        'v_mean_th_eff' : v_mean_th_eff,
        
        # --- Fits --- #
        'vf'            : vf,
        'Cf'            : Cf,
        'wf'            : wf,
        'vf_std'        : vf_std,
        'Cf_std'        : Cf_std,
        'wf_std'        : wf_std,
        
        # --- Instantaneous Speeds --- #
        'vi_mean'       : vi_mean,
        'vi_med'        : vi_med,
        'vi_mp'         : vi_mp,
        
        # --- Compaction --- #
        'vc_mean'    : vc_mean,
        'vc_med'     : vc_med,
        'vc_mp'      : vc_mp
        
        })

        if data_return:
            data_result.update({
                't_matrix'     : t_matrix,
                'x_matrix'     : x_matrix,
                'results'      : results,
            })
        
        if total_return:
            data_result.update({

                # --- Chromatin (full) --- #
                'obstacles'    : obstacles,
                's_points'     : s_points,
                's_distrib'    : s_distrib,
                'l_points'     : l_points,
                'l_distrib'    : l_distrib,
                'l_view'       : l_view,

                # --- Raw Datas --- #
                'p'            : p,

                # --- Results --- #
                'results_mean' : results_mean,
                'results_med'  : results_med,
                'results_std'  : results_std,

                # --- Between Jumps --- #
                'pos_hist'     : pos_hist,
                'xbj_points'   : xbj_points,
                'xbj_distrib'  : xbj_distrib,
                'tbj_points'   : tbj_points,
                'tbj_distrib'  : tbj_distrib,

                # --- FPT --- #
                'fpt_distrib'  : fpt_distrib,
                'fpt_number'   : fpt_number,

                # --- Instantaneous stats --- #
                'dx_points'    : dx_points,
                'dx_distrib'   : dx_distrib,
                'dx_mean'      : dx_mean,
                'dx_med'       : dx_med,
                'dx_mp'        : dx_mp,

                'dt_points'    : dt_points,
                'dt_distrib'   : dt_distrib,
                'dt_mean'      : dt_mean,
                'dt_med'       : dt_med,
                'dt_mp'        : dt_mp,

                'vi_points'    : vi_points,
                'vi_distrib'   : vi_distrib,
                
                'vc_points'    : vc_points,
                'vc_distrib'   : vc_distrib,

                # --- Fits --- #
                'alpha0'       : alpha0,
                'xt_over_t'    : xt_over_t,
                'G'            : G,
                'bound_low'    : bound_low,
                'bound_high'   : bound_high,

            })


        # ------------------- Writing ------------------- #

        # Types of data registered if needed
        inspect_data_types(data_result, launch=False)

        # Writing data
        writing_parquet(title=title, data_result=data_result)

        # Clean raw datas
        del data_result
        gc.collect()
        
    except Exception as e:
        print(f"Error in Writing : {e}")
        
        
    # ------------------- Return ------------------- #

    return None


# 2.3 : One run


def process_run(params: dict, formalism: dict, chromatin: dict, time: dict, meta:dict) -> None:
    """
    Executes one simulation with the given parameters and shared constants.
    
    Args:
        params (dict): One combination of geometry + probas + rates + meta parameters.
        chromatin (dict): Dict with Lmin, Lmax, bps, origin.
        time (dict): Dict with tmax, dt.
    """
    checking_inputs(
        algo=formalism["algo"],
        fact=formalism["fact"],
        mode=formalism["mode"], 
            
        land=params['land'],
        s=params['s'],
        l=params['l'],
        bpmin=params['bpmin'],

        mu=params['mu'],
        theta=params['theta'],
        alphaf=params['alphaf'],
        alphao=params['alphao'],
        beta=params['beta'],

        rcapt = params["rcapt"],
        rrest = params["rrest"],

        alphac=params['alphac'],
        alphad=params['alphad'],
        alphar=params['alphar'],

        Ktot=params['Ktot'],
        Kp=params['Kp'],
        Kz=params['Kz'],
        
        Lmin=chromatin["Lmin"],
        Lmax=chromatin["Lmax"],
        bps=chromatin["bps"],
        origin=chromatin["origin"],
        
        tmax=time["tmax"],
        dt=time["dt"],

        nt=meta["nt"],
        path=meta["path"],
        data_return=meta["data_return"],
        total_return=meta["total_return"]
    )

    sw_nucleo(
        formalism["algo"], formalism["fact"], formalism["mode"], formalism["dstr"],
        params["land"], params["s"], params["l"], params["bpmin"],
        params["mu"], params["theta"],

        params["alphaf"], params["alphao"], params["beta"],
        params["rcapt"], params["rrest"],
        params["alphac"], params["alphad"], params["alphar"],
        params["Ktot"], params["Kp"], params["Kz"],
        
        chromatin["Lmin"], chromatin["Lmax"], chromatin["bps"], chromatin["origin"],
        time["tmax"], time["dt"],

        meta["nt"], meta["path"], 
        meta["data_return"], meta["total_return"]
    )