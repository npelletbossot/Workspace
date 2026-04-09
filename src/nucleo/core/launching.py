"""
nucleo.launching_functions
------------------------
Launching functions for simulations.
"""


# ─────────────────────────────────────────────
# 1 : Libraries
# ─────────────────────────────────────────────

# 1.1 : Standard 
from __future__ import annotations

import os
import cProfile
import pstats

from enum import Enum
from pathlib import Path
from itertools import product
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date

import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

# 1.2 : Package 
from nucleo.core.configs import choose_configuration
from nucleo.core.run_simulation import process_run
from nucleo.io.writing import set_working_environment


# ─────────────────────────────────────────────
# 2 : Helpers & Enums
# ─────────────────────────────────────────────

class Mode(Enum):
    PSMN        = "PSMN"        # SLURM/cluster
    PC          = "PC"          # Local machine
    SNAKEVIZ    = "SNAKEVIZ"    # Local + profiling


def _detect_mode(execution_mode: str | None) -> Mode:
    """
    Priority:
      1) explicit function arg `execution_mode`
      2) env var EXECUTION_MODE
      3) SLURM env auto-detection -> PSMN
      4) default -> PC
    """
    # 1) Explicit
    if execution_mode and execution_mode.upper() in Mode.__members__:
        return Mode[execution_mode.upper()]

    # 2) ENV
    env_mode = os.getenv("EXECUTION_MODE", "").upper()
    if env_mode in Mode.__members__:
        return Mode[env_mode]

    # 3) SLURM auto
    slurm_markers = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_SUBMIT_DIR", "SLURM_CPUS_PER_TASK")
    if any(k in os.environ for k in slurm_markers):
        return Mode.PSMN

    # 4) Default
    return Mode.PC


def _choose_num_workers(default_workers: int = 2) -> int:
    """
    Use SLURM_CPUS_PER_TASK if available, otherwise fallback.
    """
    return int(os.getenv("SLURM_CPUS_PER_TASK", default_workers))


# ─────────────────────────────────────────────
# 3 : Functions
# ─────────────────────────────────────────────

def generate_param_combinations(cfg: dict) -> list[dict]:
    """
    Generates the list of parameter combinations from the configuration.
    """
    
    # Every specific compartments
    formalism   = cfg['formalism']
    geometry    = cfg['geometry']
    probas      = cfg['probas']
    rates       = cfg['rates']
    meta        = cfg['meta']

    # The keys must be in arrays
    keys = [
        'landscape', 's', 'l', 'bpmin',
        'mu', 'theta', 'lmbda', 'alphaf', 'alphao', 'beta', 'alphad',
        'ktot', 'klist', 'alphar',
        'rtot_capt', 'rtot_rest'
    ]
    
    # All combinations
    values = product(
        geometry['landscape'], geometry['s'], geometry['l'], geometry['bpmin'],
        probas['mu'], probas['theta'], 
        probas['lmbda'], probas['alphaf'], probas['alphao'], probas['beta'], probas['alphad'],
        rates['ktot'], rates['klist'], probas['alphar'], 
        rates['rtot_capt'], rates['rtot_rest']
    )
        
    return [
        dict(zip(keys, vals)) | {
            "algorithm": formalism['algorithm'], 
            "fact": formalism['fact'], 
            "factmode": formalism['factmode'], 
            "nt": meta['nt'], 
            "path": meta['path']}
        for vals in values
    ]


def run_parallel(params: list[dict], formalism: dict, chromatin: dict, time: dict, meta: dict, num_workers: int, use_tqdm: bool = False) -> None:
    """
    Runs processes in parallel with optional progress bar.
    """
    process = partial(process_run, formalism=formalism, chromatin=chromatin, time=time, meta=meta)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process, p) for p in params]
        iterator = tqdm(as_completed(futures), total=len(futures), desc="Processing") if use_tqdm else as_completed(futures)

        for future in iterator:
            try:
                future.result()
            except Exception as e:
                print(f"Process failed with exception: {e}")


def execute_in_parallel(config: str,
                        execution_mode: str | None = None,
                        slurm_params: dict | None = None,
                        study: str | None = None) -> None:
    """
    Launches multiple processes based on selected configuration and execution mode.
    - Auto-detects SLURM (PSMN)
    - Supports SNAKEVIZ profiling (cProfile -> snakeviz_profile.prof)
    """
    mode = _detect_mode(execution_mode)
    slurm_params = slurm_params or {}
    
    env_task_id   = int(os.getenv("SLURM_ARRAY_TASK_ID", os.getenv("SLURM_PROCID", "0")))
    env_num_tasks = int(os.getenv("SLURM_ARRAY_TASK_COUNT", os.getenv("SLURM_NTASKS", "1")))
    task_id       = int(slurm_params.get("task_id", env_task_id))
    num_tasks     = int(slurm_params.get("num_tasks", env_num_tasks))
    num_cores     = int(slurm_params.get("num_cores_used", _choose_num_workers()))

    cfg         = choose_configuration(config)
    project     = cfg['project']
    formalism   = cfg['formalism']
    chromatin   = cfg['chromatin']
    time        = cfg['time']
    meta        = cfg['meta']

    all_params  = generate_param_combinations(cfg)

    if mode == Mode.PSMN:
        # Split équilibré par tâche SLURM
        chunks      = np.array_split(all_params, num_tasks)
        this_params = list(chunks[task_id]) if num_tasks > 1 else all_params
        num_workers = num_workers = min(8, num_cores)
        base_dir    = "/Xnfs/physbiochrom/npellet/"
        use_tqdm    = False
        task_suffix = str(task_id)
    else:
        this_params = all_params
        base_dir    = Path.home() / "Documents" / "Workspace"
        use_tqdm    = True
        task_suffix = str(slurm_params.get('task_id', 0))
        if cfg['meta']['nt'] == 10_000:
            num_workers = 2
        else:
            num_workers = 12

    project_name   = project['project_name']
    folder_name    = f"{cfg['meta']['path']}__{task_suffix}"
    study_label    = f"{study}" if study else ""
    launch_date = os.getenv("LAUNCH_DATE", str(date.today()))
    subfolder_name = f"{project_name}/outputs/{study_label}__{mode.value}__{launch_date}/{folder_name}"

    set_working_environment(base_dir=base_dir, subfolder=subfolder_name)

    if mode == Mode.SNAKEVIZ:
        profile_path = Path("snakeviz_profile.prof")
        pr = cProfile.Profile()
        pr.enable()
        try:
            run_parallel(this_params, formalism, chromatin, time, meta, num_workers=num_workers, use_tqdm=use_tqdm)
        finally:
            pr.disable()
            pr.dump_stats(str(profile_path))
            pstats.Stats(pr).sort_stats("cumtime").print_stats(30)
    else:
        run_parallel(this_params, formalism, chromatin, time, meta, num_workers=num_workers, use_tqdm=use_tqdm)
        
        
# ─────────────────────────────────────────────
# 4 : Main
# ─────────────────────────────────────────────

def main(STUDY):
    
    # All valuable configurations
    CONFIG = {
        "NUCLEO": ["NU", "BP", "LSLOW", "LSHIGH"],
        "COMPACTION": ["COMPACTION_RANDOM", "COMPACTION_PERIODIC"],
        "RYU": ["TWO_STEPS"],
        "FACT": ["FACT_PASSIVE_FULL", "FACT_PASSIVE_MEMORY", "FACT_ACTIVE_FULL", "FACT_ACTIVE_MEMORY"],
        "FIGURES": ["FIGURE_1", "FIGURE_2", "FIGURE_3"],
        "TEST": ["TEST"]
    }

    all_values = [v for values in CONFIG.values() for v in values]

    # Error if outsides of our studies
    if STUDY not in CONFIG and STUDY not in all_values:
        raise ValueError(f"You gave STUDY={STUDY} which is not in {CONFIG}.")
    
    # Filter in sort that we can either launch a config or a list of congifs
    if STUDY in CONFIG:
        configs = CONFIG[STUDY]
    else:
        configs = [STUDY]  

    # Lenght of the study
    n_configs = len(configs)
    print(f"\nNumber of configurations to launch: {n_configs}\n STUDY = {configs}")

    # Main loop
    for i, config in enumerate(
        tqdm(configs, desc="Global progress", unit="config"),
        start=1
    ):
        launch_time = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
        print(f"\n# [{i}/{n_configs}] --- Launched {config} on {launch_time} --- #")

        start_time = time.time()

        try:
            execute_in_parallel(config, study=STUDY)
        except Exception as e:
            print(f"[ERROR in {config}] Process failed:\n{e}")
            continue

        elapsed = time.time() - start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        end_time = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
        print(
            f"# [{i}/{n_configs}] --- Finished {config} on {end_time} --- "
            f"Execution time: {h}h {m}m {s}s --- #\n"
        )