"""
nucleo.config_functions
------------------------
Config functions for simulations, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────

def choose_configuration(config: str) -> dict:
    """
    Returns a dictionary of study parameters organized in logical blocks.
    All list-like parameters are converted to np.array.
    3 levels logic.
    """

    # ──────────────────────────────────
    # Shared constants (used everywhere)
    # ──────────────────────────────────
    
    PROJECT = {
        "project_name": "nucleo"
    }
    
    FORMALISMS = {
        # one_step
        "alg1": {
            "algo": "1S",
            "dstr": False,
            "fact": False,
            "mode": "none",
        },
        
        # one_step + destruction
        "alg1_destroy": {
            "algo": "1S",
            "dstr": True,
            "fact": False,
            "mode": "none",
        },
        
        # two steps
        "alg2": {
            "algo": "2S",
            "dstr": False,
            "fact": False,
            "mode": "none",
        },
        
        # two_steps + fact passive
        "alg2_passive_full": {
            "algo": "2S",
            "dstr": False,
            "fact": True,
            "mode": "passfull",
        },
        "alg2_passive_memory": {
            "algo": "2S",
            "dstr": False,
            "fact": True,
            "mode": "passmemo",
        },
        
        # two_steps + fact active
        "alg2_active_full": {
            "algo": "2S",
            "dstr": False,
            "fact": True,
            "mode": "actifull",
        },
        "alg2_active_memory": {
            "algo": "2S",
            "dstr": False,
            "fact": True,
            "mode": "actimemo",
        },
        
    }

    CHROMATIN = {
        "Lmin": 0,          # First point of chromatin (included !)
        "Lmax": 50_000,     # Last point of chromatin (excluded !)
        "bps": 1,           # Based pair step 1 per 1
        "origin": 10_000    # Falling point of condensin on chromatin 
    }

    TIME = {
        "tmax": 100,        # Total time of modeling : 0 is taken into account
        "dt": 1e0           # Step of time
    }

    PROBAS = {
        "alphaf": 1.00,     # Probability of binding if linker
        "alphao": 0.00,     # Probability of binding if obstacle
        "beta": 0.00,       # Probability of in vitro condensin to unbind and leaving DNA
        "alphac": 0.60,     # Probability of in vitro condensin to extrude and beeing accepted
        "alphad": 0.00,     # Probability of nucleosome to drop out
        "alphar": 0.00      # Probability of binding while FACT is there
    }

    RATES = {
        "rcapt": 1/6,   # Rate of capturing (1/6)
        "rrest": 1/6,   # Rate of resting (1/6)
        # "kB" : 0.50,        # Rate of FACT Binding
        # "kU": 0.50,         # Rate of FACT Unbinding
        "Ktot" : 1.0,       # = kB + kU         : New formalism -> Checking that even with =1.0 it doesn't affect if non called          
        "Kp": 1.0,          # = kB / (kB + kU)  : New formalism -> Checking that even with =1.0 it doesn't affect if non called
        "Kz": 1.0           # = P_F(t=0)        : New formalism -> Checking that even with =1.0 it doesn't affect if non called
    }
    
    # ──────────────────────────────────
    # Shared configurations
    # ──────────────────────────────────
    
    ONESTEP__BASE = {
        "formalism": {**FORMALISMS['alg1']},
        "probas": {
            "mu": np.arange(100, 605, 5),
            "theta": np.arange(1, 101, 1),
            "alphao": np.array([PROBAS["alphao"]], dtype=float),
            "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
            "beta": np.array([PROBAS["beta"]], dtype=float),
            "alphac": np.array([1.00], dtype=float),
            "alphad": np.array([0.00], dtype=float),
            "alphar": np.array([0.00], dtype=float),
        },
        "rates": {
            "rcapt": np.array([RATES["rcapt"]], dtype=float),
            "rrest": np.array([RATES["rrest"]], dtype=float),
            "Ktot": np.array([RATES["Ktot"]], dtype=float),
            "Kp": np.array([RATES["Kp"]], dtype=float),
            "Kz": np.array([RATES["Kz"]], dtype=float),
        },
        "meta": {
            "nt": 10_000,
            "data_return": True,
            "total_return": True
        }
    }
    
    COMPACTION_BASE = {
        "formalism": {**FORMALISMS['alg1_destroy']},
        "geometry":{
            "s": np.array([35], dtype=int)
        }, 
        "probas": {
            "mu": np.array([150], dtype=int),
            "theta": np.array([25, 50, 100], dtype=int),
            "alphao": np.array([PROBAS["alphao"]], dtype=float),
            "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
            "beta": np.array([PROBAS["beta"]], dtype=float),
            "alphac": np.array([1.00], dtype=float),
            "alphad": np.array([0.00], dtype=float),
            "alphar": np.array([0.00], dtype=float)
        },
        "rates": {
            "rcapt": np.array([RATES["rcapt"]], dtype=float),
            "rrest": np.array([RATES["rrest"]], dtype=float),
            "Ktot": np.array([RATES["Ktot"]], dtype=float),
            "Kp": np.array([RATES["Kp"]], dtype=float),
            "Kz": np.array([RATES["Kz"]], dtype=float),
        },
        "meta": {
            "nt": 10_000,
            "data_return": True,
            "total_return": True
        }
    }
    
    TWOSTEPS__BASE = {
        "formalism": {**FORMALISMS['alg2']},
        "geometry": {
            "landscape": np.array(['homogen', 'periodic', 'random']),
            "s": np.array([35], dtype=int),
            "l": np.array([10, 35, 100], dtype=int),
            "bpmin": np.array([0], dtype=int)
        },
        "probas": {
            "mu": np.array([150], dtype=int),
            "theta": np.array([25, 50, 100], dtype=int),
            "alphao": np.array([PROBAS["alphao"]], dtype=float),
            "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
            "beta": np.array([PROBAS["beta"]], dtype=float),
            "alphac": np.array([PROBAS["alphac"]], dtype=float),
            "alphad": np.array([PROBAS["alphad"]], dtype=float),
            "alphar": np.arange(0.00, 1.00 + 0.10, 0.10, dtype=float),
        },
        "rates": {
            "rcapt": np.array([RATES["rcapt"]], dtype=float),
            "rrest": np.array([RATES["rrest"]], dtype=float),
            "Ktot": np.array([1.00], dtype=float),
            "Kp": np.arange(0.0, 1.0 + 0.10, 0.10, dtype=float),
            "Kz": np.arange(0.0, 1.0 + 0.10, 0.10, dtype=float),
        },
        "meta": {
            "nt": 10_000,
            "data_return": True,
            "total_return": True
        }
    }
    
    TEST__BASE = {
        "formalism": {**FORMALISMS['alg2_active_memory']},
        "geometry": {
            "landscape": np.array(['homogen', 'periodic', 'random']),
            "s": np.array([35], dtype=int),
            "l": np.array([10], dtype=int),
            "bpmin": np.array([0], dtype=int)
        },
        "probas": {
            "mu": np.array([150], dtype=int),
            "theta": np.array([100], dtype=int),
            "alphao": np.array([PROBAS["alphao"]], dtype=float),
            "alphaf": np.array([PROBAS["alphaf"]], dtype=float),
            "beta": np.array([PROBAS["beta"]], dtype=float),
            "alphac": np.array([PROBAS["alphac"]], dtype=float),
            "alphad": np.array([PROBAS["alphad"]], dtype=float),
            "alphar": np.array([PROBAS["alphar"]], dtype=float),
        },
        "rates": {
            "rcapt": np.array([RATES["rcapt"]], dtype=float),
            "rrest": np.array([RATES["rrest"]], dtype=float),
            "Ktot": np.array([RATES["Ktot"]], dtype=float),
            "Kp": np.arange(0.0, 1.0 + 0.10, 0.10, dtype=float),
            "Kz": np.arange(0.0, 1.0 + 0.10, 0.10, dtype=float),
        },
        "meta": {
            "nt": 100,
            "data_return": True,
            "total_return": True
        }
    }

    # ──────────────────────────────────
    # Presets for study configurations
    # ──────────────────────────────────

    presets = {

        # ---- STATIC : ONE STEP ---- #
        
        "NU": {
            **ONESTEP__BASE,
            "geometry": {
                "landscape": np.array(['homogen', 'periodic', 'random']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "meta": {
                **ONESTEP__BASE["meta"],
                "path": f"{PROJECT['project_name']}__nu"
            }
        },

        "BP": {
            **ONESTEP__BASE,
            "geometry": {
                "landscape": np.array(['random']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([5, 10, 15], dtype=int)
            },
            "meta": {
                **ONESTEP__BASE["meta"],
                "path": f"{PROJECT['project_name']}__bp"
            }
        },

        "LSLOW": {
            **ONESTEP__BASE,
            "geometry": {
                "landscape": np.array(['random']),
                "s": np.array([150], dtype=int),
                "l": np.array([5, 15, 20, 25], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "meta": {
                **ONESTEP__BASE["meta"],
                "path": f"{PROJECT['project_name']}__lslow"
            }
        },

        "LSHIGH": {
            **ONESTEP__BASE,
            "geometry": {
                "landscape": np.array(['random']),
                "s": np.array([150], dtype=int),
                "l": np.array([50, 100, 150], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "meta": {
                **ONESTEP__BASE["meta"],
                "path": f"{PROJECT['project_name']}__lshigh"
            }
        },
            
        # ---- ACCESSIBILITY WITH DESTRUCTION ---- #

        "COMPACTION_RANDOM": {
            **COMPACTION_BASE,
            "geometry": {
                **COMPACTION_BASE["geometry"],
                "landscape": np.array(["random"]),
                "l" : np.arange(10, 450 + 10, 10, dtype=int),
                "bpmin": np.arange(0, 20 + 5, 5, dtype=int),
            },
            "probas": {
                **COMPACTION_BASE["probas"],
                "alphad": np.array([0.00], dtype=float)
            },
            "meta": {
                **COMPACTION_BASE["meta"],
                "path": f"{PROJECT['project_name']}__compactionrandom"
            }
        },
        
        "COMPACTION_PERIODIC": {
            **COMPACTION_BASE,
            "geometry": {
                **COMPACTION_BASE["geometry"],
                "landscape": np.array(["periodic"]),
                "l" : np.arange(10, 200 + 10, 10, dtype=int),
                "bpmin": np.array([0], dtype=int),
            },
            "probas": {
                **COMPACTION_BASE["probas"],
                "alphad": np.arange(0.00, 1.00 + 0.10, 0.10, dtype=float),
            },
            "meta": {
                **COMPACTION_BASE["meta"],
                "path": f"{PROJECT['project_name']}__compactionperiodic"
            }
        },
        
        # ---- STATIC : TWO STEPS ---- #
        
        "RYU": {
            **TWOSTEPS__BASE,
            "geometry": {
                "landscape": np.array(['homogen', 'periodic', 'random']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "meta": {
                **TWOSTEPS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__ryu"
            }
        },

        # ---- DYNAMIC ---- #
        
        "FACT_PASSIVE_FULL": {
            **TWOSTEPS__BASE,
            "formalism": {**FORMALISMS["alg2_passive_full"]},
            "meta": {
                **TWOSTEPS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__passfull"
            }
        },

        "FACT_PASSIVE_MEMORY": {
            **TWOSTEPS__BASE,
            "formalism": {**FORMALISMS["alg2_passive_memory"]},
            "meta": {
                **TWOSTEPS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__passmemo"
            }
        },

        "FACT_ACTIVE_FULL": {
            **TWOSTEPS__BASE,
            "formalism": {**FORMALISMS["alg2_active_full"]},
            "meta": {
                **TWOSTEPS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__actifull"
            }
        },

        "FACT_ACTIVE_MEMORY": {
            **TWOSTEPS__BASE,
            "formalism": {**FORMALISMS["alg2_active_memory"]},
            "meta": {
                **TWOSTEPS__BASE["meta"],
                "path": f"{PROJECT['project_name']}__actimemo"
            }
        },
        
        # ---- TESTS ---- #
        
        "TEST": {
            **TEST__BASE,
            "meta": {
                **TEST__BASE["meta"],
                "path": f"{PROJECT['project_name']}__test"
            }
        },

        # ---- FIGURES ---- #

        "FIGURE_1": {
            **ONESTEP__BASE,
            "geometry": {
                "landscape": np.array(['random']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                **ONESTEP__BASE["probas"],
                "mu": np.array([160], dtype=int),
                "theta": np.arange(1, 1001, 1, dtype=int)
            },
            "meta": {
                **ONESTEP__BASE["meta"],
                "nt": 10,
                "path": f"{PROJECT['project_name']}__fig1"
            }
        },

        "FIGURE_2": {
            **ONESTEP__BASE,
            "geometry": {
                "landscape": np.array(['homogen']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                **ONESTEP__BASE["probas"],
                "mu": np.array([160]),
                "theta": np.array([20, 200]),
            },
            "meta": {
                **ONESTEP__BASE["meta"],
                "nt": 10_000,
                "path": f"{PROJECT['project_name']}__fig2"
            }
        },

        "FIGURE_3": {
            **ONESTEP__BASE,
            "geometry": {
                "landscape": np.array(['homogen']),
                "s": np.array([150], dtype=int),
                "l": np.array([10], dtype=int),
                "bpmin": np.array([0], dtype=int)
            },
            "probas": {
                **ONESTEP__BASE["probas"],
                "mu": np.array([160]),
                "theta": np.array([20, 200]),
            },
            "meta": {
                **ONESTEP__BASE["meta"],
                "nt": 10_000,
                "path": f"{PROJECT['project_name']}__fig3"
            }
        },
    }
    
    # ---- RETURN ---- #

    if config not in presets:
        raise ValueError(f"Unknown configuration: {config}")

    return {
        **presets[config],
        "project": PROJECT,
        "chromatin": CHROMATIN,
        "time": TIME
    }