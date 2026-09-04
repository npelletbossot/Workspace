"""
nucleo.launching_functions
------------------------
Launching functions for simulations.
"""


# ─────────────────────────────────────────────
# 1 : Package
# ─────────────────────────────────────────────

from nucleo.core.launching import main
import multiprocessing


# ─────────────────────────────────────────────
# 2 : Call
# ─────────────────────────────────────────────

# # Options : 
# CONFIG = {
#     "NUCLEO": ["NU", "BP", "LSLOW", "LSHIGH"],
#     "COMPACTION": ["COMPACTION_RANDOM", "COMPACTION_PERIODIC"],
#     "RYU": ["TWO_STEPS"],
#     "FACT": ["FACT_PASSIVE_FULL", "FACT_PASSIVE_MEMORY", "FACT_ACTIVE_FULL", "FACT_ACTIVE_MEMORY"],
#     "FIGURES": ["FIGURE_1", "FIGURE_2", "FIGURE_3"],
#     "TESTS": ["TEST_1S", "TEST_2S"]
# }

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main(
        STUDY = "TESTS"
)
