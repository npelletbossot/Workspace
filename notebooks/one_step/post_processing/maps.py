"""
map_analysis
------------------------
Plotting Heatmaps of Velocities.
"""

# ─────────────────────────────────────────────
# 1 : Python Version + Librairies
# ────────────────────────────────────────────

import sys, site
print("Python:", sys.executable)
print("User site:", site.getusersitepackages())
print("Site-packages:", site.getsitepackages() if hasattr(site, "getsitepackages") else "n/a")

import numpy as np
import polars as pl
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

from nucleo.io.reading import reading_heatmap_one_config
from nucleo.io.plots import plot_single_heatmap, plot_all_heatmaps

plt.rcParams["font.size"] = 12

# ─────────────────────────────────────────────
# 2 : Paths + Configuration
# ────────────────────────────────────────────

root = Path.home() / "Documents" / "Workspace" / "nucleo" / "PSMN" / "outputs" / "2026-03-09__PSMN"

config_to_plot = {
    "s": 150,
    "l": 10,
    "bpmin": 0,
    "alpha_choice": "periodic"
}

data_to_plot = "heatmap_raw"

# ─────────────────────────────────────────────
# 3 : Lecture des données
# ────────────────────────────────────────────

df_main, config_data = reading_heatmap_one_config(
    config=config_to_plot,
    data_type=data_to_plot,
    root=root)

# Data 
mu_values = config_data["mu_values"]
theta_values = config_data["theta_values"]
data = config_data["v_mean"]
config = config_data["config"]

# ─────────────────────────────────────────────
# 4 : Single Heatmap
# ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6,5), dpi=300)
plot_single_heatmap(
    ax=ax,
    mu_values=mu_values,
    theta_values=theta_values,
    data=data,
    speed_col="v_mean",
    config=config,
    plot_log2=False,
    vmin=0,
    vmax=50,
    title_bar="v"
)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 5 : Full Heatmaps
# ────────────────────────────────────────────

plot_all_heatmaps(
    speed_cols=["v_mean", "vi_med", "wf"],
    root=root,
    type_of_data="raw",
    plot_log2=False
)