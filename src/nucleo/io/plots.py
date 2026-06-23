"""
nucleo.plot_functions
------------------------
Plot functions for writing results, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import pickle
from tqdm import tqdm


# ─────────────────────────────────────────────
# 2 : Functions : 1D Plots
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# Labels LaTeX pour chaque speed_col (titre des subplots)
# ─────────────────────────────────────────────────────────────────────────

SPEED_COL_LABELS = {
    "vf": r"$v_{init}$",
    "wf": r"$w_f$",
    "v_mean": r"$v_{mean}$",
}


def speed_col_label(speed_col):
    """Renvoie le label LaTeX d'un speed_col, avec un fallback générique."""
    return SPEED_COL_LABELS.get(speed_col, rf"${speed_col}$")


def plot_single_heatmap(
    ax,
    mu_values,
    theta_values,
    data,
    speed_col,
    config,
    type_of_data="raw",
    plot_log2=False,
    dashed_line=True,
    title=True
):
    """
    Plot a single heatmap inside a given axis.

    type_of_data must be one of ["raw", "norm_mu", "norm_th"] -- this is the
    SAME vocabulary used by reading_heatmap_one_config, so the value you
    pass to the reader and the value you pass here should always match
    (e.g. read with type_of_data="norm_th" -> plot with type_of_data="norm_th").

    Toute la logique (choix de vmin/vmax/cmap/titre de colorbar selon
    type_of_data et plot_log2, cas spécial wf, ligne théorique pointillée)
    vit ici. plot_all_heatmaps ne fait que charger les données et appeler
    cette fonction en boucle.
    """

    # ─────────────────────────────────────────────
    # Convert everything to numpy
    # ─────────────────────────────────────────────

    mu_values = np.asarray(mu_values)
    theta_values = np.asarray(theta_values)
    data = np.asarray(data)

    # ─────────────────────────────────────────────
    # Safety check on dimensions
    # ─────────────────────────────────────────────

    expected_shape = (len(theta_values), len(mu_values))

    if data.shape != expected_shape:
        raise ValueError(
            f"\nHeatmap dimension mismatch\n"
            f"Expected data shape : {expected_shape}\n"
            f"Received data shape : {data.shape}\n"
            f"len(theta_values) = {len(theta_values)}\n"
            f"len(mu_values) = {len(mu_values)}"
        )

    # ─────────────────────────────────────────────
    # type_of_data -> vmin / vmax / title_bar
    # ─────────────────────────────────────────────

    if plot_log2:
        title_bar_mini = "log₂ ("
    else:
        title_bar_mini = ""

    if type_of_data not in ["raw", "norm_mu", "norm_th"]:
        raise ValueError(f"type_of_data not in : ['raw', 'norm_mu', 'norm_th'] got {type_of_data}")

    elif type_of_data == "raw":
        title_bar = title_bar_mini + "v"
        if plot_log2:
            vmin, vmax = -2, 10
        else:
            vmin, vmax = 0, 50

    elif type_of_data == "norm_mu":
        title_bar = title_bar_mini + f"{speed_col_label(speed_col)} / mean value)"
        if plot_log2:
            vmin, vmax = -1, 0.010
        else:
            vmin, vmax = 0, 0.50

    elif type_of_data == "norm_th":
        title_bar = title_bar_mini + rf"{speed_col_label(speed_col)} " + "/ $v_{hom}$)"
        if plot_log2:
            vmin, vmax = -2, 2
        else:
            vmin, vmax = 0, 2

    # ─────────────────────────────────────────────
    # Special case wf
    # ─────────────────────────────────────────────

    if speed_col == "wf":

        cmap = "jet"
        wmin, wmax = 0, 1
        data_to_plot = data

        c = ax.pcolormesh(
            mu_values,
            theta_values,
            data_to_plot,
            cmap=cmap,
            vmin=wmin,
            vmax=wmax,
            shading="auto"
        )

    # ─────────────────────────────────────────────
    # Other variables
    # ─────────────────────────────────────────────

    else:

        cmap = "bwr"

        # Les nan déjà présents dans `data` AVANT le log2 = points non
        # simulés / donnée absente. On les retient ici pour pouvoir les
        # distinguer plus bas des inf/nan qui apparaissent À CAUSE du log2
        # (valeur nulle, négative, ou juste très petite/grande).
        missing_mask = np.isnan(np.asarray(data, dtype=float))

        if plot_log2:
            # np.errstate pour ne pas spammer la console avec les
            # RuntimeWarning de log2(0) et log2(négatif), qu'on gère
            # explicitement juste après.
            with np.errstate(divide="ignore", invalid="ignore"):
                data_to_plot = np.log2(data, dtype=float)
        else:
            data_to_plot = data

        # ─────────────────────────────────────────────
        # Gestion des valeurs divergentes (points blancs)
        # ─────────────────────────────────────────────
        # log2(0) -> -inf, log2(négatif) -> nan. pcolormesh affiche en blanc
        # tout ce qui sort de [vmin, vmax] OU qui est nan/inf -- on clippe
        # donc explicitement aux bornes : les valeurs très petites
        # (log2 -> -inf) sont rabattues à vmin (couleur extrême "froide" de
        # la bwr), pas mises à 0 qui n'a pas de sens particulier ici.
        data_to_plot = np.nan_to_num(data_to_plot, nan=vmin, posinf=vmax, neginf=vmin)
        data_to_plot = np.clip(data_to_plot, vmin, vmax)

        # Les vrais points manquants (nan d'origine, donnée absente) sont
        # forcés à vmin pour être colorés en bleu plutôt qu'en blanc, comme
        # demandé -- contrairement aux divergences du log2 ci-dessus, ce
        # n'est pas un effet de bord du clip mais un choix explicite ici.
        data_to_plot[missing_mask] = vmin

        c = ax.pcolormesh(
            mu_values,
            theta_values,
            data_to_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto"
        )

        cbar = plt.colorbar(c, ax=ax)
        cbar.set_label(title_bar)

        # ─────────────────────────────────────────────
        # Théorie v_mp
        # ─────────────────────────────────────────────

        s = config["s"]
        l = config["l"]

        MU, THETA = np.meshgrid(mu_values, theta_values)

        vmp_th = (MU - (THETA**2) / MU) / (l + s)

        if dashed_line:

            levels = [1, 2, 3]

            cs = ax.contour(
                mu_values,
                theta_values,
                vmp_th,
                levels=levels,
                colors="black",
                linestyles="dotted",
                linewidths=1.5
            )

            ax.clabel(cs, inline=True, fontsize=10, fmt="%.1f")

    # ─────────────────────────────────────────────
    # Labels
    # ─────────────────────────────────────────────

    if title:
        ax.set_title(f"{config}")
    ax.set_xlabel("$\\mu(\\sigma)$")
    ax.set_ylabel("$\\theta(\\sigma)$")

    return c


def plot_all_heatmaps(
    speed_cols,
    root=Path.home() / "Documents" / "Workspace" / "nucleo" / "outputs" / "2025-01-01_PSMN",
    type_of_data="raw",
    plot_log2=False,
    dashed_line=True,
):
    """
    Plots heatmaps of either raw or log2-transformed values depending on the variable:
    - log2 + bwr for all except wf
    - linear + jet for wf

    Data are already normalized by the theoretical values of the constant_value scenario !

    Ne fait que charger les données et boucler sur plot_single_heatmap, qui
    porte toute la logique de style (vmin/vmax/cmap/titre/ligne théorique).
    """

    if type_of_data not in ["raw", "norm_mu", "norm_th"]:
        raise ValueError(f"type_of_data not in : ['raw', 'norm_mu', 'norm_th'] got {type_of_data}")

    elif type_of_data == "raw":
        main_file_path = root / "ncl_hm_raw.pkl"

    elif type_of_data == "norm_mu":
        main_file_path = root / "ncl_hm_nmu.pkl"

    elif type_of_data == "norm_th":
        main_file_path = root / "ncl_hm_nth.pkl"

    with open(main_file_path, "rb") as f:
        computed_data = pickle.load(f)

    n_combinations = len(computed_data)
    fig, axes = plt.subplots(nrows=n_combinations, ncols=len(speed_cols), figsize=(18, 4 * n_combinations), dpi=400)
    axes = np.atleast_2d(axes)

    for idx, (key, config_data) in enumerate(tqdm(computed_data.items(), total=n_combinations, desc="Plotting heatmaps")):
        mu_values = config_data["mu_values"]
        theta_values = config_data["theta_values"]
        config = config_data["config"]

        for col_idx, speed_col in enumerate(speed_cols):
            ax = axes[idx, col_idx]
            data = config_data[speed_col]

            plot_single_heatmap(
                ax=ax,
                mu_values=mu_values,
                theta_values=theta_values,
                data=data,
                speed_col=speed_col,
                config=config,
                type_of_data=type_of_data,
                plot_log2=plot_log2,
                dashed_line=dashed_line,
            )

    plt.tight_layout()
    plt.show()
    return fig, axes