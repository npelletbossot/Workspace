"""
nucleo.plot_functions
------------------------
Plot functions for writing results, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


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

        missing_mask = np.isnan(np.asarray(data, dtype=float))

        if plot_log2:
            with np.errstate(divide="ignore", invalid="ignore"):
                data_to_plot = np.log2(data, dtype=float)
        else:
            data_to_plot = data

        # ─────────────────────────────────────────────
        # Gestion des valeurs divergentes (points blancs)
        # ─────────────────────────────────────────────
        data_to_plot = np.nan_to_num(data_to_plot, nan=vmin, posinf=vmax, neginf=vmin)
        data_to_plot = np.clip(data_to_plot, vmin, vmax)


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

    ax.set_xticks(np.arange(100,601,100))
    ax.set_yticks(np.arange(20,101,20))

    return c



def get_heatmap_matrix(heatmaps_df, config_idx, speed_col, value_col="value_raw"):
    """
    Repivote une sous-partie du DataFrame long (issu de compute_heatmap_data_df)
    en matrice 2D (theta x mu).

    Ne fait QUE de l'extraction / pivot. Aucune logique de plotting ici --
    c'est le seul endroit où le DataFrame est lu, pour que plot_single_heatmap
    et plot_all_heatmaps n'aient jamais à connaître le format du parquet.

    Args:
        heatmaps_df (pl.DataFrame): le DataFrame long (lu depuis ncl_heatmaps.parquet)
        config_idx (int): index de la configuration voulue
        speed_col (str): colonne de vitesse voulue ('v_mean', 'wf', ...)
        value_col (str): "value_raw", "value_norm_mu" ou "value_norm_th"

    Returns:
        Z (np.ndarray): matrice theta x mu
        theta_values (np.ndarray)
        mu_values (np.ndarray)
        config (dict): metadata de la config (s, l, bpmin, land)
    """
    sub = heatmaps_df.filter(
        (pl.col("config_idx") == config_idx) & (pl.col("speed_col") == speed_col)
    )

    if sub.is_empty():
        raise ValueError(f"Pas de données pour config_idx={config_idx}, speed_col={speed_col}")

    pivot = sub.pivot(index="theta", on="mu", values=value_col).sort("theta")

    theta_values = pivot["theta"].to_numpy()
    mu_values = np.array([float(c) for c in pivot.columns if c != "theta"])
    Z = pivot.drop("theta").to_numpy()

    row0 = sub.row(0, named=True)
    config = {"s": row0["s"], "l": row0["l"], "bpmin": row0["bpmin"], "land": row0["land"]}

    return Z, theta_values, mu_values, config


def _value_col_from_type(type_of_data):
    if type_of_data not in ["raw", "norm_mu", "norm_th"]:
        raise ValueError(f"type_of_data not in : ['raw', 'norm_mu', 'norm_th'] got {type_of_data}")
    return f"value_{type_of_data}"



def plot_one_heatmap_from_df(
    heatmaps_df,
    config_idx,
    speed_col,
    ax=None,
    type_of_data="raw",
    plot_log2=False,
    dashed_line=True,
    title=True,
):
    """
    Plot UNE heatmap (une config, un speed_col) à partir du DataFrame long.

    C'est la seule fonction qui fait le pont entre le format DataFrame et
    plot_single_heatmap : elle extrait la matrice via get_heatmap_matrix,
    puis appelle plot_single_heatmap avec les bons arguments. Aussi bien
    plot_all_heatmaps que l'usage "une seule heatmap" passent par ici --
    donc toute la logique d'affichage (vmin/vmax/cmap/etc.) reste à un seul
    endroit, dans plot_single_heatmap.

    Args:
        heatmaps_df (pl.DataFrame): DataFrame long, lu depuis ncl_heatmaps.parquet
        config_idx (int): index de la config voulue
        speed_col (str): colonne de vitesse voulue
        ax (plt.Axes | None): axe sur lequel dessiner ; si None, en crée un
        type_of_data ("raw" | "norm_mu" | "norm_th")
        plot_log2 (bool)
        dashed_line (bool)
        title (bool)

    Returns:
        ax (plt.Axes), c (le mappable retourné par pcolormesh, pour colorbar manuelle si besoin)
    """
    value_col = _value_col_from_type(type_of_data)

    Z, theta_values, mu_values, config = get_heatmap_matrix(
        heatmaps_df, config_idx=config_idx, speed_col=speed_col, value_col=value_col
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=150)

    c = plot_single_heatmap(
        ax=ax,
        mu_values=mu_values,
        theta_values=theta_values,
        data=Z,
        speed_col=speed_col,
        config=config,
        type_of_data=type_of_data,
        plot_log2=plot_log2,
        dashed_line=dashed_line,
        title=title,
    )

    return ax, c


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

    Ne fait QUE charger le parquet, créer la grille d'axes, et boucler en
    appelant plot_one_heatmap_from_df pour chaque (config, speed_col).
    Toute la logique de style et toute la logique d'extraction vivent
    ailleurs (plot_single_heatmap et get_heatmap_matrix respectivement) --
    cette fonction n'est qu'un orchestrateur.
    """
    _value_col_from_type(type_of_data)  # valide type_of_data tôt

    heatmaps_df = pl.read_parquet(Path(root) / "ncl_heatmaps.parquet")

    config_indices = sorted(heatmaps_df["config_idx"].unique().to_list())
    n_combinations = len(config_indices)

    fig, axes = plt.subplots(nrows=n_combinations, ncols=len(speed_cols), figsize=(18, 4 * n_combinations), dpi=400)
    axes = np.atleast_2d(axes)

    for row_idx, config_idx in enumerate(tqdm(config_indices, desc="Plotting heatmaps")):
        for col_idx, speed_col in enumerate(speed_cols):
            plot_one_heatmap_from_df(
                heatmaps_df,
                config_idx=config_idx,
                speed_col=speed_col,
                ax=axes[row_idx, col_idx],
                type_of_data=type_of_data,
                plot_log2=plot_log2,
                dashed_line=dashed_line,
            )

    plt.tight_layout()
    plt.show()
    return fig, axes