"""
nucleo.merging_function
------------------------
Merging parquet files in order to calculate heatmaps.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import os
import pickle
import json

import numpy as np
from pathlib import Path
from tqdm import tqdm

import polars as pl
import polars.selectors as cs
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.1 : Dialog box 


def ask_confirmation_input():
    response = input("Are you sure you want to merge all the parquet files? (Yes/No): ")
    if response != "Yes":
        raise RuntimeError("Stopped by the user.")


# 2.2 : Merging into one file


def merging_parquet_files(root_directory: str = Path.home() / "Documents" / "Workspace" / "nucleo" / "outputs" / "2025-01-01_PSMN", output_name="ncl_output.parquet"):

    ask_confirmation_input()

    root_directory = Path(root_directory)
    parquet_files = list(root_directory.rglob("*.parquet"))

    print(f"Found {len(parquet_files)} parquet files total.")

    dataframes = []
    loaded = 0

    for pq_path in tqdm(parquet_files, desc="Loading parquet files"):

        # Optional filter: only keep simulation outputs
        if "data" not in pq_path.name:
            continue

        try:
            df = pl.read_parquet(pq_path)

            # Keep only useful columns
            df = df.select(cs.numeric() | cs.boolean() | cs.string())

            dataframes.append(df)
            loaded += 1

        except Exception as e:
            print(f"Error reading {pq_path}: {e}")

    print(f"Loaded {loaded} parquet simulation files.")

    if not dataframes:
        raise RuntimeError("No parquet files loaded!")

    merged_df = pl.concat(dataframes, how="vertical")

    output_path = root_directory / output_name
    merged_df.write_parquet(output_path)

    print("Merged file written to:", output_path)
    print("Total rows:", merged_df.shape[0])

    return merged_df


def merging_parquet_lazy(root_directory, output_name="ncl_output.parquet"):

    ask_confirmation_input()

    root_directory = Path(root_directory)

    print("Scanning parquet dataset...")

    df = (
        pl.scan_parquet(
            str(root_directory / "**/*.parquet")
            # extra_columns="ignore"
        )
        .select(cs.numeric() | cs.boolean() | cs.string())
        .collect(engine="streaming")
    )

    output_path = root_directory / output_name
    df.write_parquet(output_path)

    print("Merged file written to:", output_path)
    print("Total rows:", df.shape[0])

    return df


# 2.3 : Getting and ordering configurations


def getting_and_ordering_configurations(data_frame, scenario_path = Path.home() / "Documents" / "Workspace" / "nucleo" / "outputs" / "2025-01-01_PSMN"): 
    """
    We're extracting the different configurations of modeling and ordering them for a proper representation

    Args:
        data_frame (df): filtered data frame

    Returns:
        sorted_combinations_configs: the configurations
    """

    df = data_frame
    filtered_combinations = df.filter(
        ~(
            ((pl.col("land") == 'periodic') | (pl.col("land") == 'homogen')) &
            (pl.col("bpmin") == 5)

            # ((pl.col("alpha_choice") == 'periodic') | (pl.col("alpha_choice") == 'constant_mean')) &
            # (pl.col("bpmin") == 5)
        )
    )

    # Getting the unique combinations of 's', 'l', 'bpmin' and 'land'
    unique_combinations = filtered_combinations.select(['s', 'l', 'bpmin', 'land']).unique()

    # Ordering it by land in priority
    alpha_order = pl.when(pl.col("land") == 'homogeneous').then(1)\
                    .when(pl.col("land") == 'periodic').then(2)\
                    .when(pl.col("land") == 'random').then(3)\
                    .otherwise(4)
    # alpha_order = pl.when(pl.col("alpha_choice") == 'constant_mean').then(1)\
    #                 .when(pl.col("alpha_choice") == 'periodic').then(2)\
    #                 .when(pl.col("alpha_choice") == 'nt_random').then(3)\
    #                 .otherwise(4)
    unique_combinations = unique_combinations.with_columns(
        alpha_order.alias("alpha_order")
    )

    # Ordering by 'alpha_order', then by 'bpmin', and finally by 'l' with l=10 prioritazed
    sorted_combinations = unique_combinations.sort(by=['alpha_order', 'bpmin', 'l'])

    # Suppressing the temporary column 'alpha_order'
    sorted_combinations = sorted_combinations.drop('alpha_order')

    # Convertiing it into a list of dict
    sorted_combinations_configs = sorted_combinations.rows()
    sorted_combinations_configs = [
        {"s": row[0], "l": row[1], "bpmin": row[2], "land": row[3]} 
        for row in sorted_combinations_configs
    ]

    with open("configs.json", "w") as f:
        json.dump(sorted_combinations_configs,f)

    # Print
    for config in sorted_combinations_configs:
        print(config)

    with open(f"{scenario_path}/scenarios.json", "w") as f:
        json.dump(sorted_combinations_configs,f)

    return sorted_combinations_configs


# 2.4 : From main file to heatmaps


def compute_heatmap_data_df(df, config_list, speed_cols, root):
    """
    Calcule les données de heatmap pour chaque config / speed_col, et renvoie
    un DataFrame polars en format LONG (une ligne par point (config, speed_col, theta, mu)).

    Colonnes du DataFrame retourné :
        config_idx, s, l, bpmin, land, speed_col, theta, mu,
        value_raw, value_norm_mu, value_norm_th

    Args:
        df (pl.DataFrame): données fusionnées (issues de merging_parquet_lazy par ex.)
        config_list (list[dict]): liste des configs (sorted_combinations_configs)
        speed_cols (list[str]): colonnes de vitesse à traiter
        root (str | Path): dossier où écrire le parquet de sortie

    Returns:
        pl.DataFrame: le DataFrame en format long, déjà écrit sur disque en parquet
    """

    rows = []

    for idx, config in tqdm(enumerate(config_list), total=len(config_list), desc="Heatmaps (DF)"):

        df_f = df.filter(
            (pl.col("s") == config["s"]) &
            (pl.col("l") == config["l"]) &
            (pl.col("bpmin") == config["bpmin"]) &
            (pl.col("land") == config["land"])
        )

        if df_f.is_empty():
            continue

        mu_values = (
            df_f.select("mu").unique().sort("mu").to_series().to_list()
        )
        theta_values = (
            df_f.select("theta").unique().sort("theta").to_series().to_list()
        )

        alphaf = df_f["alphaf"][0]
        alphao = df_f["alphao"][0]
        s = df_f["s"][0]
        l = df_f["l"][0]
        prefactor_th = (alphaf * l + alphao * s) / (l + s)

        grouped = (
            df_f.group_by(["theta", "mu"])
            .agg([pl.col(c).mean().alias(c) for c in speed_cols])
        )

        for speed_col in speed_cols:

            pivot = grouped.pivot(
                index="theta",
                on="mu",
                values=speed_col,
                aggregate_function=None,
            )

            mu_cols = [str(m) for m in mu_values]
            pivot = pivot.sort("theta").select(["theta"] + mu_cols)

            Z = pivot.drop("theta").to_numpy()
            Z = np.nan_to_num(Z, nan=0.0)

            if speed_col in {"v_mean", "vi_med", "vi_mp", "vf"}:
                MU = np.array(mu_values)[None, :]
                VTH = prefactor_th * MU
                Z_raw, Z_nmu, Z_nth = Z, Z / MU, Z / VTH
            else:
                Z_raw = Z_nmu = Z_nth = Z

            # On "déplie" la matrice theta x mu en lignes individuelles
            for i_theta, theta in enumerate(theta_values):
                for i_mu, mu in enumerate(mu_values):
                    rows.append({
                        "config_idx": idx,
                        "s": config["s"],
                        "l": config["l"],
                        "bpmin": config["bpmin"],
                        "land": config["land"],
                        "speed_col": speed_col,
                        "theta": theta,
                        "mu": mu,
                        "value_raw": float(Z_raw[i_theta, i_mu]),
                        "value_norm_mu": float(Z_nmu[i_theta, i_mu]),
                        "value_norm_th": float(Z_nth[i_theta, i_mu]),
                    })

    result_df = pl.DataFrame(rows)

    output_path = os.path.join(root, "ncl_heatmaps.parquet")
    result_df.write_parquet(output_path)

    print("Heatmaps (format long) écrites dans :", output_path)
    print("Lignes :", result_df.shape[0])

    return result_df


def get_heatmap_matrix(heatmaps_df, config_idx, speed_col, value_col="value_raw"):
    """
    Repivote une sous-partie du DataFrame long en matrice 2D (theta x mu) pour le plot.

    Args:
        heatmaps_df (pl.DataFrame): sortie de compute_heatmap_data_df (ou lue depuis le parquet)
        config_idx (int): index de la configuration voulue
        speed_col (str): colonne de vitesse voulue
        value_col (str): "value_raw", "value_norm_mu" ou "value_norm_th"

    Returns:
        Z (np.ndarray): matrice theta x mu
        theta_values (np.ndarray)
        mu_values (np.ndarray)
    """
    sub = heatmaps_df.filter(
        (pl.col("config_idx") == config_idx) & (pl.col("speed_col") == speed_col)
    )

    pivot = sub.pivot(index="theta", on="mu", values=value_col).sort("theta")

    theta_values = pivot["theta"].to_numpy()
    mu_values = np.array([float(c) for c in pivot.columns if c != "theta"])

    Z = pivot.drop("theta").to_numpy()

    return Z, theta_values, mu_values


# ─────────────────────────────────────────────
# 3 : Call
# ─────────────────────────────────────────────

root = Path("/home/nicolas/Documents/Workspace/nucleo/outputs/NUCLEO__PSMN__2026-07-09")
merged_df = merging_parquet_lazy(root)
sorted_combinations_configs = getting_and_ordering_configurations(merged_df, root)
speed_columns = ['v_mean', 'vi_med', 'vi_mp', 'vf', 'wf']
heatmaps_df = compute_heatmap_data_df(merged_df, sorted_combinations_configs, speed_columns, root)

#.