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


def read_parquet_safe(path):
    """
    Reads a parquet file and keeps only numeric, boolean, or string columns.
    Returns None on error.
    """
    try:
        df = pl.read_parquet(path)
        df = df.select(cs.numeric() | cs.boolean() | cs.string())
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def merging_parquet_files_parallel(root_directory: str, output_name="ncl_output.parquet", n_workers=10):

    ask_confirmation_input()

    root_directory = Path(root_directory)

    # Recursively find all parquet files
    parquet_files = [f for f in root_directory.rglob("*.parquet") if "data" in f.name]
    print(f"Found {len(parquet_files)} parquet files to merge.")

    dataframes = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(read_parquet_safe, f): f for f in parquet_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading parquets"):
            df = future.result()
            if df is not None:
                dataframes.append(df)

    if not dataframes:
        raise RuntimeError("No parquet files loaded!")

    # Concatenate vertically
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


def compute_heatmap_data(df: pl.DataFrame, config_list: list, speed_cols: list, root: str) -> dict:
    """
    Computes and stores heatmap data for each configuration and speed column.
    Linear speeds are getting normalized by the theoretical values of constant_mean scenario.

    Args:
        df (pl.DataFrame): Data containing multiple speed configurations.
        config_list (list): List of configuration dictionaries.
        speed_cols (list): List of speed columns to process.

    Returns:
        dict: Nested dictionary {config_idx -> {speed_col -> heatmap_data, config_metadata}}.
    """

    all_data_raw = {}
    all_data_norm_mu = {}
    all_data_norm_th = {}

    for idx, config in tqdm(enumerate(config_list), total=len(config_list), desc="Computing heatmaps"):

        df_filtered = df.filter(
            (pl.col('s') == config['s']) &
            (pl.col('l') == config['l']) &
            (pl.col('bpmin') == config['bpmin']) &
            (pl.col('land') == config['land'])
        )

        if df_filtered.is_empty():
            print(f"No data for config {config}")
            continue

        mu_values = df_filtered['mu'].unique().sort()
        theta_values = df_filtered['theta'].unique().sort()

        # Store configuration metadata (init both dicts)
        meta = {
            "mu_values": mu_values,
            "theta_values": theta_values,
            "config": config
        }

        all_data_raw[idx] = dict(meta)
        all_data_norm_mu[idx] = dict(meta)
        all_data_norm_th[idx] = dict(meta)

        # Loop on data
        for speed_col in speed_cols:
            heatmap_raw = []
            heatmap_norm_mu = []
            heatmap_norm_th = []

            for theta in tqdm(theta_values, desc=f"Processing theta for config {idx}", leave=False):
                data_raw = []
                data_norm_mu = []
                data_norm_th = []

                for mu in mu_values:
                    values = df_filtered.filter((pl.col('mu') == mu) & (pl.col('theta') == theta))[speed_col].mean()
                    norm_mu = mu
                    norm_th = ((df_filtered['alphaf'][0] * df_filtered['l'][0] +
                                df_filtered['alphao'][0] * df_filtered['s'][0]) /
                                (df_filtered['l'][0] + df_filtered['s'][0])) * mu

                    if speed_col in {'v_mean', 'vi_med', 'vi_mp', 'vf'}:
                        data_raw.append(values if values is not None else 0)
                        data_norm_mu.append(values / norm_mu if values is not None else 0)
                        data_norm_th.append(values / norm_th if values is not None else 0)
                       
                    elif speed_col in {'Cf', 'wf'}:
                        data_raw.append(values if values is not None else 0)
                        data_norm_mu.append(values if values is not None else 0)
                        data_norm_th.append(values if values is not None else 0)

                heatmap_raw.append(data_raw)
                heatmap_norm_mu.append(data_norm_mu)
                heatmap_norm_th.append(data_norm_th)

            all_data_raw[idx][speed_col] = np.array(heatmap_raw)
            all_data_norm_mu[idx][speed_col] = np.array(heatmap_norm_mu)
            all_data_norm_th[idx][speed_col] = np.array(heatmap_norm_th)

    # Saving datas
    path_raw  = os.path.join(root, "ncl_hm_raw.pkl")
    path_norm_mu = os.path.join(root, "ncl_hm_nmu.pkl")
    path_norm_th = os.path.join(root, "ncl_hm_nth.pkl")

    with open(path_raw, "wb") as f:
        pickle.dump(all_data_raw, f)
    with open(path_norm_mu, "wb") as f:
        pickle.dump(all_data_norm_mu, f)
    with open(path_norm_th, "wb") as f:
        pickle.dump(all_data_norm_th, f)

    print("Files written at :", path_raw, "-", path_norm_mu, "-", path_norm_th)
    return None


def compute_heatmap_data_fast(df, config_list, speed_cols, root):

    all_data_raw = {}
    all_data_norm_mu = {}
    all_data_norm_th = {}

    for idx, config in tqdm(enumerate(config_list),
                            total=len(config_list),
                            desc="Heatmaps (FAST)"):

        # --- Filter config
        df_f = df.filter(
            (pl.col("s") == config["s"]) &
            (pl.col("l") == config["l"]) &
            (pl.col("bpmin") == config["bpmin"]) &
            (pl.col("land") == config["land"])
        )

        if df_f.is_empty():
            continue

        # --- Unique sorted axes
        mu_values = (
            df_f.select("mu")
            .unique()
            .sort("mu")
            .to_series()
            .to_list()
        )

        theta_values = (
            df_f.select("theta")
            .unique()
            .sort("theta")
            .to_series()
            .to_list()
        )

        # --- Theoretical normalization constant
        alphaf = df_f["alphaf"][0]
        alphao = df_f["alphao"][0]
        s = df_f["s"][0]
        l = df_f["l"][0]

        prefactor_th = (alphaf * l + alphao * s) / (l + s)

        # --- Metadata
        meta = {
            "mu_values": np.array(mu_values),
            "theta_values": np.array(theta_values),
            "config": config
        }

        all_data_raw[idx] = dict(meta)
        all_data_norm_mu[idx] = dict(meta)
        all_data_norm_th[idx] = dict(meta)

        # =====================================================
        # Core: group_by(theta, mu) once for all speed columns
        # =====================================================

        grouped = (
            df_f.group_by(["theta", "mu"])
            .agg([pl.col(c).mean().alias(c) for c in speed_cols])
        )

        # --- Loop speed columns (cheap)
        for speed_col in speed_cols:

            # Pivot => matrix theta × mu
            pivot = grouped.pivot(
                index="theta",
                on="mu",
                values=speed_col,
                aggregate_function=None
            )

            # Force correct ordering
            mu_cols = [str(m) for m in mu_values]

            pivot = (
                pivot.sort("theta")
                .select(["theta"] + mu_cols)
            )


            # Convert to numpy matrix
            Z = pivot.drop("theta").to_numpy()
            Z = np.nan_to_num(Z, nan=0.0)

            # --- Normalisations
            if speed_col in {"v_mean", "vi_med", "vi_mp", "vf"}:

                MU = np.array(mu_values)[None, :]
                VTH = prefactor_th * MU

                Z_raw = Z
                Z_nmu = Z / MU
                Z_nth = Z / VTH

            else:
                # Cf, wf : no normalization
                Z_raw = Z
                Z_nmu = Z
                Z_nth = Z

            # Store
            all_data_raw[idx][speed_col] = Z_raw
            all_data_norm_mu[idx][speed_col] = Z_nmu
            all_data_norm_th[idx][speed_col] = Z_nth

    # =====================================================
    # Save pickle files
    # =====================================================

    path_raw = os.path.join(root, "ncl_hm_raw.pkl")
    path_nmu = os.path.join(root, "ncl_hm_nmu.pkl")
    path_nth = os.path.join(root, "ncl_hm_nth.pkl")

    with open(path_raw, "wb") as f:
        pickle.dump(all_data_raw, f)

    with open(path_nmu, "wb") as f:
        pickle.dump(all_data_norm_mu, f)

    with open(path_nth, "wb") as f:
        pickle.dump(all_data_norm_th, f)

    print("Heatmaps written to:")
    print(" -", path_raw)
    print(" -", path_nmu)
    print(" -", path_nth)

    return None


# ─────────────────────────────────────────────
# 3 : Call
# ─────────────────────────────────────────────

# 3.0 : Root
# root = Path("/home/nicolas/Documents/Workspace/nucleo/outputs/NUCLEO__PSMN__2026-05-01")
root = Path("/home/nicolas/Documents/Workspace/nucleo/outputs/FACT__PSMN__2026-05-21")
root_parquet = root / "ncl_output.parquet"

# # 3.1 : Merging
merged_df = merging_parquet_lazy(root)
print(merged_df.head(12))

# 3.2 : Configurations    
print('\nConfigurations :')
sorted_combinations_configs = getting_and_ordering_configurations(merged_df, root)

# 3.3 : Comuting heatmaps
speed_columns = ['v_mean', 'vi_med', 'vi_mp', 'vf', 'wf']
compute_heatmap_data_fast(merged_df, sorted_combinations_configs, speed_columns, root)
print("Heatmaps computed")

