"""
nucleo.reading_functions
------------------------
Reading results of simulations, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import polars as pl
from pathlib import Path
import pickle


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


def getting_main_file_with_verifications(
    df: pl.DataFrame,
    nt: int = 10000,
    tmax: int = 100,
    dt: int = 1,
    alphao: float = 0.0,
    alphaf: float = 1.0,
    beta: float = 0.0,
    Lmin: int = 0,
    Lmax: int = 50_000,
    origin: int = 10_000,
    bps: int = 1
) -> pl.DataFrame:
    """
    Filters a Polars DataFrame based on specific criteria.

    Args:
        df (pl.DataFrame): The input DataFrame containing the dataset.
        nt (int, default=1000): Number of time steps in the simulation.
        tmax (int, default=100000): Maximum simulation time.
        dt (int): Time step interval.
        alphao (float, default=0.5): Initial alpha value.
        alphaf (float, default=0.5): Final alpha value.
        beta (float, default=1.0): Beta value used in the simulation.
        Lmin (int, default=0): Minimum linker length.
        Lmax (int, default=500): Maximum linker length.
        origin (int, default=0): Origin value used in the setup.
        bps (int, default=10): Base pairs per step.

    Returns:
        pl.DataFrame: The filtered DataFrame based on the given parameters.
    """

    selected_columns = {
        "alpha_choice", "s", "l", "bpmin", 
        "mu", "theta", 
        "nt", "tmax", "dt", "times", 
        "alphao", "alphaf", "beta",
        "Lmin", "Lmax", "origin", "bps",
        
        "v_mean", "v_med", 
        "vf", "Cf", "wf", "vf_std", "Cf_std", "wf_std", 
        "vi_mean", "vi_med", "vi_mp",
    }

    # Select only the required columns
    filtered_columns = [col for col in df.columns if col in selected_columns]
    filtered_df = df.select(filtered_columns)

    # # Verify that all rows have 's' equal to 150
    # if (filtered_df["s"] == 150).all():
    #     print("All rows have s = 150.")
    # else:
    #     print("Some rows do not have s = 150.")
    #     print(filtered_df.filter(pl.col("s") != 150))

    # Apply filtering based on predefined conditions
    filtered_df = (
        filtered_df
        .filter(pl.col("nt") == nt)
        .filter(pl.col("tmax") == tmax)
        .filter(pl.col("dt") == dt)
        .filter(pl.col("alphaf") == alphaf)
        .filter(pl.col("alphao") == alphao)
        .filter(pl.col("beta") == beta)
        .filter(pl.col("Lmin") == Lmin)
        .filter(pl.col("Lmax") == Lmax)
        .filter(pl.col("origin") == origin)
        .filter(pl.col("bps") == bps)
    )

    return filtered_df


def reading_heatmap_one_config(
    config,
    data_type,
    root=Path.home() / "Documents" / "Workspace" / "nucleo" / "outputs" / "2025-01-01_PSMN"
):
    """
    Reads heatmap data for one specific configuration.

    Args:
        config (dict): must contain 's', 'l', 'bpmin', 'alpha_choice' or 'landscape'
        data_type (str): one of ["full_data", "heatmap_raw", "heatmap_mu", "heatmap_th"]
        root (Path): directory containing parquet + heatmap pickle files

    Returns:
        df_main (pl.DataFrame | None)
        config_data (dict)
    """

    # ─────────────────────────────────────────────
    # 1 : Validate data_type
    # ─────────────────────────────────────────────

    valid_types = ["full_data", "heatmap_raw", "heatmap_mu", "heatmap_th"]
    if data_type not in valid_types:
        raise ValueError(f"Incorrect data_type: {data_type}. Must be in {valid_types}")

    # ─────────────────────────────────────────────
    # 2 : Load full simulation data (optional)
    # ─────────────────────────────────────────────

    df_main = None

    if data_type == "full_data":
        main_file_path = root / "ncl_output.parquet"

        df_polars = (
            pl.scan_parquet(str(main_file_path), extra_columns="ignore")
            .collect()
        )

        if "getting_main_file_with_verifications" in globals():
            df_main = getting_main_file_with_verifications(df_polars)
        else:
            df_main = df_polars

        print(
            "Full simulation dataframe loaded:\n"
            f"Shape: {df_main.shape}\n"
            f"Preview:\n{df_main.head(5)}"
        )

    # ─────────────────────────────────────────────
    # 3 : Select heatmap file
    # ─────────────────────────────────────────────

    heatmap_files = {
        "heatmap_raw": "ncl_hm_raw.pkl",
        "heatmap_mu": "ncl_hm_nmu.pkl",
        "heatmap_th": "ncl_hm_nth.pkl",
    }

    if data_type == "full_data":
        # default heatmap
        heatmap_file = "ncl_hm_raw.pkl"
    else:
        heatmap_file = heatmap_files[data_type]

    with open(root / heatmap_file, "rb") as f:
        computed_data = pickle.load(f)

    # ─────────────────────────────────────────────
    # 4 : Remap keys for readability
    # ─────────────────────────────────────────────

    new_computed_data = {}

    for _, cfg_data in computed_data.items():
        cfg = cfg_data["config"]

        alpha = cfg.get("landscape", cfg.get("alpha_choice", "unknown"))
        s = cfg["s"]
        l = cfg["l"]
        bpmin = cfg["bpmin"]

        new_key = f"s{s}_l{l}_bp{bpmin}_{alpha}"
        new_computed_data[new_key] = cfg_data

    computed_data = new_computed_data

    # ─────────────────────────────────────────────
    # 5 : Build configuration key
    # ─────────────────────────────────────────────

    key_alpha = config.get("landscape", config.get("alpha_choice", "unknown"))
    my_key = f"s{config['s']}_l{config['l']}_bp{config['bpmin']}_{key_alpha}"

    if my_key not in computed_data:
        raise KeyError(
            f"No heatmap data for key {my_key}\n"
            f"Available keys: {list(computed_data.keys())}"
        )

    config_data = computed_data[my_key]

    return df_main, config_data