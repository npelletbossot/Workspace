"""
nucleo.reading_functions
------------------------
Reading results of simulations, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import pickle
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


# 2.0 PARAMS


PARAMS = [
    "algo", "fact", "mode",  "dstr",
    "land", "s", "l", "bpmin",
    "mu", "theta",
    "alphar", "Kp", "Kz",
    "alphaf", "alphao", 
    "beta", "alphac", 
    "alphad",
    "rcapt", "rrest"
]


_SCALAR_TYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
    pl.Boolean, pl.String,
)


# 2.1 All files


def checking_parameters(
    df: pl.DataFrame,
    params: list[str] = PARAMS
) -> dict[str, list]:
    """Retourne les valeurs uniques des paramètres présents dans le DataFrame."""

    param_dict = {}

    for param in params:
        if param in df.columns:
            param_dict[param] = df[param].unique().sort().to_list()

    return param_dict


def reading_one_parquet(root: str | Path) -> pl.DataFrame:
    return pl.read_parquet(Path(root))


def finding_one_parquet(root: str, params: dict) -> pl.DataFrame:

    required_params = {
        "algo", "fact", "mode", "dstr",
        "land", "s", "l", "bpmin",
        "mu", "theta", "alphar", "Kp", "Kz"
    }
    missing = required_params - params.keys()
    if missing:
        raise ValueError(f"Missing parameters: {missing}")

    root = Path(root)
    paths = [
        str(p)
        for d in root.iterdir() if d.is_dir()
        for p in d.rglob("*.parquet")
        if p.name not in {"ncl_output.parquet", "ncl_heatmaps.parquet"}
    ]
    df_check = pl.scan_parquet(paths[0]).collect()

    if not paths:
        raise FileNotFoundError("No parquet files found.")

    FLOAT_COLS = {"alphar", "Kp", "Kz"}
    EPS = 1e-6

    def make_filter(k, v):
        if k in FLOAT_COLS:
            return pl.col(k).is_between(v - EPS, v + EPS)
        return pl.col(k) == v
    
    print(f"Scanning {len(paths)} parquet files...")

    df = (
        pl.scan_parquet(paths, include_file_paths="source_file")
        .filter(pl.all_horizontal([make_filter(k, v) for k, v in params.items()]))
        .collect()
    )

    if df.height == 0:
        raise ValueError("No dataframe found with the given parameters.")
    else:
        # print(f"Found in : {df['source_file'].unique().to_list()}")
        df = df.drop("source_file")
    return df


def _scalar_column_names(schema: pl.Schema) -> list[str]:
    """Retourne les noms des colonnes de types scalaires dans un schéma."""
    return [name for name, dtype in schema.items()
            if isinstance(dtype, _SCALAR_TYPES)]


def load_scalar_columns(
    paths: list[str],
    extra_cols: list[str] | str | None = None,
) -> pl.DataFrame:
    """Charge uniquement les colonnes scalaires depuis une liste de Parquets.

    Args:
        paths:      Liste de chemins vers des fichiers .parquet.
        extra_cols: Colonnes supplémentaires à inclure (même non scalaires).

    Returns:
        DataFrame avec les colonnes scalaires (+ extra_cols si précisé).

    Raises:
        ValueError: Si `paths` est vide.
    """
    if not paths:
        raise ValueError("Aucun fichier Parquet trouvé.")

    lf = pl.scan_parquet(paths)
    cols = _scalar_column_names(lf.schema)

    if extra_cols is not None:
        if isinstance(extra_cols, str):
            extra_cols = [extra_cols]
        cols = list(dict.fromkeys(cols + extra_cols))  # déduplique, conserve l'ordre

    return lf.select(cols).collect()


def reading_all_parquet(
    root: str | Path,
    extra_cols: list[str] | str | None = None
) -> pl.DataFrame:
    """Charge récursivement tous les fichiers Parquet d'un répertoire.

    Seules les colonnes scalaires sont conservées (+ extra_cols).
    Le résultat est trié selon `sort_cols` (uniquement les colonnes présentes).

    Args:
        root:       Répertoire racine à parcourir récursivement.
        extra_cols: Colonnes non scalaires à inclure malgré tout.

    Returns:
        DataFrame trié et fusionné de tous les fichiers Parquet trouvés.

    Raises:
        FileNotFoundError: Si aucun fichier .parquet n'est trouvé dans `root`.
    """
    root = Path(root)
    paths = [str(p) for p in root.rglob("*.parquet")]

    if not paths:
        raise FileNotFoundError(
            f"Aucun fichier .parquet trouvé dans : {root}"
        )

    df = load_scalar_columns(paths, extra_cols)

    return df.sort(by=PARAMS)


# 2.2 Main files


def getting_main_file_with_verifications(
    df: pl.DataFrame,
    nt: int = 10000,
    tmax: int = 100,
    dt: int = 1,
    alphao: float = 0.0,
    alphaf: float = 1.0,
    beta: float = 0.0,
    Lmin: int = 0,
    Lmax: int = 70_000,
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
        "land", "s", "l", "bpmin", 
        "mu", "theta", 
        "nt", "tmax", "dt", "times", 
        "alphao", "alphaf", "beta",
        "Lmin", "Lmax", "origin", "bps",
        
        "v_mean", "v_med", 
        "vf", "Cf", "wf", "vf_std", "Cf_std", "wf_std", 
        "vi_mean", "vi_med", "vi_mp",
    }

    filtered_columns = [col for col in df.columns if col in selected_columns]
    filtered_df = df.select(filtered_columns)

    conditions = {
        "nt": nt,
        "tmax": tmax,
        "dt": dt,
        "alphaf": alphaf,
        "alphao": alphao,
        "beta": beta,
        "Lmin": Lmin,
        "Lmax": Lmax,
        "origin": origin,
        "bps": bps,
    }

    for param, value in conditions.items():
        filtered_df = filtered_df.filter(pl.col(param) == value)

        if filtered_df.is_empty():
            raise ValueError(
                f"Aucune simulation trouvée après filtrage sur '{param}' = {value}."
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
        config (dict): must contain 's', 'l', 'bpmin', 'land' or 'land'
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

        alpha = cfg.get("land", cfg.get("land", "unknown"))
        s = cfg["s"]
        l = cfg["l"]
        bpmin = cfg["bpmin"]

        new_key = f"s{s}_l{l}_bp{bpmin}_{alpha}"
        new_computed_data[new_key] = cfg_data

    computed_data = new_computed_data

    # ─────────────────────────────────────────────
    # 5 : Build configuration key
    # ─────────────────────────────────────────────

    key_alpha = config.get("land", config.get("land", "unknown"))
    my_key = f"s{config['s']}_l{config['l']}_bp{config['bpmin']}_{key_alpha}"

    if my_key not in computed_data:
        raise KeyError(
            f"No heatmap data for key {my_key}\n"
            f"Available keys: {list(computed_data.keys())}"
        )

    config_data = computed_data[my_key]

    return df_main, config_data


def reading_heatmap_one_config(
    config,
    type_of_data,
    root=Path.home() / "Documents" / "Workspace" / "nucleo" / "outputs" / "2025-01-01_PSMN",
    load_full_data=False,
):
    """
    Reads heatmap data for one specific configuration.

    Args:
        config (dict): must contain 's', 'l', 'bpmin', 'land'
        type_of_data (str): one of ["raw", "norm_mu", "norm_th"]
            -> same vocabulary as plot_single_heatmap / plot_all_heatmaps,
               so you never have to mentally translate between
               "heatmap_th" and "norm_th" again.
        root (Path): directory containing parquet + heatmap pickle files
        load_full_data (bool): if True, also loads the full simulation
            parquet (ncl_output.parquet) into df_main. This is independent
            of type_of_data: you can ask for "norm_th" heatmaps AND the
            full dataframe at the same time.

    Returns:
        df_main (pl.DataFrame | None) : None unless load_full_data=True
        config_data (dict)
    """

    # ─────────────────────────────────────────────
    # 1 : Validate type_of_data
    # ─────────────────────────────────────────────

    valid_types = ["raw", "norm_mu", "norm_th"]
    if type_of_data not in valid_types:
        raise ValueError(f"Incorrect type_of_data: {type_of_data}. Must be in {valid_types}")

    # ─────────────────────────────────────────────
    # 2 : Load full simulation data (optional, independent of type_of_data)
    # ─────────────────────────────────────────────

    df_main = None

    if load_full_data:
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
        "raw": "ncl_hm_raw.pkl",
        "norm_mu": "ncl_hm_nmu.pkl",
        "norm_th": "ncl_hm_nth.pkl",
    }

    heatmap_file = heatmap_files[type_of_data]

    with open(root / heatmap_file, "rb") as f:
        computed_data = pickle.load(f)

    # ─────────────────────────────────────────────
    # 4 : Remap keys for readability
    # ─────────────────────────────────────────────

    new_computed_data = {}

    for _, cfg_data in computed_data.items():
        cfg = cfg_data["config"]

        # NOTE: si tes configs stockent parfois ce champ sous un autre nom
        # (ex: "alpha", "boundary"...), remplace "land" par la bonne clé
        # ici. Avant: cfg.get("land", cfg.get("land", "unknown")) était un
        # no-op (les deux fallback utilisaient la même clé "land").
        alpha = cfg.get("land", "unknown")
        s = cfg["s"]
        l = cfg["l"]
        bpmin = cfg["bpmin"]

        new_key = f"s{s}_l{l}_bp{bpmin}_{alpha}"
        new_computed_data[new_key] = cfg_data

    computed_data = new_computed_data

    # ─────────────────────────────────────────────
    # 5 : Build configuration key
    # ─────────────────────────────────────────────

    key_alpha = config.get("land", "unknown")
    my_key = f"s{config['s']}_l{config['l']}_bp{config['bpmin']}_{key_alpha}"

    if my_key not in computed_data:
        raise KeyError(
            f"No heatmap data for key {my_key}\n"
            f"Available keys: {list(computed_data.keys())}"
        )

    config_data = computed_data[my_key]

    return df_main, config_data