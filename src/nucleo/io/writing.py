"""
nucleo.writing_functions
------------------------
Writing functions for writing results, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


def set_working_environment(base_dir: str = Path.home() / "Documents" / "PhD" / "Workspace", subfolder: str = "") -> None:
    """
    Ensure the specified folder exists and change the current working directory to it.
        Check if the folder exists; if not, create it
        Change the current working directory to the specified folder

    Args:
        folder_path (str): Path to the folder where the working environment should be set.

    Returns:
        None.
    """
    root = os.getcwd()
    full_path = os.path.join(root, base_dir, subfolder)
    
    os.makedirs(full_path, exist_ok=True)
    os.chdir(full_path)

    return full_path


def prepare_value(value):
    """
    Convert various data types to Parquet-compatible formats
    while preserving numerical dtypes (especially Float64).
    """

    # NumPy arrays / matrices → list
    if isinstance(value, (np.ndarray, np.matrix)):
        return [prepare_value(v) for v in np.array(value).tolist()]

    # NumPy scalars
    elif isinstance(value, np.integer):
        return value.item()

    elif isinstance(value, np.floating):
        # IMPORTANT: keep NaN as NaN (do NOT convert to None)
        return float(value)

    # Python float
    elif isinstance(value, float):
        return value  # NaN stays NaN

    # Lists
    elif isinstance(value, list):
        return [prepare_value(v) for v in value]

    # Native scalars / strings
    elif isinstance(value, (int, str)):
        return value

    # None is allowed
    elif value is None:
        return None

    else:
        raise ValueError(f"Unsupported data type: {type(value)}")


def writing_parquet(file:str, title: str, data_result: dict, data_info = False) -> None:
    """
    Write a dictionary directly into a Parquet file using PyArrow.
    Ensures that all numerical values, arrays, and lists are properly handled.

    Note:
        - Each key in the Parquet file must correspond to a list or an array.
        - Compatible only with native Python types.
        - Even a number like 1.797e308 only takes up 8 bytes (64 bits) in the Parquet file.

    Args:
        title (str): The base name for the Parquet file and folder.
        data_result (dict): Dictionary containing the data to write. 
            Supported types for values:
                - NumPy arrays (converted to lists).
                - NumPy matrices (converted to lists).
                - NumPy scalars (converted to native Python types).
                - Python scalars (int, float).
                - Lists (unchanged).
                - Strings (unchanged).

    Returns:
        None: This function does not return any value.

    Raises:
        ValueError: If a value in the dictionary has an unsupported data type.
        Exception: If writing to the Parquet file fails for any reason.
    """
    
    os.path.join(Path.home() / "Documents" / "PhD" / "Workspace")

    # If you need to see the details od the data registered
    if data_info :
        inspect_data_types(data_result)

    # Define the Parquet file path
    data_file_name = os.path.join(title, f'{title}.parquet')
    os.makedirs(title, exist_ok=True)

    # Prepare the data for Parquet
    prepared_data = {key: [prepare_value(value)] if not isinstance(value, list) else prepare_value(value)
                     for key, value in data_result.items()}
    
    prepared_data = {k: [prepare_value(v)] for k, v in data_result.items()}


    try:        
        table = pa.table(prepared_data)                                         # Create a PyArrow Table from the dictionary
        pq.write_table(table, data_file_name, compression='gzip')               # Write the table to a Parquet file

    except Exception as e:
        print(f"Failed to write Parquet file due to: {e}")
    
    return None


def inspect_data_types(data: dict, launch = True) -> None:
    """
    Inspect and print the types and dimensions of values in a dictionary.

    Args:
        data (dict): Dictionary containing the data to inspect. 
            Keys are expected to be strings, and values can be of various types, such as:
            - NumPy arrays (prints dimensions).
            - Lists (prints length).
            - Other types (prints the type of the value).

    Returns:
        None
    """
    if launch:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"Key: {key}, Dimensions: {value.shape}")     # Check if the value is a NumPy array
            elif isinstance(value, list):
                print(f"Key: {key}, Length: {len(value)}")          # Check if the value is a list
            else:
                print(f"Key: {key}, Type: {type(value)}")           # Other types
    return None