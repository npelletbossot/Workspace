#!/bin/bash
#SBATCH --job-name=ncl_test
#SBATCH --output=logs/nucleo_%j.out
#SBATCH --error=logs/nucleo_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4G

# --- Go to project folder --- #
cd /Xnfs/physbiochrom/npellet/nucleo

# --- Activate venv with absolute path --- #
source /Xnfs/physbiochrom/npellet/nucleo/.venv_nucleo_PSMN/bin/activate

# --- Run script --- #
python3 main.py
