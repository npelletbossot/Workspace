#!/bin/bash
#SBATCH --job-name=ncl_compaction
#SBATCH --output=logs/nucleo_%A_%a.out
#SBATCH --error=logs/nucleo_%A_%a.err
#SBATCH --time=168:00:00
#SBATCH --array=0-47%40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=Lake

# --- Go to project folder --- #
cd /Xnfs/physbiochrom/npellet/nucleo

# --- Export exact date --- #
export LAUNCH_DATE=$(date +%Y-%m-%d)

# --- Activate venv with absolute path --- #
source /Xnfs/physbiochrom/npellet/nucleo/.venv_nucleo_PSMN/bin/activate

# --- Run script --- #
python3 main_compaction.py
