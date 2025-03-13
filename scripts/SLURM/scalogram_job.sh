#!/bin/bash
#SBATCH --job-name=plot_scalograms
#SBATCH --output=output_%A_%a.log
#SBATCH --error=error_%A_%a.log
#SBATCH --time=05:00:00
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --ntasks=1       
#SBATCH --cpus-per-task=8  # Or adjust as needed
#SBATCH --mem=10000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ciprian.bangu@pasteur.fr
#SBATCH --array=0-20       # 21 tasks

# Load config file
CONFIG_FILE="$1"
if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: No config file provided! Usage: sbatch plotting_job.sh <config_file>"
    exit 1
fi

source "$CONFIG_FILE"

# Activate virtual environment
source "$HOME/venvs/scalograms/bin/activate"

mkdir -p "$SAVE_DIR"
mkdir -p "$LABEL_DIR"
mkdir -p "$TRAIN_TEST_DIR"

# Run Python script with epoch_workers=4 (or change as needed)
srun python3 $HOME/MEG-Decoding/plotting_script.py \
    --data_dir "$DATA_DIR" --save_dir "$SAVE_DIR" \
    --dimensions $DIMENSIONS --cmap "$CMAP" --index_list "$INDEX_LIST" \
    --resolution "$RESOLUTION" --epoch_workers 4 --task_id $SLURM_ARRAY_TASK_ID \
    ${AVERAGE:+--average}
    