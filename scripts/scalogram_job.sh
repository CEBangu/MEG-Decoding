#!/bin/bash
#SBATCH --job-name=plot_scalograms
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --time=02:00:00
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ciprian.bangu@pasteur.fr

# Check if config file was provided
CONFIG_FILE="$1"

if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: No config file provided! Usage: sbatch plotting_job.sh <config_file>"
    exit 1
fi

# Load config file
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    echo "Loaded config: $CONFIG_FILE"
else
    echo "ERROR: Config file $CONFIG_FILE not found!"
    exit 1
fi

echo "DATA_DIR: $DATA_DIR"
echo "SAVE_DIR: $SAVE_DIR"
echo "DIMENSIONS: $DIMENSIONS"
echo "CMAP: $CMAP"
echo "RESOLUTION: $RESOLUTION"
echo "INDEX_LIST: $INDEX_LIST"
echo "AVERAGE: $AVERAGE"

# Activate virtual environment
if [ -f "$HOME/venvs/scalograms/bin/activate" ]; then
    source "$HOME/venvs/scalograms/bin/activate"
    echo "Virtual environment activated"
else
    echo "ERROR: Virtual environment not found or failed to activate."
    exit 1
fi

export PYTHONPATH="$HOME/MEG-Decoding:$PYTHONPATH"
echo "Python path: $PYTHONPATH"

# Construct command
CMD="srun python3 $HOME/MEG-Decoding/plotting_script.py \
    --data_dir \"$DATA_DIR\" \
    --save_dir \"$SAVE_DIR\" \
    --dimensions $DIMENSIONS \
    --cmap \"$CMAP\" \
    --index_list \"$INDEX_LIST\" \
    --resolution \"$RESOLUTION\""

if [ -n "$AVERAGE" ]; then #because otherwise it breaks the script to just leave it empty
    CMD="$CMD --average"
fi

echo "Running command: $CMD"
eval $CMD