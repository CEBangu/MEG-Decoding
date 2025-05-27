#!/bin/bash
#SBATCH --job-name=roi_percentiles
#SBATCH --output=/pasteur/appa/homes/cbangu/MEG-Decoding/scripts/SLURM/logs/percent_roi/roi_out_%A_%a.txt
#SBATCH --error=/pasteur/appa/homes/cbangu/MEG-Decoding/scripts/SLURM/logs/percent_roi/roi_err_%A_%a.txt
#SBATCH --array=0-49
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

DATA_DIR=/pasteur/appa/scratch/cbangu/coefficients/covert_producing_roi_beamforming
# DATA_DIR=/pasteur/appa/scratch/cbangu/coefficients/covert_reading_roi_beamforming
# DATA_DIR=/pasteur/appa/scratch/cbangu/coefficients/overt_producing_roi_beamforming
CHUNK_INDEX=${SLURM_ARRAY_TASK_ID}
TOTAL_CHUNKS=50
OUTPUT_DIR=/pasteur/appa/homes/cbangu/MEG-Decoding/percentiles/percentiles_roi_CP
# OUTPUT_DIR=/pasteur/appa/homes/cbangu/MEG-Decoding/percentiles/percentiles_roi_CR
# OUTPUT_DIR=/pasteur/appa/homes/cbangu/MEG-Decoding/percentiles/percentiles_roi_OP

if [ -f "$HOME/venvs/coefficients_env/bin/activate" ]; then
    source "$HOME/venvs/coefficients_env/bin/activate"
    echo "Virtual environment activated"
else
    echo "ERROR: Virtual environment not found or failed to activate."
    exit 1
fi

mkdir -p $OUTPUT_DIR logs

python /pasteur/appa/homes/cbangu/MEG-Decoding/percentile_chunks_roi.py "$DATA_DIR" "$CHUNK_INDEX" "$TOTAL_CHUNKS" "$OUTPUT_DIR"