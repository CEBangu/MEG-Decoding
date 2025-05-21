#!/bin/bash
#SBATCH --job-name=percentiles
#SBATCH --output=/pasteur/appa/homes/cbangu/MEG-Decoding/scripts/SLURM/logs/percentiles/out_%A_%a.txt
#SBATCH --error=/pasteur/appa/homes/cbangu/MEG-Decoding/scripts/SLURM/logs/percentiles/err_%A_%a.txt
#SBATCH --array=0-99             # 100 jobs total
#SBATCH --qos=fast
#SBATCH --mem=4000
#SBATCH --cpus-per-task=1

# DATA_DIR=/pasteur/appa/scratch/cbangu/coefficients/covert_producing
# DATA_DIR=/pasteur/appa/scratch/cbangu/coefficients/covert_reading
DATA_DIR=/pasteur/appa/scratch/cbangu/coefficients/overt_producing
TOTAL_CHUNKS=100
CHUNK_INDEX=${SLURM_ARRAY_TASK_ID}
# OUTFILE=/pasteur/appa/homes/cbangu/MEG-Decoding/percentiles_CP/percentile_${CHUNK_INDEX}.csv
# OUTFILE=/pasteur/appa/homes/cbangu/MEG-Decoding/percentiles_CR/percentile_${CHUNK_INDEX}.csv
OUTFILE=/pasteur/appa/homes/cbangu/MEG-Decoding/percentiles_OP/percentile_${CHUNK_INDEX}.csv

# mkdir -p /pasteur/appa/homes/cbangu/MEG-Decoding/percentiles_CP
# mkdir -p /pasteur/appa/homes/cbangu/MEG-Decoding/percentiles_CR
mkdir -p /pasteur/appa/homes/cbangu/MEG-Decoding/percentiles_OP
mkdir -p /pasteur/appa/homes/cbangu/MEG-Decoding/percentiles_OP


if [ -f "$HOME/venvs/coefficients_env/bin/activate" ]; then
    source "$HOME/venvs/coefficients_env/bin/activate"
    echo "Virtual environment activated"
else
    echo "ERROR: Virtual environment not found or failed to activate."
    exit 1
fi

python /pasteur/appa/homes/cbangu/MEG-Decoding/percentile_chunks.py "$DATA_DIR" "$CHUNK_INDEX" "$TOTAL_CHUNKS" "$OUTFILE"