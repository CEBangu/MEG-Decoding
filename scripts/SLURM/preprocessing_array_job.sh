#!/bin/bash
#SBATCH --job-name=ica_array # name
#SBATCH --output=ica_logs/meg_ica_%A_%a.out
#SBATCH --error=ica_logs/meg_ica_%A_%a.err
#SBATCH --partition=common # partition
# #SBATCH --qos=fast #superfast might be cutting it close, but we'll see how fast it is on the good machines
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks=1 # number of tasks
#SBATCH --cpus-per-task=2 # let's take 8 cpus because we parallelized the channels
#SBATCH --mem=24000 # memory in MB
#SBATCH --array=0 # usually to 20, but 5 for the rerun; number of jobs in the array
#SBATCH --mail-type=BEGIN,END,FAIL # send email on job end and if it fails
#SBATCH --mail-user=ciprian.bangu@pasteur.fr # email


# Activate virtual environment and check that it exists
if [ -f "$HOME/venvs/mne_jupyter/bin/activate" ]; then
    source "$HOME/venvs/mne_jupyter/bin/activate"
    echo "Virtual environment activated"
else
    echo "ERROR: Virtual environment not found or failed to activate."
    exit 1
fi

echo "Python version in use:"
python3 --version
which python3

# set pythonpath so we can import the custom modules
export PYTHONPATH="$HOME/MEG-Decoding:$PYTHONPATH"
echo "Python path: $PYTHONPATH"


# SUBJECT_LISTS=(
#     "BCOM_01"
#     "BCOM_02"
#     "BCOM_04"
#     "BCOM_06"
#     "BCOM_07"
#     "BCOM_08"
#     "BCOM_09"
#     "BCOM_10"
#     "BCOM_11"
#     "BCOM_12"
#     "BCOM_13"
#     "BCOM_14"
#     "BCOM_15"
#     "BCOM_16"
#     "BCOM_18"
#     "BCOM_19"
#     "BCOM_21"
#     "BCOM_22"
#     "BCOM_23"
#     "BCOM_24"
#     "BCOM_26"
# )

SUBJECT_LISTS=(
    "BCOM_19"
)


SUBJECT=${SUBJECT_LISTS[$SLURM_ARRAY_TASK_ID]}

python3 $HOME/MEG-Decoding/preprocessing.py --subject "$SUBJECT"