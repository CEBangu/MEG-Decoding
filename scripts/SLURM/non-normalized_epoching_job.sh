#!/bin/bash
#SBATCH --job-name=non_norm_epoching # name
#SBATCH --output=epoching_logs/non_norm_%A_%a.out
#SBATCH --error=epoching_logs/non_norm_%A_%a.err
#SBATCH --partition=common # partition
#SBATCH --qos=superfast #superfast might be cutting it close, but we'll see how fast it is on the good machines
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks=1 # number of tasks
#SBATCH --cpus-per-task=3 # auotreject has job options
#SBATCH --mem=3000 # memory in MB
#SBATCH --array=0-62 # one job on 
#SBATCH --mail-type=BEGIN,END,FAIL # send email on job end and if it fails
#SBATCH --mail-user=ciprian.bangu@pasteur.fr # email

export SUBJECT_LISTS=(
    "BCOM_01/2" 
    "BCOM_01/3" 
    "BCOM_01/4"
    "BCOM_02/2" 
    "BCOM_02/3" 
    "BCOM_02/4"
    "BCOM_04/2" 
    "BCOM_04/3" 
    "BCOM_04/4"
    "BCOM_06/2" 
    "BCOM_06/3" 
    "BCOM_06/4"
    "BCOM_07/2" 
    "BCOM_07/3" 
    "BCOM_07/4"
    "BCOM_08/2" 
    "BCOM_08/3" 
    "BCOM_08/4"
    "BCOM_09/2" 
    "BCOM_09/3" 
    "BCOM_09/4"
    "BCOM_10/2" 
    "BCOM_10/3" 
    "BCOM_10/4"
    "BCOM_11/2" 
    "BCOM_11/3" 
    "BCOM_11/4"
    "BCOM_12/2" 
    "BCOM_12/3" 
    "BCOM_12/4"
    "BCOM_13/2" 
    "BCOM_13/3" 
    "BCOM_13/4"
    "BCOM_14/2" 
    "BCOM_14/3" 
    "BCOM_14/4"
    "BCOM_15/2" 
    "BCOM_15/3" 
    "BCOM_15/4"
    "BCOM_16/2" 
    "BCOM_16/3" 
    "BCOM_16/4"
    "BCOM_18/2" 
    "BCOM_18/3" 
    "BCOM_18/4"
    "BCOM_19/2" 
    "BCOM_19/3" 
    "BCOM_19/4"
    "BCOM_21/2" 
    "BCOM_21/3" 
    "BCOM_21/4"
    "BCOM_22/2" 
    "BCOM_22/3" 
    "BCOM_22/4"
    "BCOM_23/2" 
    "BCOM_23/3" 
    "BCOM_23/4"
    "BCOM_24/2" 
    "BCOM_24/3" 
    "BCOM_24/4"
    "BCOM_26/2" 
    "BCOM_26/3" 
    "BCOM_26/4"
)

export ROOT="/pasteur/zeus/projets/p02/BCOM"

# Activate virtual environment and check that it exists
if [ -f "$HOME/venvs/mne_jupyter/bin/activate" ]; then # just use the jupyter notebook one, it has all the packaged required
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

SUBJECT=${SUBJECT_LISTS[$SLURM_ARRAY_TASK_ID]}

# Accept external flag from command line submission (e.g., sbatch --export=SAVE_BASELINE=1 ...)
if [ "$SAVE_BASELINE" == "1" ]; then
    python3 $HOME/MEG-Decoding/non_normalized_epoching.py --subject "$SUBJECT" --root "$ROOT" --save_baseline
else
    python3 $HOME/MEG-Decoding/non_normalized_epoching.py --subject "$SUBJECT" --root "$ROOT"
fi