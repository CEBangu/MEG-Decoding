#!/bin/bash
#SBATCH --job-name=coefficient_array_job # name
#SBATCH --output=output_%A_%a.log # Job Array ID, Task ID
#SBATCH --error=error_%A_%a.log # Job Array ID, Task ID
#SBATCH --time=02:00:00 # timelimit - shouldn't take more than 20 mins if the nodes are available
#SBATCH --partition=common # partition
#SBATCH --qos=fast #superfast might be cutting it close, but we'll see how fast it is on the good machines
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks=1 # number of tasks
#SBATCH --cpus-per-task=8 # let's take 8 cpus because we parallelized the channels
#SBATCH --mem=24000 # memory in MB
#SBATCH --array=1-21 # number of jobs in the array
#SBATCH --mail-type=END,FAIL # send email on job end and if it fails
#SBATCH --mail-user=ciprian.bangu@pasteur.fr # email

# check config passed as argument
CONFIG_FILE="$1"

if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: No config file provided! Usage: sbatch coefficient_array_job.sh <config_file>"
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

# Debugging output
echo "SCRATCH_DIR: $SCRATCH_DIR"
echo "FINAL_STORAGE: $FINAL_STORAGE"
echo "DATA_DIR: $DATA_DIR"
echo "SPEECH_TYPE: $SPEECH_TYPE"
echo "AVOID_READING: $AVOID_READING"
echo "AVOID_PRODUCING: $AVOID_PRODUCING"
echo "Job Array Index: $SLURM_ARRAY_TASK_ID"


INDEX=$((SLURM_ARRAY_TASK_ID - 1))

SUBJECTS="${SUBJECT_LISTS[$INDEX]}"

MNE_ARG=""
if [ -n "$MNE_DIR" ]; then
    MNE_ARG="--mne_dir $MNE_DIR"
fi

echo $MNE_ARG

echo "Running job for subjects: $SUBJECTS"

# make sure the directories exist
mkdir -p "$SCRATCH_DIR"
mkdir -p "$FINAL_STORAGE"

# Activate virtual environment and check that it exists
if [ -f "$HOME/venvs/coefficients_env/bin/activate" ]; then
    source "$HOME/venvs/coefficients_env/bin/activate"
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

# Print Python command for debugging
echo "Executing Python command:"
echo "python3 $HOME/MEG-Decoding/roi_coefficients.py --subject_list $SUBJECTS --speech_type $SPEECH_TYPE $AVOID_READING $AVOID_PRODUCING --data_dir $DATA_DIR --save_dir $SCRATCH_DIR $MNE_ARG"

# run the python script
python3 $HOME/MEG-Decoding/roi_coefficients.py --subject_list $SUBJECTS --speech_type $SPEECH_TYPE $AVOID_READING $AVOID_PRODUCING --data_dir "$DATA_DIR" --save_dir "$SCRATCH_DIR" $MNE_ARG