#!/bin/bash
#SBATCH --job-name=coefficient_array_job # name
#SBATCH --output=logs/coef_logs/output_%A_%a.log # Job Array ID, Task ID
#SBATCH --error=logs/coef_logs/error_%A_%a.log # Job Array ID, Task ID
#SBATCH --time=02:00:00 # timelimit - shouldn't take more than 20 mins if the nodes are available
#SBATCH --partition=common # partition
#SBATCH --qos=fast #superfast might be cutting it close, but we'll see how fast it is on the good machines
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks=1 # number of tasks
#SBATCH --cpus-per-task=1 # no internal parallelizing
#SBATCH --mem=12000 # memory in MB
#SBATCH --array=1-63 # number of jobs in the array
#SBATCH --mail-type=BEGIN,END,FAIL # send email on job end and if it fails
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
echo "DATA_DIR: $DATA_DIR"
echo "EMPTY_ROOM_DIR:$ $EMPTY_ROOM_DIR"
echo "BASELINE_DIR: $BASELINE_DIR"
echo "TRANS_DIR: $TRANS_DIR"
echo "RAW_DIR: $RAW_DIR"
echo "TC_DIR: $TC_DIR"

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
# Print Python command for debugging
echo "Executing Python command:"
echo "python3 $HOME/MEG-Decoding/beamforming_roi_coefficients.py \
    --subject_list $SUBJECTS \
    $AVOID_READING \
    $AVOID_PRODUCING \
    --speech_type $SPEECH_TYPE \
    --data_dir $DATA_DIR \
    --save_dir $SCRATCH_DIR \
    --empty_room_dir $EMPTY_ROOM_DIR \
    --baseline_dir $BASELINE_DIR \
    --trans_dir $TRANS_DIR \
    --raw_dir $RAW_DIR \
    --tc_save_dir $TC_DIR \
    $MNE_ARG"

# run the python script
python3 $HOME/MEG-Decoding/beamforming_roi_coefficients.py \
    --subject_list $SUBJECTS \
    $AVOID_READING \
    $AVOID_PRODUCING \
    --speech_type $SPEECH_TYPE \
    --data_dir $DATA_DIR \
    --save_dir $SCRATCH_DIR \
    --empty_room_dir $EMPTY_ROOM_DIR \
    --baseline_dir $BASELINE_DIR \
    --trans_dir $TRANS_DIR \
    --raw_dir $RAW_DIR \
    --tc_save_dir $TC_DIR \
    $MNE_ARG
