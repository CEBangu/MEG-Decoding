#!/bin/bash
#SBATCH --job-name=cnn_train
#SBATCH --output=logs/model_logs/output_%j.log
#SBATCH --error=logs/model_logs/error_%j.log
#SBATCH --partition=gpu #from dedicatedgpu
# #SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8000 #8GB
#SBATCH --mail-type=BEGIN,END,FAIL # send email on job end and if it fails
#SBATCH --mail-user=ciprian.bangu@pasteur.fr # email


# load cuda
module load cuda/12.8

source training_env_config.sh

# loading model config
MODEL_CONFIG_FILE="$1"

if [ -z "$MODEL_CONFIG_FILE" ]; then
    echo "ERROR: No config file provided! Usage: sbatch coefficient_array_job.sh <config_file>"
    exit 1
fi


# Load config file
if [ -f "$MODEL_CONFIG_FILE" ]; then
    source "$MODEL_CONFIG_FILE"
    echo "Loaded config: $MODEL_CONFIG_FILE"
else
    echo "ERROR: Config file $MODEL_CONFIG_FILE not found!"
    exit 1
fi

# Activate venv
if [ -f "$HOME/venvs/modeltraining/bin/activate" ]; then
    source "$HOME/venvs/modeltraining/bin/activate"
    echo "Virtual environment activated"
else
    echo "ERROR: Virtual environment not found or failed to activate."
    exit 1
fi

# Debugging output
echo "MODEL_TYPE: $MODEL_TYPE"
echo "FREEZE_TYPE: $FREEZE_TYPE"
echo "NUM_FOLDS: $NUM_FOLDS"
echo "LABELS: $LABELS"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "NUM_CLASSES: $NUM_CLASSES"

# set pythonpath so we can import the custom modules
export PYTHONPATH="$HOME/MEG-Decoding:$PYTHONPATH"
echo "Python path: $PYTHONPATH"

CMD="python3 $HOME/MEG-Decoding/cnn_training.py --model_type $MODEL_TYPE --freeze_type $FREEZE_TYPE --num_folds $NUM_FOLDS --project_name $PROJECT_NAME --labels $LABELS"
# Check if NUM_CLASSES is defined and add it to the command
if [[ -n "${NUM_CLASSES}" ]]; then
    CMD+=" --num_classes $NUM_CLASSES"
fi

# Execute the command
eval "$CMD"