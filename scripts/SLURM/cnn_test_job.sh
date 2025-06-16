#!/bin/bash
#SBATCH --job-name=cnn_seeds
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --qos=fast            
#SBATCH --output=/pasteur/appa/homes/cbangu/MEG-Decoding/scripts/SLURM/logs/model_logs/out_%A_%a.log
#SBATCH --error=/pasteur/appa/homes/cbangu/MEG-Decoding/scripts/SLURM/logs/model_logs/err_%A_%a.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ciprian.bangu@pasteur.fr


module load cuda/12.8
source /pasteur/appa/homes/cbangu/MEG-Decoding/scripts/SLURM/training_env_config.sh


MODEL_CONFIG_FILE="$1"
if [[ -z "$MODEL_CONFIG_FILE" || ! -f "$MODEL_CONFIG_FILE" ]]; then
  echo "ERROR: config file missing (usage: sbatch this_script.sh <config_file>)"
  exit 1
fi
source "$MODEL_CONFIG_FILE"
echo "Loaded config: $MODEL_CONFIG_FILE"


source /pasteur/appa/homes/cbangu/venvs/modeltraining/bin/activate


echo "Seed: $SLURM_ARRAY_TASK_ID"
echo "MODEL_TYPE: $MODEL_TYPE  TRAIN_SET: $TRAIN_SET  TEST_SET: $TEST_SET"
echo "LR=$LR  BATCH=$BATCH  WD=$WD  FREEZE=$FREEZE  TRANSFORMS=$TRANSFORMS"


export PYTHONPATH="/pasteur/appa/homes/cbangu/MEG-Decoding:$PYTHONPATH"


python3 "/pasteur/appa/homes/cbangu/MEG-Decoding/cnn_testing.py" \
  --model_type      "$MODEL_TYPE" \
  --num_classes     "$NUM_CLASSES" \
  --project_name    "$PROJECT_NAME" \
  --train_labels    "$TRAIN_SET" \
  --test_labels     "$TEST_SET" \
  --batch_size      "$BATCH" \
  --learning_rate   "$LR" \
  --weight_decay    "$WD" \
  --freeze          "$FREEZE" \
  --optimizer       "$OPTIM" \
  --transforms      "$TRANSFORMS" \
  --seed            "$SLURM_ARRAY_TASK_ID"