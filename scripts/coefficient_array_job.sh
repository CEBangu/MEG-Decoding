#!/bin/bash
#SBATCH --job-name=coefficient_array_job # name
#SBATCH --output=output_%A_%a.log # Job Array ID, Task ID
#SBATCH --error=error_%A_%a.log # Job Array ID, Task ID
#SBATCH --time=00:20:00 # timelimit - shouldn't take more than 20 mins if the nodes are available
#SBATCH --partition=common # partition
#SBATCH --qos=fast #superfast might be cutting it close, but we'll see how fast it is on the good machines
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks=1 # number of tasks
#SBATCH --cpus-per-task=8 # let's take 8 cpus because we parallelized the channels
#SBATCH --mem=24000 # memory in MB
#SBATCH --array=1-21 # number of jobs in the array
#SBATCH --mail-type=END,FAIL # send email on job end and if it fails
#SBATCH --mail-user=ciprian.bangu@pasteur.fr # email

SUBJECT_LISTS=(
    "BCOM_01_2 BCOM_01_3 BCOM_01_4"
    "BCOM_02_2 BCOM_02_3 BCOM_02_4"
    "BCOM_04_2 BCOM_04_3 BCOM_04_4"
    "BCOM_06_2 BCOM_06_3 BCOM_06_4"
    "BCOM_07_2 BCOM_07_3 BCOM_07_4"
    "BCOM_08_2 BCOM_08_3 BCOM_08_4"
    "BCOM_09_2 BCOM_09_3 BCOM_09_4"
    "BCOM_10_2 BCOM_10_3 BCOM_10_4"
    "BCOM_11_2 BCOM_11_3 BCOM_11_4"
    "BCOM_12_2 BCOM_12_3 BCOM_12_4"
    "BCOM_13_2 BCOM_13_3 BCOM_13_4"
    "BCOM_14_2 BCOM_14_3 BCOM_14_4"
    "BCOM_15_2 BCOM_15_3 BCOM_15_4"
    "BCOM_16_2 BCOM_16_3 BCOM_16_4"
    "BCOM_18_2 BCOM_18_3 BCOM_18_4"
    "BCOM_19_2 BCOM_19_3 BCOM_19_4"
    "BCOM_21_2 BCOM_21_3 BCOM_21_4"
    "BCOM_22_2 BCOM_22_3 BCOM_22_4"
    "BCOM_23_2 BCOM_23_3 BCOM_23_4"
    "BCOM_24_2 BCOM_24_3 BCOM_24_4"
    "BCOM_26_2 BCOM_26_3 BCOM_26_4"
)

# activate virtual env
source $HOME/venvs/coefficients_env/bin/activate

# set pythonpath so we can import the custom modules
export PYTHONPATH="$HOME/MEG-Decoding:$PYTHONPATH"
echo "Python path: $PYTHONPATH"

# want the correct index for the slurm array job
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# get the subject list
SUBJECTS="${SUBJECT_LISTS[$INDEX]}"

echo "Running job for subjects: $SUBJECTS"

# run the python script
python3 coefficient_array_job.py --subject_list $SUBJECTS --speech_type covert