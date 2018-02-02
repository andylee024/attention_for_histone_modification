#!/bin/bash
#SBATCH -p gpu 
#SBATCH --gres=gpu:0
#SBATCH --mem=10G
#SBATCH --job-name attn_exp
#SBATCH -e /home/andyl2/job_logs/job.%j.err
#SBATCH -o /home/andyl2/job_logs/job.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=drifter024@gmail.com

# ############################
# Setup tensorflow environment
# ############################

# setup modules
source /etc/profile.d/modules.sh
module load gcc-4.9.2
module load cuda-8.0
module load python27

# export local environment
export LC_ALL=C

# variables
SINGULARITY_CONTAINER="/containers/images/ubuntu-16.04-lts-tensorflow-1.3.0_cudnn-8.0-v6.img"
TF_EXPERIMENT_SCRIPT="/home/andyl2/Projects/komorebi/sbatch_scripts/run_tf_experiment.sh"

# User (specify script to run here)
EXECUTABLE=${TF_EXPERIMENT_SCRIPT}

# launch experiment 
singularity exec ${SINGULARITY_CONTAINER} /bin/bash ${EXECUTABLE}

# launch experiment (old)
# bash ${EXECUTABLE}

