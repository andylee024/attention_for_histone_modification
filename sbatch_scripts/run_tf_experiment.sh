#!/bin/bash

# ####################################
# Setup tensorflow variables
# ####################################

# Setup environement variables
KOMOREBI="/home/andyl2/Projects/komorebi"
RUN_EXPERIMENT="${KOMOREBI}/tools/run_experiment_lane.py"
CONFIG_DIR="${KOMOREBI}/config/experiment_config"
LOG_DIR="/home/andyl2/logs"

# ####################################
# User specified tensorflow variables
# ####################################
export EXPERIMENT_CONFIG="${CONFIG_DIR}/lane_integration_test.json"
export LOG="${LOG_DIR}/lane_integration_test_sbatch_out.txt"

echo "hello world"
echo "running tensorflow experiment on $(date)"
echo "config file ${EXPERIMENT_CONFIG}"
echo "log is located at ${LOG}"

# ####################################
# Setup tensorflow environment on lane
# ####################################

# Step 1: source environment
source /home/andyl2/.bashrc

# Step 2: setup singularity shell (not needed if script called from singularity exec) This is done in the corresponding launch_sbatch script.
# singularity shell /containers/images/ubuntu-16.04-lts-tensorflow-1.3.0_cudnn-8.0-v6.img 

# Step 3: activate tensorflow virtual environment
source /home/andyl2/virtual_environments/tf_env/bin/activate

# ####################################
# Execute tensorflow experiment
# ####################################
python ${RUN_EXPERIMENT} -c ${EXPERIMENT_CONFIG} > ${LOG} 2>&1

# ############################
# Clean up
# ############################

# deactivate virtual environment
deactivate

# final print
echo "completed tensorflow experiment on $(date)"
