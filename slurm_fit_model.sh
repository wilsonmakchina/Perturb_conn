#!/bin/bash
#SBATCH --job-name=fit_model_job         # Job name
#SBATCH --output=sjob_%j.out            # Standard output and error log, where %j is the job ID
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --partition=cpu-share
#SBATCH --mem=60G
#SBATCH --ntasks=10
# Initialize conda for bash shell using the base conda.sh script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the conda environment
conda activate Perturb_conn

export MKL_NUM_THREADS=1

export OMP_NUM_THREADS=1

export NUMEXPR_NUM_THREADS=1

# python -u create_data_set.py 

# Run the Python script with your SLURM configuration file
# python main.py submission_params/slurm_script.yml
mpiexec -n 10 python -m mpi4py main.py submission_params/exp_test.yml