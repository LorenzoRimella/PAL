#!/bin/bash
#SBATCH --job-name=le_C      # Descriptive job name
#SBATCH --time=08:00:00                       # Maximum wall time (hh:mm:ss)
#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --ntasks-per-node=1                   # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --cpus-per-task=10                    # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1                          # Number of GPUs per node (adjust to match hardware)
#SBATCH --partition=g100_usr_interactive           # GPU-enabled partition
#SBATCH --account=uBG25_EcoRim  # Project account number

# Load necessary modules
module load cuda/11.5.0         # Make sure this includes libcudart and other CUDA libs
module load openmpi             # Only if needed

# Load conda
source miniconda3/etc/profile.d/conda.sh
conda activate tf_cineca

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/g100_work/PROJECTS/spack/v0.17/prod/0.17.1/install/0.17/linux-centos8-skylake_avx512/gcc-8.4.1/cuda-11.5.0-ktwkkqqhebhe64r4ial5g632vefweb4i
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

# Run test script
python PAL_cineca/likelihood_evaluation_C.py
