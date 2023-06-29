#!/bin/bash

# general configuration of the job
#SBATCH --job-name=S-LSTM
#SBATCH --account=deepext
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=S-LSTM.out
#SBATCH --error=S-LSTM.err
#SBATCH --time=05:00:00

# configure node and process count on the CM
#SBATCH --partition=dp-esb
#SBATCH --nodes=21
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --exclusive


ml --force purge
ml Stages/2023 GCC/11.3.0 OpenMPI/4.1.4 cuDNN/8.6.0.163-CUDA-11.7 Python/3.10.4
# recent bug: https://gitlab.jsc.fz-juelich.de/software-team/easybuild/-/wikis/Failed-to-initialize-NVML-Driver-library-version-mismatch-message
ml -nvidia-driver/.default

export CUDA_VISIBLE_DEVICES="0"
export HDF5_USE_FILE_LOCKING='FALSE'

source dwave_env/bin/activate

# New CUDA drivers on the compute nodes
ln -s /usr/lib64/libcuda.so.1 .
ln -s /usr/lib64/libnvidia-ml.so.1 .
LD_LIBRARY_PATH=.:/usr/local/cuda/lib64:$LD_LIBRARY_PATH


srun --mpi=pspmix python3 -u MPI_swift_hyperband.py --dir_name=saved_models_s_lstm --seed=0 --model_name=lstm --pred_type=classical --r=300