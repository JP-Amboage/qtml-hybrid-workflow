#!/bin/bash

# general configuration of the job
#SBATCH --job-name=openml
#SBATCH --account=deepext
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=openml_quantum.out
#SBATCH --error=openml_quantum.err
#SBATCH --time=00:20:00

# configure node and process count on the CM
#SBATCH --partition=dp-esb
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=1
#SBATCH --exclusive


ml --force purge

ml Stages/2024 GCC/12.3.0 ParaStationMPI/5.9.2-1 CUDA/12 Python/3.11.3 mpi4py/3.1.4

# openml dataset id
export OPENML_ID=44973

export CUDA_VISIBLE_DEVICES="0"
export HDF5_USE_FILE_LOCKING='FALSE'

source dwave_env/bin/activate

# New CUDA drivers on the compute nodes
ln -s /usr/lib64/libcuda.so.1 .
ln -s /usr/lib64/libnvidia-ml.so.1 .
LD_LIBRARY_PATH=.:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

rm -r saved_models_r_openml
mkdir saved_models_r_openml

srun --mpi=pspmix python3 -u MPI_swift_hyperband.py --dir_name=saved_models_r_openml --seed=0 --model_name=openml_torch --r=50 --pred_type=quantum
#srun --mpi=pspmix python3 -u MPI_regular_hyperband.py --dir_name=saved_models_r_openml --seed=0 --model_name=openml_torch --r=50


remove_data() {
    rm -rf /tmp/openml
}
export -f remove_data

srun bash -c remove_data
