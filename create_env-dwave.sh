#export TMPDIR="/p/project/deepext/aach1"
#export TMPDIR="/p/project/deepext/garciaamboage1"
export TMPDIR="/p/project/deepext/$USER"

ml --force purge
##ml Stages/2022 GCC/11.2.0 OpenMPI/4.1.2 cuDNN/8.3.1.22-CUDA-11.5 NCCL/2.11.4-CUDA-11.5 Python/3.9.6 TensorFlow
ml Stages/2023 GCC/11.3.0 OpenMPI/4.1.4 cuDNN/8.6.0.163-CUDA-11.7 Python/3.10.4

python3 -m venv dwave_env

source dwave_env/bin/activate

pip3 install scikit-learn
pip3 install numpy
pip3 install pandas
pip3 install dwave-ocean-sdk
pip3 install tensorflow
pip3 install mpi4py

yes | dwave setup

deactivate