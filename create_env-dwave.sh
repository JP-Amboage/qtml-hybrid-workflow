#export TMPDIR="/p/project/deepext/aach1"
#export TMPDIR="/p/project/deepext/garciaamboage1"
export TMPDIR="/p/project/deepext/$USER"

ml --force purge
ml Stages/2024 GCC/12.3.0 ParaStationMPI/5.9.2-1 CUDA/12 Python/3.11.3 mpi4py/3.1.4

python3 -m venv dwave_env

source dwave_env/bin/activate

pip3 install scikit-learn
pip3 install numpy
pip3 install pandas
pip3 install dwave-ocean-sdk
pip3 install tensorflow
#pip3 install mpi4py
pip3 install torch torchvision
pip3 install openml

yes | dwave setup

deactivate
