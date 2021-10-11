#! /bin/bash
#SBATCH -p plgrid-gpu-v100
#SBATCH -A plggeogpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH -c 2
#SBATCH -e /net/scratch/people/plgkrusek/patenty/log%x.%a.err
#SBATCH -o /net/scratch/people/plgkrusek/patentylog%x.%a.out


cd $SCRATCH/patenty

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/net/software/local/cuda/11.2
#module load plgrid/libs/tensorflow-gpu/2.3.1-python-3.8
module load plgrid/libs/tensorflow-gpu/2.6.0-python-3.9
module load plgrid/tools/python/3.9

mkdir -p gen/bootstrap

python3 code/gravity.py \
		--nboot 500 \
		--pickle dane/clean.pickle \
		--out gen/bootstrap \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--treinablezero