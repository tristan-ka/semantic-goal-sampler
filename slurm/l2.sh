#!/bin/sh
#SBATCH --job-name=sgs-pred
#SBATCH --ntasks-per-node=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks (we ask a lot of cpus to get more memory)
#SBATCH --hint=nomultithread         # we get physical cores not logical (to have more memory)
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=slurm_logs/%x-%j.out           # output file name
#SBATCH --error=slurm_logs/%x-%j.err           # err file name
#SBATCH --account=imi@v100
#SBATCH --qos=qos_gpu-t3 #other possible partition are t3(max 20h) and t4(max 100h)
#SBATCH -C v100-16g # we can also ask for v100-32g
#SBATCH --gres=gpu:4 # one gpu per task

module purge
module load pytorch-gpu/py3/1.9.0
conda activate sgs

python main.py prompt_type=predicate
