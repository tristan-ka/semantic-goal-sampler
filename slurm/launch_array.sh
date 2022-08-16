#!/bin/sh
#SBATCH --job-name=sgs-array
#SBATCH --array=1-4
#SBATCH --ntasks-per-node=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks (we ask a lot of cpus to get more memory)
#SBATCH --hint=nomultithread         # we get physical cores not logical (to have more memory)
#SBATCH --time=00:30:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=slurm_logs/%x-%j.out           # output file name
#SBATCH --error=slurm_logs/%x-%j.err           # err file name
#SBATCH --account=imi@v100
#SBATCH --qos=qos_gpu-dev #other possible partition are t3(max 20h) and t4(max 100h)
#SBATCH -C v100-32g # we can also ask for v100-32g
#SBATCH --gres=gpu:4 

filename=$1

extract_config(){
  file=$1
  index=$2
  i=0
  while read line; do
    if [ "$i" -eq "$index" ]; then
      output=$line;
    fi
    i=$((i+1))
  done < $file
}

extract_config $filename $(SLRUM_ARRAY_TASK_ID) \

module purge
module load pytorch-gpu/py3/1.9.0
conda activate sgs

chmod +x slurm/launcher.sh

srun slurm/launcher.sh \
  rl_script_args.path=$WORK/semantic-goal-sampler/src/main.py \
  $output \
  --config-path=$WORK/semantic-goal-sampler/conf \
  --config-name=slurm_cluster_config

