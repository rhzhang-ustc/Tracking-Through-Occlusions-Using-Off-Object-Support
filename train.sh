#!/bin/bash
#SBATCH --partition=orion-interactive --qos=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name="train_basic_perceiver"
#SBATCH --output=perceiver-%j.out

# only use the following if you want email notification
#SBATCH --mail-user=zrh20010804@mail.ustc.edu.cn
#SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)

python3 train_basic_perceiver.py > logs.txt

# can try the following to list out which GPU you have access to
# srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"