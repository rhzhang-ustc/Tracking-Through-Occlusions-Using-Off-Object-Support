#!/bin/bash
#SBATCH --partition=orion --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name="mlp_eval"
#SBATCH --output=mlp_eval-%j.out

# only use the following if you want email notification
#SBATCH --mail-user=zrh20010804@mail.ustc.edu.cn
#SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# process
python3 evaluation_mlp.py

# can try the following to list out which GPU you have access to
# srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"