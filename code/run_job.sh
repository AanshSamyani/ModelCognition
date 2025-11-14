#!/bin/bash
#SBATCH --job-name=get_log_probs          # Job name
#SBATCH --partition=airawatp       # Partition
#SBATCH --cpus-per-task=32         # Number of CPU cores per task
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --nodes=1                  # Use 1 node
#SBATCH --ntasks=1                 # Use 1 task
#SBATCH --time=21:00:00            # 21 hours
#SBATCH --nodelist=scn70-10g      # Force job to run on this node
#SBATCH --output=output.out        # Stdout (%j = job ID)
#SBATCH --error=output.err         # Stderr

# Run your process
/nlsasfs/home/isea/isea10/anaconda3/envs/newenv/bin/python experiment1.py


