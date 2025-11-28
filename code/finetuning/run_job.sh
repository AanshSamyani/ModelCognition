#!/bin/bash
#SBATCH --job-name=finetuning
#SBATCH --partition=airawatp
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=21:00:00
#SBATCH --nodelist=scn71-10g
#SBATCH --output=output.out
#SBATCH --error=output.err

source /nlsasfs/home/isea/isea10/anaconda3/etc/profile.d/conda.sh
conda activate newenv

cd /nlsasfs/home/isea/isea10/aansh/introspection/code/finetuning

axolotl train llama3_1b.yml
