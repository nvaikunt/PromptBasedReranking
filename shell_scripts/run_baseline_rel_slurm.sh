#!/bin/sh
#SBATCH --job-name=trn_rel_nornk_baseline
#SBATCH --output /projects/tir5/users/nvaikunt/rel_nornk_exp/log/output.txt
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:3
#SBATCH --export=ALL
source /home/nvaikunt/miniconda3/etc/profile.d/conda.sh
conda activate baseline
cd /home/nvaikunt/PromptBasedReRanking
export PER_GPU_BATCH_SZ=45
export RANKING=FALSE
export QG=FALSE
export OUTPUT_DIR=/projects/tir5/users/nvaikunt/rel_nornk_exp/model
bash run_baseline.sh
