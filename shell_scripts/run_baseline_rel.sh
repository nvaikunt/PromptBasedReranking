#!/bin/sh
#SBATCH --job-name=trn_rel_nornk_baseline
#SBATCH --output /projects/tir5/users/nvaikunt/rel_nornk_exp/log2/output.txt
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:2
#SBATCH --export=ALL
source /home/nvaikunt/miniconda3/etc/profile.d/conda.sh
conda activate baseline
cd /home/nvaikunt/PromptBasedReRanking
export PER_GPU_BATCH_SZ=16
export LR=3e-4
export N_EPOCHS=3
export RANKING=FALSE
export QG=FALSE
export OUTPUT_DIR=/projects/tir5/users/nvaikunt/rel_nornk_exp/model2
bash shell_scripts/run_baseline.sh
