#!/bin/sh
#SBATCH --job-name=trn_rel_nornk_prompt
#SBATCH --output /projects/tir5/users/nvaikunt/rel_nornk_exp/prompt_log/output.txt
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:2
#SBATCH --export=ALL
source /home/nvaikunt/miniconda3/etc/profile.d/conda.sh
conda activate baseline
cd /home/nvaikunt/PromptBasedReRanking
export PER_GPU_BATCH_SZ=16
export LR=1e-3
export N_EPOCHS=2
export RUN_NAME="rel_prompt"
export RANKING=FALSE
export QG=FALSE
export OUTPUT_DIR=/projects/tir5/users/nvaikunt/rel_nornk_exp/prompt_model
bash shell_scripts/run_prompt_tuning.sh
