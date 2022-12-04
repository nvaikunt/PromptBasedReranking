#!/bin/sh
#SBATCH --job-name=trn_qg_nornk_prompt
#SBATCH --output /projects/tir5/users/nvaikunt/qgen_nornk_exp/prompt_log/output.txt
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:2
#SBATCH --export=ALL
source /home/nvaikunt/miniconda3/etc/profile.d/conda.sh
conda activate baseline
cd /home/nvaikunt/PromptBasedReRanking
export QG=True
export RANKING=False
export LR=7e-4
export N_EPOCHS=2
export RUN_NAME="qgnornk_prompt"
export PER_GPU_BATCH_SZ=16
export OUTPUT_DIR=/projects/tir5/users/nvaikunt/qgen_nornk_exp/prompt_model
bash shell_scripts/run_prompt_tuning.sh