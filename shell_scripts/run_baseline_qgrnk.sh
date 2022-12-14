#!/bin/sh
#SBATCH --job-name=trn_qg_rnk_baseline
#SBATCH --output /projects/tir5/users/nvaikunt/qgen_rnk_exp/log2/output.txt
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
export RANKING=True
export PER_GPU_BATCH_SZ=16
export LR=2e-5
export N_EPOCHS=3
export OUTPUT_DIR=/projects/tir5/users/nvaikunt/qgen_rnk_exp/model2
bash shell_scripts/run_baseline.sh