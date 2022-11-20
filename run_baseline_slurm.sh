#!/bin/sh
#SBATCH --job-name=trn_qg_rnk_baseline
#SBATCH --output /projects/tir5/users/nvaikunt/qgen_rnk_exp/log/output.txt
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX_8000:3
#SBATCH --export=ALL
source /home/nvaikunt/miniconda3/etc/profile.d/conda.sh
conda activate baseline
cd /home/nvaikunt/PromptBasedReRanking
export QG=TRUE
export RANKING=TRUE
export PER_GPU_BATCH_SZ=45
export OUTPUT_DIR=/projects/tir5/users/nvaikunt/qgen_rnk_exp/model
bash run_baseline.sh
