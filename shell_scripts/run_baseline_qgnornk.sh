#!/bin/sh
#SBATCH --job-name=trn_qg_nornk_baseline
#SBATCH --output /projects/tir5/users/nvaikunt/qgen_nornk_exp/log2/output.txt
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
export PER_GPU_BATCH_SZ=32
export OUTPUT_DIR=/projects/tir5/users/nvaikunt/qgen_nornk_exp/model2
bash run_baseline.sh
