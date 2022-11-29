#!/bin/sh
#SBATCH --job-name=trn_qg_rnk_baseline_eval
#SBATCH --output /projects/tir5/users/nvaikunt/qgen_no_rnk/log/output_eval.txt
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:2
#SBATCH --export=ALL
source /home/nvaikunt/miniconda3/etc/profile.d/conda.sh
conda activate baseline
cd /home/nvaikunt/PromptBasedReRanking
export QG=True
export RANKING=False
export PER_EVAL_BATCH_SZ=25
export EVAL_DIR=/projects/tir5/users/nvaikunt/downloads/data/retriever-outputs/dpr/nq-test.json
export OUTFILE=/projects/tir5/users/nvaikunt/qgen_nornk_exp/log/eval_metrics_baseline_nq_test.txt
export MODEL=/projects/tir5/users/nvaikunt/qgen_nornk_exp/model
bash run_baseline.sh
export EVAL_DIR=/projects/tir5/users/nvaikunt/downloads/data/retriever-outputs/dpr/squad1-test.json
export OUTFILE=/projects/tir5/users/nvaikunt/qgen_nornk_exp/log/eval_metrics_baseline_squad1_test.txt
bash run_baseline.sh
export EVAL_DIR=/projects/tir5/users/nvaikunt/downloads/data/retriever-outputs/dpr/trivia-test.json
export OUTFILE=/projects/tir5/users/nvaikunt/qgen_nornk_exp/log/eval_metrics_baseline_trivia_test.txt
bash run_baseline.sh
export EVAL_DIR=/projects/tir5/users/nvaikunt/downloads/data/retriever-outputs/dpr/webq-test.json
export OUTFILE=/projects/tir5/users/nvaikunt/qgen_nornk_exp/log/eval_metrics_baseline_webq_test.txt
bash run_baseline.sh