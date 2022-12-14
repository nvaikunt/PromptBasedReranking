#!/bin/sh
#SBATCH --job-name=trn_qg_rnk_baseline_eval
#SBATCH --output /projects/tir5/users/nvaikunt/qgen_rnk_exp_true/log2/output_eval.txt
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --export=ALL
source /home/nvaikunt/miniconda3/etc/profile.d/conda.sh
conda activate baseline
cd /home/nvaikunt/PromptBasedReRanking
export QG=True
export RANKING=True
export PER_EVAL_BATCH_SZ=10
export EVAL_RUN='qgen_rnk_eval_nq_test'
export EVAL_DIR=/projects/tir5/users/nvaikunt/downloads/data/retriever-outputs/dpr/nq-test.json
export OUTFILE=/projects/tir5/users/nvaikunt/qgen_rnk_exp_true/log2/eval_metrics_baseline_nq_test.txt
export MODEL=/projects/tir5/users/nvaikunt/qgen_rnk_exp_true/model2/checkpoint-60000
bash shell_scripts/run_eval.sh
export EVAL_RUN='qgen_rnk_eval_squad1_test'
export EVAL_DIR=/projects/tir5/users/nvaikunt/downloads/data/retriever-outputs/dpr/squad1-test.json
export OUTFILE=/projects/tir5/users/nvaikunt/qgen_rnk_exp_true/log2/eval_metrics_baseline_squad1_test.txt
bash shell_scripts/run_eval.sh
export EVAL_RUN='qgen_rnk_eval_trivia_test'
export EVAL_DIR=/projects/tir5/users/nvaikunt/downloads/data/retriever-outputs/dpr/trivia-test.json
export OUTFILE=/projects/tir5/users/nvaikunt/qgen_rnk_exp_true/log2/eval_metrics_baseline_trivia_test.txt
bash shell_scripts/run_eval.sh
export EVAL_RUN='qgen_rnk_eval_webq_test'
export EVAL_DIR=/projects/tir5/users/nvaikunt/downloads/data/retriever-outputs/dpr/webq-test.json
export OUTFILE=/projects/tir5/users/nvaikunt/qgen_rnk_exp_true/log2/eval_metrics_baseline_webq_test.txt
bash shell_scripts/run_eval.sh
