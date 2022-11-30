#!/bin/sh
#SBATCH --job-name=trn_rel_baseline_eval
#SBATCH --output /projects/tir5/users/nvaikunt/qgen_rnk_exp_true/log/output_eval_tst.txt
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --export=ALL
source /home/nvaikunt/miniconda3/etc/profile.d/conda.sh
conda activate baseline
cd /home/nvaikunt/PromptBasedReRanking
export QG=True
export RANKING=True
export PER_EVAL_BATCH_SZ=25
export EVAL_RUN='qg_rnk_eval_nq_test_chkpt1'
export EVAL_DIR=/projects/tir5/users/nvaikunt/downloads/data/retriever-outputs/dpr/nq-test.json
export OUTFILE=/projects/tir5/users/nvaikunt/qgen_rnk_exp_true/log/eval_metrics_baseline_nq_test_ckpt1.txt
export MODEL=/projects/tir5/users/nvaikunt/qgen_rnk_exp_true/model/checkpoint-20000
bash shell_scripts/run_eval.sh
export EVAL_RUN='qg_rnk_eval_nq_test_chkpt2'
export OUTFILE=/projects/tir5/users/nvaikunt/qgen_rnk_exp_true/log/eval_metrics_baseline_nq_test_ckpt2.txt
export MODEL=/projects/tir5/users/nvaikunt/qgen_rnk_exp_true/model/checkpoint-40000
bash shell_scripts/run_eval.sh
export EVAL_RUN='qg_rnk_eval_nq_test_chkpt3'
export OUTFILE=/projects/tir5/users/nvaikunt/qgen_rnk_exp_true/log/eval_metrics_baseline_nq_test_ckpt3.txt
export MODEL=/projects/tir5/users/nvaikunt/qgen_rnk_exp_true/model/checkpoint-60000
bash shell_scripts/run_eval.sh