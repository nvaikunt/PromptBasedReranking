#!/bin/bash
python evaluate.py --isQG "$QG" --isRanking "$RANKING" --max_eval_size "$EVAL_SZ" --outfile "$OUTFILE" --evidence_dir "$EVIDENCE_DIR" --eval_data "$EVAL_DIR" -m "$MODEL" --batch_size="$EVAL_BATCH_SZ" --k "$K" --run_name "$EVAL_RUN"
