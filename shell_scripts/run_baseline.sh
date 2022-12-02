#!/bin/bash
python baseline_train.py -ep "$N_EPOCHS" -bs "$PER_GPU_BATCH_SZ" -trsz "$TRAIN_EX" -vlsz "$VAL_SZ" -odir "$OUTPUT_DIR" \
-edir "$EVIDENCE_DIR" -tdir "$TRAIN_DIR" -vdir "$VAL_DIR" -m "$MODEL" --isQG "$QG" \
--isRanking "$RANKING" --run_name "$RUN_NAME" --push "$PUSH" --grad_accumulation_steps "$GRAD_ACCUM" -lr "$LR"
