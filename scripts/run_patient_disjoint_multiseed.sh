#!/usr/bin/env bash
# STAGE 28 TASK E: patient-disjoint sensitivity across the same 5-seed set used for the canonical
# multi-seed analysis (Stage 27 Task 1). Seed 42's patient-disjoint run already exists
# (results_v2/stage27_robustness/patient_sensitivity/model_run/, from Stage 27) and is reused unchanged,
# not rerun. Only seeds 11, 22, 33, 55 are new here. Identical protocol to Stage 27's own patient-disjoint
# run (code_v2/stage27_train_and_evaluate.py): same architecture, preprocessing, hyperparameters,
# early-stopping rule, evaluation code -- applied to the SAME patient-disjoint cache built in Stage 27
# (results_v2/stage27_robustness/patient_sensitivity/_patch_cache/), not rebuilt.
set -e
cd "$(dirname "$0")/../.."
for SEED in 11 22 33 55; do
  echo "=== PATIENT-DISJOINT SEED $SEED starting $(date) ==="
  venv/Scripts/python.exe code_v2/stage27_train_and_evaluate.py --seed $SEED \
    --out_dir results_v2/stage28/patient_disjoint_multiseed/seed_$SEED \
    --train_cache results_v2/stage27_robustness/patient_sensitivity/_patch_cache \
    --eval_single_split results_v2/stage27_robustness/patient_sensitivity/_patch_cache/held_out.npz
  echo "=== PATIENT-DISJOINT SEED $SEED done $(date) ==="
done
echo "ALL PATIENT-DISJOINT SEEDS DONE"
