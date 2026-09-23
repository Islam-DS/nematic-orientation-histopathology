"""
STAGE 14: assembles the required master results table (CSV + JSON) from
each experiment's training summary and held-out evaluation JSON. Uses NA
for any metric not applicable to a given experiment (e.g. A0 has no
Stage-14 training summary since it reuses the frozen Stage 10 artifact).
"""
import os
import json
import csv

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
STAGE14_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage14")
STAGE10_SUMMARY = os.path.join(PROJECT_ROOT, "results_v2", "model3", "metrics", "stage10_summary.json")

NA = "NA"


def get(d, *keys, default=NA):
    cur = d
    for k in keys:
        if cur is None or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_row(exp_id, arch, loss_condition, seed, train_summary, eval_result, notes):
    param_count = get(train_summary, "n_params") if train_summary else NA
    best_epoch = get(train_summary, "best_epoch") if train_summary else NA
    best_dev_loss = get(train_summary, "best_dev_loss") if train_summary else NA

    testA_auroc = get(eval_result, "testA", "classification_metrics", "auroc") if eval_result else NA
    testB_auroc = get(eval_result, "testB", "classification_metrics", "auroc") if eval_result else NA
    testA_ang = get(eval_result, "testA", "geometric_metrics", "A_angular_correspondence", "mean_deg") if eval_result else NA
    testB_ang = get(eval_result, "testB", "geometric_metrics", "A_angular_correspondence", "mean_deg") if eval_result else NA
    testA_glass = get(eval_result, "testA", "geometric_metrics", "A_angular_correspondence", "glass_delta") if eval_result else NA
    testB_glass = get(eval_result, "testB", "geometric_metrics", "A_angular_correspondence", "glass_delta") if eval_result else NA
    pooled_r = get(eval_result, "pooled_testA_testB", "geometric_metrics", "B_order_magnitude_correlation", "pearson_r") if eval_result else NA
    pooled_r_ci = get(eval_result, "pooled_testA_testB", "geometric_metrics", "B_order_magnitude_correlation", "bootstrap_95CI") if eval_result else NA
    tensor_sim = get(eval_result, "pooled_testA_testB", "geometric_metrics", "C_tensor_similarity", "mean_D_Q") if eval_result else NA

    return {
        "experiment_id": exp_id, "architecture": arch, "loss_condition": loss_condition, "seed": seed,
        "parameter_count": param_count, "best_epoch": best_epoch, "best_dev_loss": best_dev_loss,
        "testA_AUROC": testA_auroc, "testB_AUROC": testB_auroc,
        "testA_angular_error_deg": testA_ang, "testB_angular_error_deg": testB_ang,
        "testA_Glass_delta": testA_glass, "testB_Glass_delta": testB_glass,
        "pooled_r": pooled_r, "pooled_r_95CI": json.dumps(pooled_r_ci) if pooled_r_ci != NA else NA,
        "tensor_similarity_mean_D_Q": tensor_sim,
        "notes": notes,
    }


def main():
    rows = []

    stage10_summary = load_json(STAGE10_SUMMARY)
    a0_eval = load_json(os.path.join(STAGE14_DIR, "A0_reference", "held_out_evaluation.json"))
    rows.append(build_row("A0", "Model3TypedEquivariant", "lambda_class=1,lambda_Q=1 (Stage 10 reference)", 42,
                           stage10_summary, a0_eval, "reuses exact frozen Stage 10 checkpoint, no retraining"))

    for exp_dir, exp_id, arch, loss_cond, notes in [
        ("A1_no_q", "A1", "Model3TypedEquivariant", "lambda_class=1,lambda_Q=0", "Q supervision removed; Q metrics are a descriptive control, Q head untrained"),
        ("A3_q_dominant", "A3", "Model3TypedEquivariant", "lambda_class=1,lambda_Q=10", "Q-dominant weighting, 10x baseline, pre-specified before execution"),
        ("A4_q_only", "A4", "Model3TypedEquivariant", "lambda_class=0,lambda_Q=1", "classification removed; classification metrics are a descriptive control, classifier untrained"),
        ("A5_plain_cnn_control", "A5", "PlainCNNDualHead(c1=38,c2=9)", "lambda_class=1,lambda_Q=1", "parameter-matched (11676 vs 11674) non-equivariant control"),
    ]:
        train_summary = load_json(os.path.join(STAGE14_DIR, exp_dir, "stage14_train_summary.json"))
        eval_result = load_json(os.path.join(STAGE14_DIR, exp_dir, "held_out_evaluation.json"))
        rows.append(build_row(exp_id, arch, loss_cond, 42, train_summary, eval_result, notes))

    # A6 and A7 are diagnostic/evidential, not new trained models with the same metric shape -- represented as NA rows with notes.
    a6 = load_json(os.path.join(STAGE14_DIR, "A6_spatial_reduction", "a6_results.json"))
    a6_note = "diagnostic only, no training; see A6_spatial_reduction/a6_results.json for per-angle rotation-consistency under R1/R2 reductions"
    rows.append({"experiment_id": "A6", "architecture": "Model3TypedEquivariant (frozen A0 checkpoint)", "loss_condition": "N/A",
                 "seed": "N/A", "parameter_count": get(stage10_summary, "n_params"), "best_epoch": "N/A", "best_dev_loss": "N/A",
                 "testA_AUROC": NA, "testB_AUROC": NA, "testA_angular_error_deg": NA, "testB_angular_error_deg": NA,
                 "testA_Glass_delta": NA, "testB_Glass_delta": NA, "pooled_r": NA, "pooled_r_95CI": NA,
                 "tensor_similarity_mean_D_Q": NA, "notes": a6_note})

    plain_cnn_stage9 = load_json(os.path.join(PROJECT_ROOT, "results_v2", "baselines", "plain_cnn", "held_out_testA_results.json"))
    a7_note = ("evidential reuse of Stage 9 plain_cnn (c1=64,c2=64, jointly trained) + A4 (Q-only Model3); "
               "no dedicated target-learnability control exists (documented limitation, see stage14_ablation_matrix.json)")
    rows.append({"experiment_id": "A7", "architecture": "N/A (evidence reuse)", "loss_condition": "N/A",
                 "seed": "N/A", "parameter_count": "N/A", "best_epoch": "N/A", "best_dev_loss": "N/A",
                 "testA_AUROC": NA, "testB_AUROC": NA, "testA_angular_error_deg": NA, "testB_angular_error_deg": NA,
                 "testA_Glass_delta": NA, "testB_Glass_delta": NA, "pooled_r": NA, "pooled_r_95CI": NA,
                 "tensor_similarity_mean_D_Q": NA, "notes": a7_note})

    out_json = os.path.join(STAGE14_DIR, "stage14_master_results.json")
    out_csv = os.path.join(STAGE14_DIR, "stage14_master_results.csv")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main()
