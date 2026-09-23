"""
STAGE 28: assemble the machine-readable results_v2/stage28/stage28_summary.json required by the
brief. Reads only already-written Stage 28 (and reused Stage 27) result files; writes only
results_v2/stage28/stage28_summary.json.
"""
import json
import os
import statistics

ROOT = os.path.join(os.path.dirname(__file__), "..")
S28 = os.path.join(ROOT, "results_v2", "stage28")
SR27 = os.path.join(ROOT, "results_v2", "stage27_robustness")

sdist = json.load(open(os.path.join(S28, "s_distribution", "s_distribution_summary.json")))
gland = json.load(open(os.path.join(S28, "gland_patch_size", "gland_patch_size_summary.json")))

PD_SEEDS = [11, 22, 33, 42, 55]
pd_rows = {}
for sd in PD_SEEDS:
    if sd == 42:
        p = os.path.join(SR27, "patient_sensitivity", "model_run", "evaluation_results.json")
        tp = os.path.join(SR27, "patient_sensitivity", "model_run", "training_summary.json")
    else:
        p = os.path.join(S28, "patient_disjoint_multiseed", f"seed_{sd}", "evaluation_results.json")
        tp = os.path.join(S28, "patient_disjoint_multiseed", f"seed_{sd}", "training_summary.json")
    ev, tr = json.load(open(p)), json.load(open(tp))
    pooled = ev["criteria_A_B_C"]["pooled"]["metrics"]
    d = ev["criterion_D"]; cls = ev["classification"]["eval_pooled"]
    pd_rows[sd] = {
        "epochs": tr["n_epochs_run"], "glass_delta_A": pooled["A_angular_correspondence"]["glass_delta"],
        "pearson_r_B": pooled["B_order_magnitude_correlation"]["pearson_r"],
        "glass_delta_C": pooled["C_tensor_similarity"]["glass_delta"], "rotation_passed_D": d["n_angles_passed"],
        "classification_accuracy": cls["accuracy"], "classification_auroc": cls["auroc"],
        "criteria_passed_of_4": sum([pooled["A_angular_correspondence"]["PASSES_PRIMARY_glass_delta_ge_0.5"],
                                      pooled["B_order_magnitude_correlation"]["PASSES_PRIMARY_ALL_THREE"],
                                      pooled["C_tensor_similarity"]["PASSES_PRIMARY_BOTH"], d["PASSES_criterion_D"]]),
    }

canon_epochs = {}
for sd in (11, 22, 33, 55):
    canon_epochs[sd] = json.load(open(os.path.join(SR27, "multi_seed", f"seed_{sd}", "training_summary.json")))["n_epochs_run"]
canon_epochs[42] = json.load(open(os.path.join(ROOT, "results_v2", "model3", "metrics", "stage10_summary.json")))["n_epochs_run"]
canon_epoch_vals = [canon_epochs[sd] for sd in PD_SEEDS]
pd_epoch_vals = [pd_rows[sd]["epochs"] for sd in PD_SEEDS]

summary = {
    "stage": 28,
    "date": "2026-09-23",
    "objective": "Strengthen interpretation, quantify remaining uncertainty, remove reviewer vulnerabilities for the frozen negative/inconclusive primary result. No frozen hypothesis, threshold, architecture, or test-set definition was changed.",
    "task_B_s_distribution": sdist,
    "task_D_gland_patch_size": gland,
    "task_E_patient_disjoint_multiseed": {
        "seeds": PD_SEEDS, "seed_42_reused_from_stage27_unchanged": True,
        "per_seed": pd_rows,
        "n_criteria_passed_total_of_20": sum(pd_rows[sd]["criteria_passed_of_4"] for sd in PD_SEEDS),
        "auroc_mean": statistics.mean([pd_rows[sd]["classification_auroc"] for sd in PD_SEEDS]),
        "auroc_sd": statistics.stdev([pd_rows[sd]["classification_auroc"] for sd in PD_SEEDS]),
    },
    "task_F_multiseed_early_stopping": {
        "canonical_split": {"per_seed_epochs": canon_epochs, "mean": statistics.mean(canon_epoch_vals),
                             "sd": statistics.stdev(canon_epoch_vals), "min": min(canon_epoch_vals), "max": max(canon_epoch_vals)},
        "patient_disjoint_split": {"per_seed_epochs": {sd: pd_rows[sd]["epochs"] for sd in PD_SEEDS},
                                    "mean": statistics.mean(pd_epoch_vals), "sd": statistics.stdev(pd_epoch_vals),
                                    "min": min(pd_epoch_vals), "max": max(pd_epoch_vals)},
    },
    "task_C_root_cause_matrix_file": "results_v2/stage28/root_cause_evidence_matrix.md",
    "task_A_literature_verification_file": "results_v2/stage28/literature_verification.md",
    "task_H_authorship": {
        "current_marker": "shared '*' denotes 'Corresponding authors' only (manuscript's own footnote text)",
        "equal_contribution_statement_present": False,
        "documented_roles_symmetric": False,
        "flagged_for_author_decision": True,
        "note": "No equal-contribution claim exists in the manuscript at any stage; none was added or removed. The documented CRediT roles are asymmetric (Islam: 9 substantive roles; Yerkos: supervisory/review roles only), consistent with a supervisor-student relationship and not in tension with either author being a corresponding author.",
    },
    "frozen_artifacts_reverified": {
        "results_v2/model3/checkpoints/checkpoint_final.pt": "5495ddeba84d3169c2a6b63ad22fbdcb",
        "results_v2/phase2/preregistered_thresholds_v2.json": "4d68f3877d2cc0ec50aac19629182740",
        "results_v2/manuscript_stage23/01_FULL_MANUSCRIPT.md": "00036b2a4dbda394e2a798247c873434",
    },
    "reproducibility": {
        "python": "3.11.8", "pytorch": "2.6.0+cu124", "cuda": "12.4", "gpu": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        "escnn": "1.0.11", "repository_state": "not a git repository",
        "determinism_claimed": False,
    },
    "final_status": "GO",
}

out_path = os.path.join(S28, "stage28_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("wrote", out_path)
print(json.dumps(summary["task_E_patient_disjoint_multiseed"], indent=2, default=str))
