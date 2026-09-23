"""
STAGE 27 TASK 1: compile the five-seed multi-seed robustness study into a single CSV and a summary
JSON (mean +/- SD, 95% CI via normal approximation given n=5, min, max). Reads only the per-seed
evaluation_results.json / training_summary.json files already written under
results_v2/stage27_robustness/multi_seed/seed_*/. Writes only into
results_v2/stage27_robustness/multi_seed/.
"""
import csv
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
MS_DIR = os.path.join(ROOT, "results_v2", "stage27_robustness", "multi_seed")
SEEDS = [11, 22, 33, 42, 55]


def get(d, *path):
    for p in path:
        d = d[p]
    return d


def main():
    rows = []
    for seed in SEEDS:
        d = os.path.join(MS_DIR, f"seed_{seed}")
        ev_path = os.path.join(d, "evaluation_results.json")
        tr_path = os.path.join(d, "training_summary.json")
        if not os.path.isfile(ev_path):
            print(f"[compile] WARNING: missing {ev_path}, skipping seed {seed}")
            continue
        ev = json.load(open(ev_path))
        tr = json.load(open(tr_path)) if os.path.isfile(tr_path) else {}
        pooled = ev["criteria_A_B_C"]["pooled"]["metrics"]
        d_crit = ev["criterion_D"]
        cls = ev["classification"]["eval_pooled"]
        hist_path = os.path.join(d, "training", "training_history.json")
        best_dev_L_class = best_dev_L_Q = None
        if os.path.isfile(hist_path) and tr.get("best_dev_loss") is not None:
            hist = json.load(open(hist_path))
            best_row = min(hist, key=lambda r: abs(r["dev_loss"] - tr["best_dev_loss"]))
            best_dev_L_class, best_dev_L_Q = best_row["dev_L_class"], best_row["dev_L_Q"]
        rows.append({
            "seed": seed,
            "n_epochs_run": tr.get("n_epochs_run"),
            "train_duration_sec": tr.get("train_duration_sec"),
            "L_initial_total": tr.get("L_initial_total"),
            "L_final_total": tr.get("L_final_total"),
            "best_dev_L_class_classification_loss": best_dev_L_class,
            "best_dev_L_Q_qtensor_loss": best_dev_L_Q,
            "best_dev_loss": tr.get("best_dev_loss"),
            "A_glass_delta": pooled["A_angular_correspondence"]["glass_delta"],
            "A_mean_angular_error_deg": pooled["A_angular_correspondence"]["mean_deg"],
            "A_permutation_p": pooled["A_angular_correspondence"]["permutation_p_value"],
            "A_PASSES": pooled["A_angular_correspondence"]["PASSES_PRIMARY_glass_delta_ge_0.5"],
            "B_pearson_r": pooled["B_order_magnitude_correlation"]["pearson_r"],
            "B_pearson_p": pooled["B_order_magnitude_correlation"]["pearson_p"],
            "B_spearman_rho": pooled["B_order_magnitude_correlation"]["spearman_rho"],
            "B_PASSES": pooled["B_order_magnitude_correlation"]["PASSES_PRIMARY_ALL_THREE"],
            "C_glass_delta": pooled["C_tensor_similarity"]["glass_delta"],
            "C_mean_D_Q": pooled["C_tensor_similarity"]["mean_D_Q"],
            "C_permutation_p": pooled["C_tensor_similarity"]["permutation_p_value"],
            "C_PASSES": pooled["C_tensor_similarity"]["PASSES_PRIMARY_BOTH"],
            "D_n_angles_passed": d_crit["n_angles_passed"],
            "D_PASSES": d_crit["PASSES_criterion_D"],
            "eval_accuracy": cls["accuracy"],
            "eval_auroc": cls["auroc"],
            "n_criteria_passed_of_4": sum([
                pooled["A_angular_correspondence"]["PASSES_PRIMARY_glass_delta_ge_0.5"],
                pooled["B_order_magnitude_correlation"]["PASSES_PRIMARY_ALL_THREE"],
                pooled["C_tensor_similarity"]["PASSES_PRIMARY_BOTH"],
                d_crit["PASSES_criterion_D"],
            ]),
        })

    fieldnames = list(rows[0].keys()) if rows else []
    csv_path = os.path.join(MS_DIR, "STAGE27_MULTI_SEED_RESULTS.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[compile] wrote {csv_path} ({len(rows)} seeds)")

    numeric_fields = ["A_glass_delta", "A_mean_angular_error_deg", "B_pearson_r", "B_spearman_rho",
                       "C_glass_delta", "C_mean_D_Q", "D_n_angles_passed", "eval_accuracy", "eval_auroc",
                       "n_criteria_passed_of_4", "n_epochs_run", "train_duration_sec",
                       "best_dev_L_class_classification_loss", "best_dev_L_Q_qtensor_loss"]
    summary = {"n_seeds": len(rows), "seeds": [r["seed"] for r in rows]}
    for field in numeric_fields:
        vals = np.array([r[field] for r in rows], dtype=float)
        n = len(vals)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 1 else 0.0
        # 95% CI via normal approximation (n=5; small-sample caveat noted in the report, not hidden)
        ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
        summary[field] = {"mean": mean, "sd": sd, "min": float(vals.min()), "max": float(vals.max()),
                           "ci95_normal_approx": [float(ci_lo), float(ci_hi)], "values_by_seed": {str(r["seed"]): r[field] for r in rows}}

    summary["n_of_9_seeds_criteria_passed_summary"] = {
        "A_passed_count": int(sum(r["A_PASSES"] for r in rows)),
        "B_passed_count": int(sum(r["B_PASSES"] for r in rows)),
        "C_passed_count": int(sum(r["C_PASSES"] for r in rows)),
        "D_passed_count": int(sum(r["D_PASSES"] for r in rows)),
        "n_seeds": len(rows),
    }
    with open(os.path.join(MS_DIR, "STAGE27_MULTI_SEED_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["n_of_9_seeds_criteria_passed_summary"], indent=2))
    for field in ["A_glass_delta", "B_pearson_r", "C_glass_delta", "D_n_angles_passed", "eval_auroc"]:
        s = summary[field]
        print(f"{field}: mean={s['mean']:.4f} sd={s['sd']:.4f} min={s['min']:.4f} max={s['max']:.4f} "
              f"95%CI=[{s['ci95_normal_approx'][0]:.4f}, {s['ci95_normal_approx'][1]:.4f}]")
    return summary


if __name__ == "__main__":
    main()
