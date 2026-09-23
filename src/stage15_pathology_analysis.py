"""
STAGE 15: pathology / morphology association analysis. Executes EXACTLY
the pre-frozen, checksummed matrix in
configs/stage15_pathology_analysis_matrix.json (P1-P4 primary pathology,
M1-M2 morphology, C1 confound adjustment, R1 patient-level robustness).

Uses ONLY the frozen Stage 10 Model 3 checkpoint (no retraining) and the
existing, unmodified Stage 9/11 inference pipeline to produce S_DL/phi_DL
per patch, then aggregates to IMAGE level (the natural unit for the
image-level grade labels) before any statistical test -- patches are never
treated as independent observations.
"""
import sys
import os
import json
import csv

import numpy as np
import pandas as pd
from scipy import stats
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant
from stage9_train_baselines import to_tensors, per_patch_scalar_qs
from stage11_geometric_validation import get_predictions

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results_v2", "model3", "checkpoints", "checkpoint_final.pt")
MATRIX_PATH = os.path.join(PROJECT_ROOT, "configs", "stage15_pathology_analysis_matrix.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "results_v2", "baselines", "_patch_cache")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data_manifests", "glas_manifest_stage3_finalized.csv")
GLAND_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "results_v2", "anatomy_targets", "stage4_gland_manifest.csv")
GRADE_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "glas", "Warwick_QU_Dataset", "Grade.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage15")

N_BOOT = 2000
BOOT_SEED = 42
ORDINAL_ORDER = ["healthy", "adenomatous", "moderately differentiated",
                  "moderately-to-poorly differentated", "poorly differentiated"]


# ---------------------------------------------------------------- data loading

def load_split_testAB(split_name):
    d = np.load(os.path.join(CACHE_DIR, f"{split_name}.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_grade_csv():
    df = pd.read_csv(GRADE_CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["name"] = df["name"].str.strip()
    df["grade (GlaS)"] = df["grade (GlaS)"].str.strip()
    df["grade (Sirinukunwattana et al. 2015)"] = df["grade (Sirinukunwattana et al. 2015)"].str.strip()
    df["ordinal_grade"] = df["grade (Sirinukunwattana et al. 2015)"].apply(lambda v: ORDINAL_ORDER.index(v))
    df["grade_label"] = (df["grade (GlaS)"] == "malignant").astype(int)
    return df.set_index("name")


def load_gland_morphology():
    g = pd.read_csv(GLAND_MANIFEST_PATH)
    per_image = g.groupby("image_id").agg(
        mean_gland_area=("n_pixels", "mean"),
        median_gland_area=("n_pixels", "median"),
        n_glands_in_manifest=("gland_id", "count"),
        n_fragmented=("fragmented", "sum"),
    )
    manifest = pd.read_csv(MANIFEST_PATH).set_index("image_id")
    per_image["n_gland_instances"] = manifest["n_gland_instances"]
    return per_image


# ---------------------------------------------------------------- representation extraction

@torch.no_grad()
def compute_image_level_representation(model, device):
    """Returns a DataFrame indexed by image_id with S_DL (image mean) and
    phi_DL circular dispersion, computed from patch-level predictions via
    the EXACT existing Stage 9/11 pipeline (no retraining)."""
    rows = []
    for split_name in ["testA", "testB"]:
        arrs = load_split_testAB(split_name)
        assert all(str(i).startswith(split_name + "_") for i in arrs["image_id"]), \
            f"provenance check failed for {split_name}"
        q1_pred, q2_pred, q1_t, q2_t, valid = get_predictions(model, arrs, device)
        S_pred, phi_pred, S_t, phi_t, has_valid = per_patch_scalar_qs(q1_pred, q2_pred, q1_t, q2_t, valid)

        image_ids = arrs["image_id"]
        for iid in sorted(set(image_ids)):
            m = (image_ids == iid) & has_valid
            n_valid = int(m.sum())
            if n_valid == 0:
                rows.append({"image_id": iid, "split": split_name, "n_patches_valid": 0,
                             "S_DL_mean": np.nan, "phi_DL_dispersion": np.nan})
                continue
            S_img = S_pred[m]
            phi_img = phi_pred[m]
            S_DL_mean = float(S_img.mean())
            mean_vec = np.mean(np.exp(1j * 2 * phi_img))
            phi_dispersion = float(1.0 - np.abs(mean_vec))
            rows.append({"image_id": iid, "split": split_name, "n_patches_valid": n_valid,
                         "S_DL_mean": S_DL_mean, "phi_DL_dispersion": phi_dispersion})
    return pd.DataFrame(rows).set_index("image_id")


# ---------------------------------------------------------------- statistics helpers

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / (pooled_sd + 1e-12))


def glass_delta(x_treatment, x_reference):
    return float((x_treatment.mean() - x_reference.mean()) / (x_reference.std(ddof=1) + 1e-12))


def bootstrap_mean_diff_ci(x, y, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        bx = x[rng.integers(0, len(x), len(x))]
        by = y[rng.integers(0, len(y), len(y))]
        diffs[i] = bx.mean() - by.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_spearman_ci(x, y, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rhos[i] = stats.spearmanr(x[idx], y[idx]).correlation
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(lo), float(hi)


def run_binary_test(analysis_id, rep_values, grade_labels, unit_n):
    x_mal = rep_values[grade_labels == 1]
    x_ben = rep_values[grade_labels == 0]
    if len(x_mal) < 3 or len(x_ben) < 3:
        return {"analysis_id": analysis_id, "note": "too few observations in one group", "n": unit_n}
    u_stat, p = stats.mannwhitneyu(x_mal, x_ben, alternative="two-sided")
    d = cohens_d(x_mal, x_ben)
    delta = glass_delta(x_mal, x_ben)
    ci_lo, ci_hi = bootstrap_mean_diff_ci(x_mal, x_ben)
    return {
        "analysis_id": analysis_id, "test": "Mann-Whitney U (two-sided)",
        "n_malignant": int(len(x_mal)), "n_benign": int(len(x_ben)),
        "mean_malignant": float(x_mal.mean()), "mean_benign": float(x_ben.mean()),
        "cohens_d": d, "glass_delta_malignant_vs_benign_ref": delta,
        "mean_diff_95CI_bootstrap": [ci_lo, ci_hi],
        "U_statistic": float(u_stat), "p_value": float(p),
    }


def run_ordinal_test(analysis_id, rep_values, ordinal_grades, unit_n):
    if len(rep_values) < 5:
        return {"analysis_id": analysis_id, "note": "too few observations", "n": unit_n}
    rho, p = stats.spearmanr(rep_values, ordinal_grades)
    ci_lo, ci_hi = bootstrap_spearman_ci(np.asarray(rep_values), np.asarray(ordinal_grades))
    return {
        "analysis_id": analysis_id, "test": "Spearman rank correlation (two-sided)",
        "n": int(len(rep_values)), "spearman_rho": float(rho), "p_value": float(p),
        "rho_95CI_bootstrap": [ci_lo, ci_hi],
    }


def holm_bonferroni(pvals_dict):
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(items, start=1):
        adj = min(1.0, (m - rank + 1) * p)
        running_max = max(running_max, adj)
        adjusted[name] = float(running_max)
    return adjusted


# ---------------------------------------------------------------- main

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage15] device = {device}")

    with open(MATRIX_PATH) as f:
        matrix = json.load(f)
    print(f"[stage15] loaded FROZEN analysis matrix: {MATRIX_PATH}")

    model = Model3TypedEquivariant().to(device)
    model.eval()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"[stage15] loaded frozen Stage 10 checkpoint (no retraining): {CHECKPOINT_PATH}")

    rep_df = compute_image_level_representation(model, device)
    print(f"[stage15] computed image-level S_DL / phi_DL-dispersion for {len(rep_df)} images "
          f"(testA={sum(rep_df.split=='testA')}, testB={sum(rep_df.split=='testB')})")

    grade_df = load_grade_csv()
    morph_df = load_gland_morphology()

    df = rep_df.join(grade_df[["grade_label", "ordinal_grade", "patient ID"]], how="left")
    df = df.join(morph_df, how="left")
    df = df.dropna(subset=["S_DL_mean", "phi_DL_dispersion", "grade_label", "ordinal_grade"])
    print(f"[stage15] merged analysis table: {len(df)} images with complete representation+label+morphology data")

    df.to_csv(os.path.join(OUT_DIR, "tables", "stage15_image_level_table.csv"))

    results = {"n_images_total": int(len(df)), "n_testA": int((df.split == "testA").sum()),
               "n_testB": int((df.split == "testB").sum()), "n_distinct_patients": int(df["patient ID"].nunique())}

    # ---- PRIMARY (pooled) ----
    primary_pooled = {}
    primary_pooled["P1"] = run_binary_test("P1", df["S_DL_mean"].values, df["grade_label"].values, len(df))
    primary_pooled["P2"] = run_ordinal_test("P2", df["S_DL_mean"].values, df["ordinal_grade"].values, len(df))
    primary_pooled["P3"] = run_binary_test("P3", df["phi_DL_dispersion"].values, df["grade_label"].values, len(df))
    primary_pooled["P4"] = run_ordinal_test("P4", df["phi_DL_dispersion"].values, df["ordinal_grade"].values, len(df))
    results["primary_pooled"] = primary_pooled

    pvals = {k: v["p_value"] for k, v in primary_pooled.items() if "p_value" in v}
    results["holm_bonferroni_adjusted_p"] = holm_bonferroni(pvals)

    # ---- PRIMARY (testA / testB separately, raw p-values, NOT part of the corrected family) ----
    primary_by_split = {}
    for split_name in ["testA", "testB"]:
        sub = df[df.split == split_name]
        primary_by_split[split_name] = {
            "P1": run_binary_test("P1", sub["S_DL_mean"].values, sub["grade_label"].values, len(sub)),
            "P2": run_ordinal_test("P2", sub["S_DL_mean"].values, sub["ordinal_grade"].values, len(sub)),
            "P3": run_binary_test("P3", sub["phi_DL_dispersion"].values, sub["grade_label"].values, len(sub)),
            "P4": run_ordinal_test("P4", sub["phi_DL_dispersion"].values, sub["ordinal_grade"].values, len(sub)),
        }
    results["primary_by_split"] = primary_by_split

    # ---- MORPHOLOGY (pooled) ----
    morphology = {}
    morphology["M1"] = run_ordinal_test("M1", df["S_DL_mean"].values, df["mean_gland_area"].values, len(df))
    morphology["M1"]["endpoint"] = "mean_gland_area"
    morphology["M2"] = run_ordinal_test("M2", df["S_DL_mean"].values, df["n_gland_instances"].values, len(df))
    morphology["M2"]["endpoint"] = "n_gland_instances"
    results["morphology"] = morphology

    # ---- CONFOUND ADJUSTMENT (C1) ---- (closed-form OLS via numpy; statsmodels not
    # available in the frozen project environment, avoided rather than adding a new dependency)
    y = df["S_DL_mean"].astype(float).values
    x1 = df["grade_label"].astype(float).values
    x2 = df["mean_gland_area"].astype(float).values
    n_obs = len(y)
    X = np.column_stack([np.ones(n_obs), x1, x2])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n_obs - X.shape[1]
    sigma2 = float((resid @ resid) / dof)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_stats = beta / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=dof))
    t_crit = stats.t.ppf(0.975, df=dof)
    ci_lo = beta - t_crit * se
    ci_hi = beta + t_crit * se
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1 - ss_res / ss_tot

    unadjusted_binary = primary_pooled["P1"]
    results["confound_adjustment_C1"] = {
        "model": "S_DL_mean ~ grade_label + mean_gland_area (OLS, closed-form via numpy)",
        "n": int(n_obs), "dof": int(dof),
        "grade_label_coef": float(beta[1]), "grade_label_se": float(se[1]),
        "grade_label_p": float(p_vals[1]), "grade_label_95CI": [float(ci_lo[1]), float(ci_hi[1])],
        "mean_gland_area_coef": float(beta[2]), "mean_gland_area_se": float(se[2]),
        "mean_gland_area_p": float(p_vals[2]), "mean_gland_area_95CI": [float(ci_lo[2]), float(ci_hi[2])],
        "intercept": float(beta[0]),
        "r_squared": float(r_squared),
        "unadjusted_P1_cohens_d": unadjusted_binary.get("cohens_d"),
        "unadjusted_P1_p_value": unadjusted_binary.get("p_value"),
        "interpretation_note": "compare grade_label's adjusted coefficient/p here against P1's unadjusted Mann-Whitney result to assess whether gland-area confounding materially changes the picture",
    }

    # ---- ROBUSTNESS (R1): patient-level ----
    patient_df = df.groupby("patient ID").agg(
        S_DL_mean=("S_DL_mean", "mean"),
        grade_label=("grade_label", "max"),  # a patient's images should share grade in practice; max is a safe deterministic reducer, checked below
        n_images=("S_DL_mean", "count"),
    )
    grade_consistency = df.groupby("patient ID")["grade_label"].nunique()
    patients_with_mixed_grade = int((grade_consistency > 1).sum())
    r1 = {"n_patients": int(len(patient_df)), "patients_with_mixed_grade_label_across_images": patients_with_mixed_grade}
    if len(patient_df) >= 5 and patients_with_mixed_grade == 0:
        r1.update(run_binary_test("R1", patient_df["S_DL_mean"].values, patient_df["grade_label"].values, len(patient_df)))
    else:
        r1["note"] = "patient-level test not computed as a formal statistic: n too small and/or mixed-grade patients present; reported descriptively only"
        r1["patient_level_S_DL_by_grade"] = {
            "malignant_patients_mean_S_DL": float(patient_df.loc[patient_df.grade_label == 1, "S_DL_mean"].mean()) if (patient_df.grade_label == 1).any() else None,
            "benign_patients_mean_S_DL": float(patient_df.loc[patient_df.grade_label == 0, "S_DL_mean"].mean()) if (patient_df.grade_label == 0).any() else None,
        }
    results["robustness_R1_patient_level"] = r1
    patient_df.to_csv(os.path.join(OUT_DIR, "tables", "stage15_patient_level_table.csv"))

    with open(os.path.join(OUT_DIR, "stage15_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("STAGE 15 RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
