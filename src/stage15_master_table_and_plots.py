"""
STAGE 15: assembles the required master results table (CSV) and the
diagnostic figures, from results_v2/validation/stage15/stage15_results.json
(already produced by stage15_pathology_analysis.py). No new statistics are
computed here -- this only tabulates and visualizes the frozen results.
"""
import os
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage15")

with open(os.path.join(OUT_DIR, "stage15_results.json")) as f:
    R = json.load(f)

N_PATCHES = {"testA": 1470, "testB": 613}
N_PATCHES_POOLED = 2083
NA = "NA"

rows = []

# ---- P1-P4 pooled (primary) ----
for aid, endpoint, rep in [("P1", "grade_label (binary)", "S_DL"), ("P2", "ordinal_grade (5-level)", "S_DL"),
                            ("P3", "grade_label (binary)", "phi_DL_dispersion"), ("P4", "ordinal_grade (5-level)", "phi_DL_dispersion")]:
    r = R["primary_pooled"][aid]
    if "p_value" not in r:
        continue
    if "cohens_d" in r:
        eff, ci = r["cohens_d"], r["mean_diff_95CI_bootstrap"]
        atype = "Mann-Whitney U + Cohen's d"
    else:
        eff, ci = r["spearman_rho"], r["rho_95CI_bootstrap"]
        atype = "Spearman rank correlation"
    rows.append({
        "analysis_id": aid, "endpoint": endpoint, "endpoint_category": "pathology", "representation": rep,
        "unit_of_analysis": "image", "n_images": R["n_images_total"], "n_glands": NA, "n_patches": N_PATCHES_POOLED,
        "effect_size": eff, "confidence_interval": json.dumps(ci), "p_value": r["p_value"],
        "adjusted_p_value": R["holm_bonferroni_adjusted_p"].get(aid, NA), "confounders": NA,
        "analysis_type": atype, "primary_or_exploratory": "primary",
        "interpretation": "not significant after Holm-Bonferroni correction (pooled testA+testB)" if R["holm_bonferroni_adjusted_p"].get(aid, 1) >= 0.05 else "significant after correction",
    })

# ---- P1-P4 per split (primary, raw p, not corrected family) ----
for split in ["testA", "testB"]:
    for aid, endpoint, rep in [("P1", "grade_label (binary)", "S_DL"), ("P2", "ordinal_grade (5-level)", "S_DL"),
                                ("P3", "grade_label (binary)", "phi_DL_dispersion"), ("P4", "ordinal_grade (5-level)", "phi_DL_dispersion")]:
        r = R["primary_by_split"][split][aid]
        if "p_value" not in r:
            continue
        if "cohens_d" in r:
            eff, ci = r["cohens_d"], r["mean_diff_95CI_bootstrap"]
            atype = "Mann-Whitney U + Cohen's d"
        else:
            eff, ci = r["spearman_rho"], r["rho_95CI_bootstrap"]
            atype = "Spearman rank correlation"
        rows.append({
            "analysis_id": f"{aid}_{split}", "endpoint": endpoint, "endpoint_category": "pathology", "representation": rep,
            "unit_of_analysis": "image", "n_images": R[f"n_{split}"], "n_glands": NA, "n_patches": N_PATCHES[split],
            "effect_size": eff, "confidence_interval": json.dumps(ci), "p_value": r["p_value"],
            "adjusted_p_value": "NA (not part of pre-specified corrected family; per-split reporting is descriptive)", "confounders": NA,
            "analysis_type": atype, "primary_or_exploratory": "primary (split-level, uncorrected)",
            "interpretation": "raw p-value, reported for transparency per instruction section 13; must not be used to override the pooled result",
        })

# ---- M1/M2 morphology ----
for aid in ["M1", "M2"]:
    r = R["morphology"][aid]
    rows.append({
        "analysis_id": aid, "endpoint": r["endpoint"], "endpoint_category": "morphology", "representation": "S_DL",
        "unit_of_analysis": "image", "n_images": r["n"], "n_glands": NA, "n_patches": N_PATCHES_POOLED,
        "effect_size": r["spearman_rho"], "confidence_interval": json.dumps(r["rho_95CI_bootstrap"]), "p_value": r["p_value"],
        "adjusted_p_value": "NA (not part of the pathology corrected family)", "confounders": NA,
        "analysis_type": "Spearman rank correlation", "primary_or_exploratory": "primary (morphology)",
        "interpretation": "no material association detected",
    })

# ---- C1 confound adjustment ----
c1 = R["confound_adjustment_C1"]
rows.append({
    "analysis_id": "C1", "endpoint": "grade_label (adjusted for mean_gland_area)", "endpoint_category": "confound_adjustment",
    "representation": "S_DL", "unit_of_analysis": "image", "n_images": c1["n"], "n_glands": NA, "n_patches": N_PATCHES_POOLED,
    "effect_size": c1["grade_label_coef"], "confidence_interval": json.dumps(c1["grade_label_95CI"]), "p_value": c1["grade_label_p"],
    "adjusted_p_value": "NA", "confounders": "mean_gland_area", "analysis_type": "OLS regression",
    "primary_or_exploratory": "primary (confound adjustment)",
    "interpretation": "grade_label coefficient remains near-zero and non-significant after adjusting for gland area, consistent with P1's unadjusted null result",
})

# ---- R1 patient-level robustness ----
r1 = R["robustness_R1_patient_level"]
rows.append({
    "analysis_id": "R1", "endpoint": "grade_label (patient-level aggregation)", "endpoint_category": "pathology",
    "representation": "S_DL", "unit_of_analysis": "patient", "n_images": R["n_images_total"], "n_glands": NA, "n_patches": N_PATCHES_POOLED,
    "effect_size": NA, "confidence_interval": NA, "p_value": NA, "adjusted_p_value": NA, "confounders": NA,
    "analysis_type": "descriptive (formal test not computed: n=12 patients, 3 with mixed grade labels across their images)",
    "primary_or_exploratory": "robustness (pre-specified)",
    "interpretation": f"malignant-patient mean S_DL={r1['patient_level_S_DL_by_grade']['malignant_patients_mean_S_DL']:.4f}, "
                       f"benign-patient mean S_DL={r1['patient_level_S_DL_by_grade']['benign_patients_mean_S_DL']:.4f} -- "
                       f"direction is opposite to the pooled image-level P1 result, underscoring that neither is a stable, well-powered effect",
})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "stage15_master_results.csv"), index=False)
print(df.to_string())


# ---------------------------------------------------------------- figures

def fig_pathology_scatter():
    tbl = pd.read_csv(os.path.join(OUT_DIR, "tables", "stage15_image_level_table.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = tbl["grade_label"].map({0: "tab:blue", 1: "tab:red"})
    axes[0].scatter(tbl["ordinal_grade"], tbl["S_DL_mean"], c=colors, alpha=0.7)
    axes[0].set_xlabel("ordinal grade (0=healthy .. 4=poorly differentiated)")
    axes[0].set_ylabel("S_DL (image-level mean)")
    axes[0].set_title(f"P2: S_DL vs ordinal grade (pooled, n={len(tbl)} images, unit=image)\n"
                       f"Spearman rho={R['primary_pooled']['P2']['spearman_rho']:.3f}, p={R['primary_pooled']['P2']['p_value']:.3f} (Holm-adj={R['holm_bonferroni_adjusted_p']['P2']:.3f})")

    axes[1].scatter(tbl["ordinal_grade"], tbl["phi_DL_dispersion"], c=colors, alpha=0.7)
    axes[1].set_xlabel("ordinal grade (0=healthy .. 4=poorly differentiated)")
    axes[1].set_ylabel("phi_DL circular dispersion (image-level)")
    axes[1].set_title(f"P4: phi_DL dispersion vs ordinal grade (pooled, n={len(tbl)} images, unit=image)\n"
                       f"Spearman rho={R['primary_pooled']['P4']['spearman_rho']:.3f}, p={R['primary_pooled']['P4']['p_value']:.3f} (Holm-adj={R['holm_bonferroni_adjusted_p']['P4']:.3f})")

    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', label='benign', markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', label='malignant', markersize=8)]
    for ax in axes:
        ax.legend(handles=legend_elems)
    fig.suptitle("Stage 15: representation vs pathology endpoint (each point = one image; pooled testA+testB, primary analysis)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figures", "stage15_pathology_scatter.png"), dpi=130)
    plt.close(fig)


def fig_morphology_scatter():
    tbl = pd.read_csv(os.path.join(OUT_DIR, "tables", "stage15_image_level_table.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(tbl["mean_gland_area"], tbl["S_DL_mean"], alpha=0.7)
    axes[0].set_xlabel("mean gland area (pixels, image-level)"); axes[0].set_ylabel("S_DL (image-level mean)")
    axes[0].set_title(f"M1: rho={R['morphology']['M1']['spearman_rho']:.3f}, p={R['morphology']['M1']['p_value']:.3f}")
    axes[1].scatter(tbl["n_gland_instances"], tbl["S_DL_mean"], alpha=0.7)
    axes[1].set_xlabel("n_gland_instances (image-level)"); axes[1].set_ylabel("S_DL (image-level mean)")
    axes[1].set_title(f"M2: rho={R['morphology']['M2']['spearman_rho']:.3f}, p={R['morphology']['M2']['p_value']:.3f}")
    fig.suptitle("Stage 15: representation vs morphology endpoint (each point = one image; pooled testA+testB)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figures", "stage15_morphology_scatter.png"), dpi=130)
    plt.close(fig)


def fig_confound_adjusted():
    tbl = pd.read_csv(os.path.join(OUT_DIR, "tables", "stage15_image_level_table.csv"))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = tbl["grade_label"].map({0: "tab:blue", 1: "tab:red"})
    ax.scatter(tbl["mean_gland_area"], tbl["S_DL_mean"], c=colors, alpha=0.7)
    c1 = R["confound_adjustment_C1"]
    ax.set_xlabel("mean gland area (image-level, the adjustment covariate)")
    ax.set_ylabel("S_DL (image-level mean)")
    ax.set_title(f"C1: S_DL ~ grade_label + area (OLS)\ngrade_label coef={c1['grade_label_coef']:.5f} "
                 f"(95% CI [{c1['grade_label_95CI'][0]:.5f}, {c1['grade_label_95CI'][1]:.5f}]), p={c1['grade_label_p']:.3f}, R^2={c1['r_squared']:.3f}")
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', label='benign', markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', label='malignant', markersize=8)]
    ax.legend(handles=legend_elems)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figures", "stage15_confound_adjusted.png"), dpi=130)
    plt.close(fig)


fig_pathology_scatter()
fig_morphology_scatter()
fig_confound_adjusted()
print("[stage15] figures written")
