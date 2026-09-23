"""
STAGE 20: publication figure/table package. Documentation/visualization
ONLY -- every number is read from already-frozen Stage 0-19 source files
(no new experiment, no retraining, no recomputation of any statistic).
"""
import os
import json
import csv

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(PROJECT_ROOT, "results_v2", "figures", "stage20")
TAB_DIR = os.path.join(PROJECT_ROOT, "results_v2", "tables", "stage20")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

SYN_DIR = os.path.join(PROJECT_ROOT, "results_v2", "statistical_synthesis")


def load(path):
    with open(os.path.join(PROJECT_ROOT, path)) as f:
        return json.load(f)


s11 = load("results_v2/validation/stage11/stage11_results.json")
s12 = load("results_v2/validation/stage12/stage12_results.json")
s14 = load("results_v2/validation/stage14/stage14_all_evaluations.json")
s14_a6 = load("results_v2/validation/stage14/A6_spatial_reduction/a6_results.json")
s15 = load("results_v2/validation/stage15/stage15_results.json")
s16 = load("results_v2/validation/stage16/stage16_results.json")
s17 = load("results_v2/validation/stage17_crag/stage17_results.json")
s18 = load("results_v2/validation/stage18_panda/compatibility_results.json")

provenance_records = []


def record_provenance(fig_or_table, source_file, field, dataset, unit, transform, checksum=None):
    provenance_records.append({"artifact": fig_or_table, "source_file": source_file, "source_field": field,
                                "dataset": dataset, "unit": unit, "transformation": transform,
                                "script": "code_v2/stage20_figures_and_tables.py", "source_checksum": checksum or "see STAGE19_SOURCE_INVENTORY.md"})


NEUTRAL_GRAY = "#6b6b6b"
FAIL_RED = "#b23a3a"
PASS_GREEN = "#3a7d44"

# ============================================================ FIGURE 1: conceptual framework
def figure1():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")

    stages = ["H&E\nhistology\nimage", "Gland instance\nmask\n(segmentation)",
              "Principal\norientation theta,\nanisotropy S\n(PCA)", "Rank-2 tensor\nQ_anat\n(anatomical target)",
              "D8-equivariant\nCNN\n(Model 3)", "Predicted\nQ_DL", "Validation\n(Stages 11-18)"]
    xs = np.linspace(0.9, 12.1, len(stages))
    y = 4.3
    for x, label in zip(xs, stages):
        box = FancyBboxPatch((x - 0.75, y - 0.65), 1.5, 1.3, boxstyle="round,pad=0.05",
                              facecolor="#eef2f7", edgecolor="#33475b", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.3)
    for i in range(len(xs) - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + 0.75, y), (xs[i + 1] - 0.75, y), arrowstyle="-|>",
                                      mutation_scale=14, color="#33475b", linewidth=1.2))

    eq_box = FancyBboxPatch((0.5, 0.3), 12, 2.6, boxstyle="round,pad=0.08", facecolor="#fbf8ef", edgecolor="#9a8452")
    ax.add_patch(eq_box)
    ax.text(6.5, 2.45, "Nematic (head-tail symmetric, director) representation", ha="center", fontsize=10, weight="bold")
    ax.text(6.5, 1.75, "Q = [[q1, q2], [q2, -q1]],    (q1, q2) = S · (cos 2θ, sin 2θ)",
            ha="center", fontsize=13, family="monospace")
    ax.text(6.5, 0.95,
            "Rotation transformation law:   Q'(Rα x) = R(α) Q(x) R(α)ᵀ   <=>   (q1,q2)' = R(2α) (q1,q2)",
            ha="center", fontsize=11, family="monospace")
    ax.text(6.5, 0.5, r"Double-angle mapping enforces $\theta \equiv \theta + \pi$ (head-tail symmetry) automatically", ha="center", fontsize=9, style="italic", color="#555")

    ax.text(6.5, 5.6, "Figure 1. Conceptual and mathematical framework", ha="center", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure1_conceptual_framework.png"), dpi=140)
    plt.close(fig)
    record_provenance("Figure 1", "N/A (schematic; equations from docs_v2/phase2_target_math.md and nematic_math.py, unmodified)",
                       "Q construction / rotation law definitions", "N/A", "N/A", "schematic diagram, no numerical data plotted")
    print("[fig1] done")


# ============================================================ FIGURE 2: target construction example
def figure2():
    image_id = "testA_1"
    manifest = pd.read_csv(os.path.join(PROJECT_ROOT, "data_manifests", "glas_manifest_stage3_finalized.csv"))
    row = manifest[manifest.image_id == image_id].iloc[0]
    img = np.array(Image.open(os.path.join(PROJECT_ROOT, row["image_path"])).convert("RGB"))
    mask = np.array(Image.open(os.path.join(PROJECT_ROOT, row["mask_path"])))
    gland_df = pd.read_csv(os.path.join(PROJECT_ROOT, "results_v2", "anatomy_targets", "stage4_gland_manifest.csv"))
    glands = gland_df[gland_df.image_id == image_id]
    field = np.load(os.path.join(PROJECT_ROOT, "results_v2", "anatomy_targets", "stage4_fields", f"{image_id}.npz"))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img); axes[0].set_title(f"(a) H&E image\n({image_id})"); axes[0].axis("off")

    axes[1].imshow(mask, cmap="nipy_spectral"); axes[1].set_title("(b) gland instance mask\n(Stage 4, frozen)"); axes[1].axis("off")

    axes[2].imshow(img)
    from skimage.measure import regionprops
    for _, g in glands.iterrows():
        if g["validity_category"] != "valid":
            continue
        ys, xs = np.nonzero(mask == g["gland_id"])
        cy, cx = ys.mean(), xs.mean()
        theta = g["phi_cov"]; S = g["S_cov"]
        length = 40 + 60 * S
        dx, dy = length * np.cos(theta), length * np.sin(theta)
        axes[2].plot([cx - dx / 2, cx + dx / 2], [cy - dy / 2, cy + dy / 2], color="lime", linewidth=2)
    axes[2].set_title("(c) principal orientation theta\n(valid glands only, length ~ S)"); axes[2].axis("off")

    S_field = field["S"]
    im = axes[3].imshow(S_field, cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("(d) dense order-magnitude field S\n(from Q_anat, valid pixels only)"); axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046)

    fig.suptitle("Figure 2. Anatomical target construction from GlaS gland masks (frozen Stage 4 pipeline, unmodified)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure2_target_construction.png"), dpi=140)
    plt.close(fig)
    record_provenance("Figure 2", "data_manifests/glas_manifest_stage3_finalized.csv; results_v2/anatomy_targets/stage4_gland_manifest.csv; results_v2/anatomy_targets/stage4_fields/testA_1.npz",
                       "image_path, mask_path, phi_cov, S_cov, validity_category, S field", "GlaS (testA_1, representative example)",
                       "gland (panel c), pixel (panel d)", "direct visualization of frozen Stage 4 outputs; no recomputation")
    print("[fig2] done")


# ============================================================ FIGURE 3: primary GlaS validation
def figure3():
    m = s11["pooled_testA_testB_80_images"]["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    a = m["A_angular_correspondence"]
    axes[0].bar(["observed\nGlass's delta"], [a["glass_delta"]], color=NEUTRAL_GRAY, width=0.5)
    axes[0].axhline(0.5, color=FAIL_RED, linestyle="--", linewidth=1.5, label="frozen threshold (>=0.5)")
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_ylim(-0.1, 0.6)
    axes[0].set_title(f"Criterion A: angular correspondence\nGlass's delta={a['glass_delta']:.4f} (perm p={a['permutation_p_value']})\nn={m['n_patches']} patches, pooled testA+testB")
    axes[0].legend(fontsize=8)

    b = m["B_order_magnitude_correlation"]
    axes[1].bar(["observed r"], [b["pearson_r"]], color=NEUTRAL_GRAY, width=0.5,
                yerr=[[b["pearson_r"] - b["bootstrap_95CI"][0]], [b["bootstrap_95CI"][1] - b["pearson_r"]]], capsize=6)
    axes[1].axhline(0.30, color=FAIL_RED, linestyle="--", linewidth=1.5, label="frozen threshold (r>=0.30)")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylim(-0.15, 0.4)
    axes[1].set_title(f"Criterion B: order-magnitude correlation\nr={b['pearson_r']:.4f} (95% CI shown), p={b['pearson_p']:.3f}")
    axes[1].legend(fontsize=8)

    c = m["C_tensor_similarity"]
    axes[2].bar(["observed\nGlass's delta"], [c["glass_delta"]], color=NEUTRAL_GRAY, width=0.5)
    axes[2].axhline(0.5, color=FAIL_RED, linestyle="--", linewidth=1.5, label="frozen threshold (>=0.5)")
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_ylim(-0.1, 0.6)
    axes[2].set_title(f"Criterion C: tensor similarity (D_Q)\nGlass's delta={c['glass_delta']:.4f} (perm p={c['permutation_p_value']:.3g})\nmean D_Q={c['mean_D_Q']:.4f}")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_ylabel("effect size (Glass's delta / r)")
    fig.suptitle("Figure 3. Primary GlaS anatomical validation (Stage 11) -- preregistered criteria, pooled testA+testB\n"
                 "Small p-values at some criteria reflect large sample size (n=2083), NOT a meaningful anatomical effect -- see effect-size axis", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure3_primary_glas_validation.png"), dpi=140)
    plt.close(fig)
    record_provenance("Figure 3", "results_v2/validation/stage11/stage11_results.json",
                       "pooled_testA_testB_80_images.metrics.{A_angular_correspondence,B_order_magnitude_correlation,C_tensor_similarity}",
                       "GlaS", "patch (pooled testA+testB, n=2083)", "direct plot of frozen effect sizes/CIs/thresholds, no recomputation",
                       "74918811257c8351ddbe3d28978ba203")
    print("[fig3] done")


# ============================================================ FIGURE 4: rotation consistency
def figure4():
    angles = [r["angle_deg"] for r in s12["per_angle"] if r["angle_deg"] != 0]
    errs = [r["model_error_mean"] for r in s12["per_angle"] if r["angle_deg"] != 0]
    thresh = [r["frozen_local_threshold"] for r in s12["per_angle"] if r["angle_deg"] != 0]
    passed = [r["PASSES_criterion_D_this_angle"] for r in s12["per_angle"] if r["angle_deg"] != 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(angles))
    colors = [PASS_GREEN if p else FAIL_RED for p in passed]
    ax.bar(x - 0.18, errs, width=0.36, color=colors, label="observed model error")
    ax.bar(x + 0.18, thresh, width=0.36, color="none", edgecolor="black", hatch="//", label="frozen pass threshold")
    ax.set_xticks(x); ax.set_xticklabels([str(a) for a in angles])
    ax.set_xlabel("rotation angle (degrees)"); ax.set_ylabel("normalized equivariance error")
    n_pass = sum(passed)
    ax.set_title(f"Figure 4. Rotation consistency (Stage 12), all 9 non-zero angles, pooled testA+testB (n=2083)\n"
                 f"{n_pass}/9 passed (green=pass, red=fail vs. frozen per-angle threshold; >=7/9 required)")
    green_patch = mpatches.Patch(color=PASS_GREEN, label="observed error (angle passes)")
    red_patch = mpatches.Patch(color=FAIL_RED, label="observed error (angle fails)")
    hatch_patch = mpatches.Patch(facecolor="none", edgecolor="black", hatch="//", label="frozen threshold")
    ax.legend(handles=[green_patch, red_patch, hatch_patch], fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure4_rotation_consistency.png"), dpi=140)
    plt.close(fig)
    record_provenance("Figure 4", "results_v2/validation/stage12/stage12_results.json",
                       "per_angle[*].{model_error_mean,frozen_local_threshold,PASSES_criterion_D_this_angle}",
                       "GlaS", "patch (pooled testA+testB, n=2083)", "direct plot, all 9 non-zero angles shown, none omitted",
                       "f43a186332149abbda83420623da1a9c")
    print("[fig4] done")


# ============================================================ FIGURE 5: ablation + multiscale
def figure5():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    labels = ["A0\nreference", "A1\nno Q sup.", "A3\nQ-dominant", "A4\nQ-only", "A5\nnon-equiv."]
    exps = ["A0_reference", "A1_no_q", "A3_q_dominant", "A4_q_only", "A5_plain_cnn_control"]
    glass_a = [s14[e]["pooled_testA_testB"]["geometric_metrics"]["A_angular_correspondence"]["glass_delta"] for e in exps]
    r_vals = [s14[e]["pooled_testA_testB"]["geometric_metrics"]["B_order_magnitude_correlation"]["pearson_r"] for e in exps]
    x = np.arange(len(labels))
    axes[0].bar(x, glass_a, color=NEUTRAL_GRAY)
    axes[0].axhline(0.5, color=FAIL_RED, linestyle="--", linewidth=1.2, label="Criterion A threshold (context)")
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_title("Ablations: Glass's delta_A\n(pooled testA+testB, diagnostic only)"); axes[0].legend(fontsize=7)

    axes[1].bar(x, r_vals, color=NEUTRAL_GRAY)
    axes[1].axhline(0.30, color=FAIL_RED, linestyle="--", linewidth=1.2, label="Criterion B threshold (context)")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_title("Ablations: order-magnitude r\n(pooled testA+testB, diagnostic only)"); axes[1].legend(fontsize=7)

    scale_labels = ["center", "3x3", "5x5", "9x9", "17x17", "full-field"]
    scale_keys = ["1", "3", "5", "9", "17", "full"]
    ms_glass_a = [s16["anatomy_by_scale"][k]["pooled"]["metrics"]["A_angular_correspondence"]["glass_delta"] for k in scale_keys]
    ms_rot = [s16["rotation_by_scale"][k]["n_angles_passed_vs_original_threshold"] for k in scale_keys]
    ax2 = axes[2]
    ax2.plot(scale_labels, ms_glass_a, marker="o", color=NEUTRAL_GRAY, label="Glass's delta_A (left axis)")
    ax2.axhline(0.5, color=FAIL_RED, linestyle="--", linewidth=1, label="Criterion A threshold (context)")
    ax2.set_ylabel("Glass's delta_A"); ax2.tick_params(axis="x", labelrotation=20)
    ax3 = ax2.twinx()
    ax3.plot(scale_labels, ms_rot, marker="s", color="#2f6690", label="rotation angles passed (right axis)")
    ax3.axhline(7, color="#2f6690", linestyle=":", linewidth=1)
    ax3.set_ylabel("rotation angles passed (of 9)"); ax3.set_ylim(0, 9.5)
    lines1, labels1 = ax2.get_legend_handles_labels(); lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax2.set_title("Multi-scale (Stage 16): 6 frozen scales\n(no scale selected/preferred)")

    fig.suptitle("Figure 5. Ablation (Stage 14) and multi-scale (Stage 16) evidence -- neutral comparison, no ranking, no winner", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure5_ablation_multiscale.png"), dpi=140)
    plt.close(fig)
    record_provenance("Figure 5", "results_v2/validation/stage14/stage14_all_evaluations.json; results_v2/validation/stage16/stage16_results.json",
                       "<exp>.pooled_testA_testB.geometric_metrics.{A_angular_correspondence,B_order_magnitude_correlation}; anatomy_by_scale.<scale>.pooled.metrics; rotation_by_scale.<scale>.n_angles_passed_vs_original_threshold",
                       "GlaS", "patch (pooled testA+testB)", "direct plot, all 5 ablation conditions and all 6 scales shown, none omitted or reordered by result",
                       "29cbdb82c83b71257955c2594dac8dc2 / 3c87d6ced4304b3d4d0a350441018bfa")
    print("[fig5] done")


# ============================================================ FIGURE 6: pathology + CRAG (never pooled)
def figure6():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # NOTE: P1/P3 report Cohen's d as the effect size, but Stage 15's only bootstrap CI for
    # those two tests is for the RAW mean difference (different units/scale) -- plotting a
    # CI in d-units for those would misrepresent an uncomputed quantity. P2/P4 report
    # Spearman rho with a CI for rho itself (matched units), so those get error bars; P1/P3
    # are shown as point estimates only, explicitly labeled.
    p_labels = ["P1\n(S_DL vs\nbinary grade)\nCohen's d", "P2\n(S_DL vs\nordinal grade)\nSpearman rho",
                "P3\n(phi-disp vs\nbinary grade)\nCohen's d", "P4\n(phi-disp vs\nordinal grade)\nSpearman rho"]
    p_ids = ["P1", "P2", "P3", "P4"]
    p_eff, p_err_lo, p_err_hi = [], [], []
    for pid in p_ids:
        r = s15["primary_pooled"][pid]
        if "cohens_d" in r:
            eff = r["cohens_d"]; p_eff.append(eff); p_err_lo.append(0.0); p_err_hi.append(0.0)
        else:
            eff = r["spearman_rho"]; ci = r["rho_95CI_bootstrap"]
            p_eff.append(eff); p_err_lo.append(eff - ci[0]); p_err_hi.append(ci[1] - eff)
    x = np.arange(4)
    axes[0].bar(x, p_eff, yerr=[p_err_lo, p_err_hi], capsize=5, color=NEUTRAL_GRAY)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(p_labels, fontsize=7.2)
    axes[0].set_title("(a) Pathology analysis on GlaS (Stage 15)\nimage-level, n=80, Holm-corrected (all adj. p=1.0)\nNo association demonstrated", fontsize=9.5)
    axes[0].set_ylabel("effect size (Cohen's d or Spearman rho, as labeled)")

    crag_pops = ["train\n(n=171)", "test\n(n=40)", "pooled\n(n=211)"]
    crag_glass_a = [s17["image_level_primary"][p]["metrics"]["A_angular_correspondence"]["glass_delta"] for p in ["train", "test", "pooled"]]
    axes[1].bar(crag_pops, crag_glass_a, color=NEUTRAL_GRAY)
    axes[1].axhline(0.5, color=FAIL_RED, linestyle="--", linewidth=1.2, label="GlaS-derived threshold (context only)")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_title("(b) CRAG external cross-dataset validation (Stage 17)\nimage-level Glass's delta_A, by population\nNot pooled with GlaS; partial institutional overlap", fontsize=9.5)
    axes[1].set_ylabel("Glass's delta_A"); axes[1].legend(fontsize=8)

    fig.suptitle("Figure 6. Pathology (GlaS) and CRAG (external) evidence -- shown separately, never pooled", fontsize=11, y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "figure6_pathology_crag.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    record_provenance("Figure 6", "results_v2/validation/stage15/stage15_results.json; results_v2/validation/stage17_crag/stage17_results.json",
                       "primary_pooled.{P1,P2,P3,P4}; image_level_primary.{train,test,pooled}.metrics.A_angular_correspondence.glass_delta",
                       "GlaS (panel a) / CRAG (panel b), reported separately", "image-level (both panels)",
                       "direct plot; panels never combined into one statistic", "1c3d2bbb53557649016bc477abfdeff1 / 44913f300fdc941a16437a3a5404103f")
    print("[fig6] done")


# ============================================================ FIGURE 7: PANDA compatibility gate (supplementary)
def figure7():
    rows = []
    for case in s18:
        benign_med = case["component_stats"]["benign_epithelium"].get("area_px_median")
        if not benign_med:
            continue
        for cls in ["gleason_3", "gleason_4", "gleason_5"]:
            largest = case["component_stats"][cls].get("area_px_max")
            if largest:
                rows.append({"case": case["image_id"][:8], "isup": case["isup_grade"], "class": cls, "ratio": largest / benign_med})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    class_colors = {"gleason_3": "#d4a017", "gleason_4": "#d97b29", "gleason_5": "#b23a3a"}
    for i, (_, r) in enumerate(df.iterrows()):
        ax.bar(i, r["ratio"], color=class_colors[r["class"]])
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{r['case']}\n(ISUP {r['isup']})\n{r['class'].replace('gleason_','G')}" for _, r in df.iterrows()], fontsize=7, rotation=0)
    ax.set_yscale("log")
    ax.set_ylabel("largest connected component size /\nsame-case benign-gland median size (log scale)")
    ax.axhline(1, color="black", linewidth=0.8, label="same size as typical benign gland")
    handles = [mpatches.Patch(color=c, label=l.replace("gleason_", "Gleason ")) for l, c in class_colors.items()]
    ax.legend(handles=handles + [plt.Line2D([0], [0], color="black", lw=0.8, label="benign-gland reference (=1)")], fontsize=8)
    ax.set_title("Figure 7 (Supplementary). PANDA Radboud compatibility gate (Stage 18)\n"
                 "Largest same-class connected component vs. same-case benign-gland size, 10-case pilot\n"
                 "Not a model result -- a data/annotation-semantics compatibility finding")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure7_supplementary_panda_compatibility.png"), dpi=140)
    plt.close(fig)
    record_provenance("Figure 7 (Supplementary)", "results_v2/validation/stage18_panda/compatibility_results.json",
                       "component_stats.<class>.area_px_max; component_stats.benign_epithelium.area_px_median",
                       "PANDA Radboud (10-case pilot)", "connected component (within case)",
                       "ratio computed as largest_component_px / same-case benign_median_px (already computed and reported in Stage 18; replotted here, not recomputed)",
                       "bd1a6895a9ad4df3822c86678454418d")
    print("[fig7] done")


figure1()
figure2()
figure3()
figure4()
figure5()
figure6()
figure7()
print("All 7 figures complete.")

# ============================================================ TABLE 1: datasets and experimental design
def table1():
    rows = [
        {"dataset": "GlaS", "role": "primary (preregistered)", "n_images": 165, "canonical_split": "train=85 / testA=60 / testB=20",
         "organ": "colorectal", "annotation": "individual gland-instance masks", "stages_used": "0-16, 19-20"},
        {"dataset": "CRAG", "role": "external cross-dataset validation", "n_images": 213, "canonical_split": "train=173 / test=40 (CRAG's own split; all held out from Model 3 training)",
         "organ": "colorectal", "annotation": "individual gland-instance masks (documented large-fused-instance minority, Stage 17)", "stages_used": "17, 19-20"},
        {"dataset": "PANDA (Radboud)", "role": "attempted cross-organ external validation; compatibility gate only", "n_images": "10 (pilot; full Radboud subset=5160 in train.csv)",
         "canonical_split": "N/A (never used for training or inference)", "organ": "prostate",
         "annotation": "semantic Gleason-pattern tissue-class masks (0-5) -- NOT individual gland-instance masks", "stages_used": "18-20"},
    ]
    with open(os.path.join(TAB_DIR, "table1_datasets_and_design.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    record_provenance("Table 1", "reports/STAGE11_AUDIT_REPORT.md, STAGE17_AUDIT_REPORT.md, STAGE18_AUDIT_REPORT.md",
                       "dataset sizes/splits as stated in each stage's audit report", "GlaS/CRAG/PANDA", "image", "descriptive summary only, no computation")
    print("[table1] done")


# ============================================================ TABLE 2: primary preregistered criteria
def table2():
    m = s11["pooled_testA_testB_80_images"]["metrics"]
    rows = [
        {"criterion": "A - angular correspondence", "statistic": f"Glass's delta = {m['A_angular_correspondence']['glass_delta']:.4f}",
         "frozen_threshold": ">= 0.5", "p_value": f"perm p={m['A_angular_correspondence']['permutation_p_value']}",
         "n": m["n_patches"], "unit": "patch (pooled testA+testB)", "outcome": "FAIL", "note": "perm p<0.01 met, but effect size (Glass's delta) far below threshold"},
        {"criterion": "B - order-magnitude correlation", "statistic": f"Pearson r = {m['B_order_magnitude_correlation']['pearson_r']:.4f}",
         "frozen_threshold": "r>=0.30 AND p<0.01 AND CI lower>0", "p_value": f"{m['B_order_magnitude_correlation']['pearson_p']:.4f}",
         "n": m["n_patches"], "unit": "patch (pooled testA+testB)", "outcome": "FAIL", "note": "all three sub-conditions failed"},
        {"criterion": "C - tensor similarity", "statistic": f"mean D_Q = {m['C_tensor_similarity']['mean_D_Q']:.4f}, Glass's delta = {m['C_tensor_similarity']['glass_delta']:.4f}",
         "frozen_threshold": "perm p<0.01 AND Glass's delta>=0.5", "p_value": f"{m['C_tensor_similarity']['permutation_p_value']}",
         "n": m["n_patches"], "unit": "patch (pooled testA+testB)", "outcome": "FAIL", "note": "perm p<0.01 met, but effect size far below threshold"},
    ]
    with open(os.path.join(TAB_DIR, "table2_primary_criteria.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    record_provenance("Table 2", "results_v2/validation/stage11/stage11_results.json", "pooled_testA_testB_80_images.metrics",
                       "GlaS", "patch (pooled testA+testB, n=2083)", "direct transcription, no recomputation", "74918811257c8351ddbe3d28978ba203")
    print("[table2] done")


# ============================================================ TABLE 3: rotation consistency per angle
def table3():
    rows = []
    for r in s12["per_angle"]:
        rows.append({"angle_deg": r["angle_deg"], "is_identity_reference": r["is_identity_sanity_check_only"],
                     "model_error_mean": r["model_error_mean"], "frozen_threshold": r.get("frozen_local_threshold", "N/A"),
                     "passes": r.get("PASSES_criterion_D_this_angle", "N/A (identity, not scored)")})
    with open(os.path.join(TAB_DIR, "table3_rotation_consistency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    record_provenance("Table 3", "results_v2/validation/stage12/stage12_results.json", "per_angle[*]",
                       "GlaS", "patch (pooled testA+testB, n=2083)", "direct transcription, all 10 angles (incl. identity) shown", "f43a186332149abbda83420623da1a9c")
    print("[table3] done")


# ============================================================ TABLE 4: ablation + multiscale
def table4():
    rows = []
    labels = {"A0_reference": "A0 reference", "A1_no_q": "A1 no Q supervision", "A3_q_dominant": "A3 Q-dominant (lambda_Q=10)",
              "A4_q_only": "A4 Q-only (lambda_class=0)", "A5_plain_cnn_control": "A5 non-equivariant control (parameter-matched)"}
    for exp, label in labels.items():
        p = s14[exp]["pooled_testA_testB"]["geometric_metrics"]
        rows.append({"analysis": label, "type": "ablation", "n": s14[exp]["pooled_testA_testB"]["n_patches"], "unit": "patch (pooled testA+testB)",
                     "glass_delta_A": round(p["A_angular_correspondence"]["glass_delta"], 4), "pearson_r_B": round(p["B_order_magnitude_correlation"]["pearson_r"], 4),
                     "mean_D_Q_C": round(p["C_tensor_similarity"]["mean_D_Q"], 4), "rotation_angles_passed": "NA", "outcome_vs_frozen_criteria": "context only, diagnostic (no new criterion)"})
    rows.append({"analysis": "A6 rotation diagnostic, R1 (full-field)", "type": "ablation", "n": 2083, "unit": "patch (pooled)",
                 "glass_delta_A": "NA", "pearson_r_B": "NA", "mean_D_Q_C": "NA",
                 "rotation_angles_passed": f"{s14_a6['reductions']['R1_full_spatial_mean']['n_angles_passed_diagnostic_only']}/9", "outcome_vs_frozen_criteria": "context only vs original Stage 12 threshold"})
    rows.append({"analysis": "A6 rotation diagnostic, R2 (fixed mask)", "type": "ablation", "n": 2083, "unit": "patch (pooled)",
                 "glass_delta_A": "NA", "pearson_r_B": "NA", "mean_D_Q_C": "NA",
                 "rotation_angles_passed": f"{s14_a6['reductions']['R2_fixed_original_valid_mask_mean']['n_angles_passed_diagnostic_only']}/9", "outcome_vs_frozen_criteria": "context only vs original Stage 12 threshold"})
    for k, label in [("1", "center"), ("3", "3x3"), ("5", "5x5"), ("9", "9x9"), ("17", "17x17"), ("full", "full-field")]:
        m = s16["anatomy_by_scale"][k]["pooled"]["metrics"]
        rot = s16["rotation_by_scale"][k]
        rows.append({"analysis": f"multi-scale: {label}", "type": "multi-scale", "n": s16["anatomy_by_scale"][k]["pooled"]["n_patches_total"],
                     "unit": "patch (pooled testA+testB)", "glass_delta_A": round(m["A_angular_correspondence"]["glass_delta"], 4),
                     "pearson_r_B": round(m["B_order_magnitude_correlation"]["pearson_r"], 4), "mean_D_Q_C": round(m["C_tensor_similarity"]["mean_D_Q"], 4),
                     "rotation_angles_passed": f"{rot['n_angles_passed_vs_original_threshold']}/9", "outcome_vs_frozen_criteria": "context only, diagnostic (no new criterion)"})
    with open(os.path.join(TAB_DIR, "table4_ablation_multiscale.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    record_provenance("Table 4", "results_v2/validation/stage14/stage14_all_evaluations.json; .../A6_spatial_reduction/a6_results.json; results_v2/validation/stage16/stage16_results.json",
                       "see individual rows", "GlaS", "patch (pooled testA+testB)", "direct transcription; all 5 ablations and all 6 scales shown, no selection")
    print("[table4] done")


# ============================================================ TABLE 5: pathology + CRAG
def table5():
    rows = []
    for pid, label in [("P1", "S_DL vs binary grade"), ("P2", "S_DL vs ordinal grade"),
                        ("P3", "phi-dispersion vs binary grade"), ("P4", "phi-dispersion vs ordinal grade")]:
        r = s15["primary_pooled"][pid]
        eff = r.get("cohens_d", r.get("spearman_rho"))
        rows.append({"analysis": f"{pid} ({label})", "domain": "pathology (GlaS)", "n": 80, "unit": "image",
                     "statistic": round(eff, 4), "p_value": round(r["p_value"], 4), "adjusted_p": s15["holm_bonferroni_adjusted_p"][pid],
                     "outcome": "NOT SIGNIFICANT"})
    for mid, label in [("M1", "S_DL vs gland area"), ("M2", "S_DL vs gland count")]:
        r = s15["morphology"][mid]
        rows.append({"analysis": f"{mid} ({label})", "domain": "morphology (GlaS)", "n": 80, "unit": "image",
                     "statistic": round(r["spearman_rho"], 4), "p_value": round(r["p_value"], 4), "adjusted_p": "NA (not in corrected family)",
                     "outcome": "NOT SIGNIFICANT"})
    c1 = s15["confound_adjustment_C1"]
    rows.append({"analysis": "C1 (grade_label | gland area, OLS)", "domain": "confound adjustment (GlaS)", "n": c1["n"], "unit": "image",
                 "statistic": round(c1["grade_label_coef"], 5), "p_value": round(c1["grade_label_p"], 4), "adjusted_p": "NA (unconditional)",
                 "outcome": "NOT SIGNIFICANT"})
    r1 = s15["robustness_R1_patient_level"]
    rows.append({"analysis": "R1 (patient-level sensitivity)", "domain": "robustness (GlaS)", "n": r1["n_patients"], "unit": "patient",
                 "statistic": "NA (no formal test)", "p_value": "NA", "adjusted_p": "NA",
                 "outcome": f"INCONCLUSIVE ({r1['patients_with_mixed_grade_label_across_images']}/{r1['n_patients']} patients mixed-grade)"})
    for pop in ["train", "test", "pooled"]:
        m = s17["image_level_primary"][pop]["metrics"]
        rows.append({"analysis": f"CRAG image-level, {pop}", "domain": "external validation (CRAG)", "n": s17["image_level_primary"][pop]["n_images"], "unit": "image",
                     "statistic": f"Glass's delta_A={m['A_angular_correspondence']['glass_delta']:.4f}, r={m['B_order_magnitude_correlation']['pearson_r']:.4f}",
                     "p_value": round(m["B_order_magnitude_correlation"]["pearson_p"], 4), "adjusted_p": "NA (descriptive, not corrected family)",
                     "outcome": "NOT SUPPORTED (context vs GlaS thresholds)"})
    with open(os.path.join(TAB_DIR, "table5_pathology_crag.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    record_provenance("Table 5", "results_v2/validation/stage15/stage15_results.json; results_v2/validation/stage17_crag/stage17_results.json",
                       "see individual rows", "GlaS (pathology) / CRAG (external), never pooled", "image / patient (as labeled)",
                       "direct transcription", "1c3d2bbb53557649016bc477abfdeff1 / 44913f300fdc941a16437a3a5404103f")
    print("[table5] done")


# ============================================================ SUPPLEMENTARY: full evidence + traceability
def supplementary():
    import shutil
    shutil.copy(os.path.join(SYN_DIR, "MASTER_STATISTICAL_TABLE.csv"), os.path.join(TAB_DIR, "supplementary_full_statistical_evidence.csv"))
    shutil.copy(os.path.join(SYN_DIR, "MASTER_EVIDENCE_TABLE.csv"), os.path.join(TAB_DIR, "supplementary_evidence_domain_summary.csv"))
    with open(os.path.join(TAB_DIR, "supplementary_provenance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(provenance_records[0].keys()))
        w.writeheader()
        for r in provenance_records:
            w.writerow(r)
    print("[supplementary] done")


table1()
table2()
table3()
table4()
table5()
supplementary()
print("All 5 tables + supplementary complete.")
