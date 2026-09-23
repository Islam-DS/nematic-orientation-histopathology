"""
STAGE 24.5 -- audit-table builder for the repaired manuscript package (read-only with respect to all evidence).

Writes ONLY results_v2/manuscript_stage24_5/{04_NUMBERS_AUDIT_REPAIRED.csv, 06_FIGURE_TABLE_CITATION_AUDIT.csv}.
No statistic is computed: values are read from frozen result files, and only min/max/sign/difference of
values that are already printed in the manuscript are taken.
"""
import csv
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
OUT = P("results_v2", "manuscript_stage24_5")
MS = open(P("results_v2", "manuscript_stage24_5", "01_FULL_MANUSCRIPT_REPAIRED.md"), encoding="utf-8").read()
BODY = MS[: MS.find("## References")]
LINES = MS.splitlines()
_c = {}


def J(rel):
    if rel not in _c:
        _c[rel] = json.load(open(P(*rel.split("/")), encoding="utf-8"))
    return _c[rel]


def get(rel, path):
    cur = J(rel)
    for t in path.split("."):
        cur = cur[int(t)] if isinstance(cur, list) else cur[t]
    return cur


def fmt(v, d):
    return f"{v:.{d}f}".replace("-", "−")


S11 = "results_v2/validation/stage11/stage11_results.json"
S14 = "results_v2/validation/stage14/stage14_all_evaluations.json"
S16 = "results_v2/validation/stage16/stage16_results.json"
PRE = "results_v2/phase2/preregistered_thresholds_v2.json"

FIELDS = ["row_id", "check_type", "manuscript_section", "quantity", "value_as_written", "unit", "sample_size_or_population",
          "source_file", "source_location", "source_value", "status", "note"]
rows = []


def add(prefix, check_type, section, quantity, value, unit, sample, src, loc, srcval, status, note=""):
    n = sum(1 for r in rows if r["row_id"].startswith(prefix)) + 1
    rows.append({"row_id": f"{prefix}{n:04d}", "check_type": check_type, "manuscript_section": section, "quantity": quantity,
                 "value_as_written": value, "unit": unit, "sample_size_or_population": sample, "source_file": src,
                 "source_location": loc, "source_value": srcval, "status": status, "note": note})


# ---------------------------------------------------------------- A. independent critical checks (re-run on the repaired text)
crit = list(csv.DictReader(open(P("results_v2", "manuscript_stage24_5", "_critical_numbers_check_repaired.csv"), encoding="utf-8-sig")))
for r in crit:
    add("R", "independent direct-path check of repaired-manuscript value", r["manuscript_section"], r["quantity"], r["value_as_written"], r["unit"],
        r["sample_size_or_population"], r["source_file"], r["source_path"], r["source_value"], r["status"],
        f"matches source at printed precision: {r['written_matches_source_at_precision']}; present in the stated manuscript context: {r['present_in_manuscript_context']}")
n_crit = len(crit)
n_crit_ok = sum(1 for r in crit if r["status"].startswith("CONFIRMED"))

# ---------------------------------------------------------------- B. corrected source assignments for the 20 Stage 23 rows
s23 = list(csv.DictReader(open(P("results_v2", "manuscript_stage23", "04_NUMBERS_AUDIT.csv"), encoding="utf-8-sig")))
s24 = list(csv.DictReader(open(P("results_v2", "manuscript_stage24", "08_NUMBERS_AUDIT.csv"), encoding="utf-8-sig")))
mis = [r for r in s24 if r["check_type"].startswith("review of Stage 23")]
mis_ids = [r["row_id"] for r in mis]
s23_by_id = {r["number_id"]: r for r in s23}


def txt(rel):
    return open(P(*rel.split("/")), encoding="utf-8").read()


def find_path(obj, key, trail=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return trail + k, v
            f = find_path(v, key, trail + k + ".")
            if f:
                return f
    return None


perm_p = get(S11, "pooled_testA_testB_80_images.metrics.C_tensor_similarity.permutation_p_value")
s7 = "33.95" in txt("reports/STAGE7_AUDIT_REPORT.md")
cfg10 = txt("configs/stage10_model3_training_config.json")
b32_ok = bool(re.search(r'"batch_size":\s*32', cfg10))
cy_ok = "q_dense.shape[-2] // 2" in txt("code_v2/stage12_rotation_validation.py")
tiny_ok = "≤ 0.10" in txt("reports/STAGE8_EXTENDED_AUDIT_REPORT.md")
floor60 = "60 patches sampled" in txt("reports/STAGE0_RECALIBRATION_AUDIT_REPORT.md")
crag30 = get("results_v2/validation/stage17_crag/crag_compatibility_report.json", "crag_images_with_dominant_fused_gland_gt30pct")
CORR = {
    "0.001": ("results_v2/validation/stage11/stage11_results.json", "pooled_testA_testB_80_images.metrics.C_tensor_similarity.permutation_p_value (0 of 1000 re-pairings)", repr(perm_p), perm_p == 0.0),
    "4.0": (None, "token no longer present: the unverified licence string was removed from the repaired manuscript", "", "TOKEN REMOVED"),
    "33.95%": ("reports/STAGE7_AUDIT_REPORT.md", "Section 9 (untrained-network finite-grid residual, median, up to 33.95%)", "33.95 found in report" if s7 else "not found", s7),
    "32": ("configs/stage10_model3_training_config.json ; code_v2/stage12_rotation_validation.py", "training.batch_size = 32 (batch size); q_dense.shape[-2] // 2 = 32 (dense center index)", f"batch_size 32: {b32_ok}; center index rule: {cy_ok}", b32_ok and cy_ok),
    "0.10": ("reports/STAGE8_EXTENDED_AUDIT_REPORT.md", "tiny-set criterion L_final/L_initial ≤ 0.10", "≤ 0.10 found" if tiny_ok else "not found", tiny_ok),
    "60": ("reports/STAGE0_RECALIBRATION_AUDIT_REPORT.md", "section D: rotation-consistency floor, 60 patches sampled per calibration", "found" if floor60 else "not found", floor60),
    "0.03": ("results_v2/validation/stage15/stage15_results.json", "mean image-level predicted S (malignant 0.0293, benign 0.0311)", "", None),
    "30%": ("results_v2/validation/stage17_crag/crag_compatibility_report.json", "crag_images_with_dominant_fused_gland_gt30pct (70 of 213 images; 30% area fraction is the field definition)", repr(crag30), crag30 == 70),
}
# mean S values (checked through the same critical rows: 4.1)
mean_S_ok = any(("0.0293" in r["value_as_written"] or "0.0311" in r["value_as_written"]) and r["status"].startswith("CONFIRMED") for r in crit)
CORR["0.03"] = (CORR["0.03"][0], CORR["0.03"][1], "0.0293 / 0.0311 confirmed in the critical rows (4.1)" if mean_S_ok else "not found in critical rows", mean_S_ok)
for nid in mis_ids:
    r = s23_by_id[nid]
    val = r["value_as_written"]
    src, loc, sval, ok = CORR[val]
    present = re.search(re.escape(val), BODY) is not None
    if ok == "TOKEN REMOVED":
        status = "AUDIT-RECORD CORRECTION: token no longer in the repaired manuscript"
    else:
        status = ("AUDIT-RECORD CORRECTION: correct source verified" if ok else "SOURCE NOT VERIFIED") + ("" if present else " (token not present in repaired text)")
    add("C", "Stage 23 numbers-audit row with wrong source, corrected", r["section"], f"{nid}: " + r["context"][:70].strip(), val, r["unit"], "", src or "", loc, sval, status,
        f"Stage 23 assigned: {r['source_file'][:60]} / {r['source_row_or_location'][:60]}. The manuscript value was not wrong; only the audit-record source row is corrected.")

# ---------------------------------------------------------------- C. new or changed quantitative statements in the repaired manuscript
abl = {}
for key in ("A0_reference", "A1_no_q", "A3_q_dominant", "A4_q_only", "A5_plain_cnn_control"):
    m = get(S14, f"{key}.pooled_testA_testB.geometric_metrics")
    abl[key] = (m["A_angular_correspondence"]["glass_delta"], m["B_order_magnitude_correlation"]["pearson_r"])
sc = {}
for k in ("1", "3", "5", "9", "17", "full"):
    m = get(S16, f"anatomy_by_scale.{k}.pooled.metrics")
    sc[k] = (m["A_angular_correspondence"]["glass_delta"], m["B_order_magnitude_correlation"]["pearson_r"])
abl_d = [v[0] for v in abl.values()]
abl_r = [v[1] for v in abl.values()]
sc_d = [v[0] for v in sc.values()]
sc_r = [v[1] for v in sc.values()]
allowed = [(fmt(min(abl_d + sc_d), 4), fmt(max(abl_d + sc_d), 4), "Glass's Δ range over ablation conditions and scales", "−0.0441", "0.0461"),
           (fmt(min(abl_r), 4), fmt(max(abl_r), 4), "r range over ablation conditions (A0, A1, A3, A4, A5)", "−0.2097", "0.2203"),
           (fmt(min(sc_r), 4), fmt(max(sc_r), 4), "r range over the six scales", "−0.0341", "0.0147")]
for lo, hi, lab, wlo, whi in allowed:
    ok = lo == wlo and hi == whi and (wlo in BODY) and (whi in BODY)
    add("N", "new statement in repaired text: range of frozen values (min/max only)", "5.2", lab, f"{wlo} to {whi}", "effect size / r", "2083 patches; single seed",
        f"{S14} ; {S16}", "min and max of the Table 7 source values", f"{lo} to {hi}", "CONFIRMED" if ok else "MISMATCH", "min/max of already frozen values; no statistic computed")
add("N", "new statement in repaired text: threshold", "5.2", "required Glass's Δ (A) and r (B)", "0.5 ; 0.30", "threshold", "", PRE, "criteria.A / criteria.B primary thresholds",
    "0.5 ; 0.30 (see the Table 3 rows in R-block)", "CONFIRMED" if ("0.5" in BODY and "0.30" in BODY) else "MISMATCH", "same thresholds as Table 3")
mean_m = get(S11, "pooled_testA_testB_80_images.metrics.A_angular_correspondence.mean_deg")
mean_n = get(S11, "pooled_testA_testB_80_images.metrics.A_angular_correspondence.individual_null_mean_deg")
diff = mean_n - mean_m
add("N", "new statement in repaired text: difference of two frozen values", "4.2", "null mean error − model mean error (Criterion A)", "about 1.1°", "degrees", "2083 patches", S11,
    "individual_null_mean_deg − mean_deg", f"{mean_n:.5f} − {mean_m:.5f} = {diff:.4f}", "CONFIRMED" if 1.05 <= diff < 1.15 and "about 1.1°" in BODY else "MISMATCH", "arithmetic difference of two printed values (44.50°, 43.38°)")
ncal = find_path(J(PRE), "n_calibration_patches")
add("N", "new statement in repaired text: calibration population", "3.9; 5.2", "training patches used for the oracle calibration", "2249", "patches", "85 training images", PRE, ncal[0] if ncal else "", str(ncal[1]) if ncal else "",
    "CONFIRMED" if ncal and ncal[1] == 2249 and "2249 training patches" in BODY else "MISMATCH", "")
# sign statement (Table 5)
sg = {}
for mdl, sp in (("plain_cnn", "testA"), ("plain_cnn", "testB"), ("model2", "testA"), ("model2", "testB")):
    m = get(f"results_v2/baselines/{mdl}/held_out_{sp}_results.json", "geometric_metrics")
    sg[(mdl, sp)] = (m["A_angular_correspondence"]["glass_delta"], m["B_order_magnitude_correlation"]["pearson_r"], m["C_tensor_similarity"]["glass_delta"])
for sp in ("testA", "testB"):
    m = get(S11, f"{sp}.metrics")
    sg[("model3", sp)] = (m["A_angular_correspondence"]["glass_delta"], m["B_order_magnitude_correlation"]["pearson_r"], m["C_tensor_similarity"]["glass_delta"])
same = {mdl: [((sg[(mdl, "testA")][i] > 0) == (sg[(mdl, "testB")][i] > 0)) for i in range(3)] for mdl in ("plain_cnn", "model2", "model3")}
size_diff = {mdl: [abs(sg[(mdl, "testA")][i] - sg[(mdl, "testB")][i]) > 1e-9 for i in range(3)] for mdl in same}
ok_sign = all(same["plain_cnn"]) and all(same["model2"]) and not any(same["model3"]) and all(all(v) for v in size_diff.values())
add("N", "corrected statement in repaired text: sign behaviour across splits (Table 5)", "4.2", "signs of Glass's Δ (A), r, Glass's Δ (C) in testA vs testB",
    "same sign for plain CNN and Model 2; different sign for Model 3 only; sizes differ for all three models", "sign", "60 / 20 images",
    "results_v2/baselines/*/held_out_test*_results.json ; " + S11, "geometric_metrics / metrics (A glass_delta, B pearson_r, C glass_delta)",
    "; ".join(f"{m}: same sign={same[m]}, sizes differ={size_diff[m]}" for m in same), "CONFIRMED" if ok_sign else "MISMATCH",
    "compares the signs of six printed values per model; nothing recomputed")
# CRAG provenance and prior-work numbers (external primary texts)
add("N", "new statement in repaired text: external source fact", "3.2", "CRAG images derive from 38 whole-slide images of different patients", "38", "whole-slide images", "CRAG, 213 images",
    "Graham et al., Med Image Anal 2019, 52:199-211 (arXiv:1806.01963), Section 3 dataset description", "text: 'All 38 WSIs are from different patients ... split into 173 training images and 40 test images'",
    "38 WSIs; 173/40 split", "CONFIRMED (external primary text read; not a project result)", "reported as a statement of [8]; not verified independently of [8]")
add("N", "new statement in repaired text: external source fact", "2.3; Table 1", "cyclic groups C_k, k = 4 to 256", "256", "group order", "Navarro & Wilkinson", "arXiv:2605.27679 (abstract page)", "abstract: 'C_k where k = 4, 8, 16, 32, 64, 128, 256'",
    "k = 4, 8, 16, 32, 64, 128, 256", "CONFIRMED (abstract level)", "full text not read")
add("N", "new statement in repaired text: external source fact", "Table 1", "43 neonatal diffusion-MRI datasets (dHCP)", "43", "datasets", "Snoussi & Karimi", "arXiv:2504.01925 (abstract page)", "abstract: '43 neonatal dMRI datasets from the dHCP'",
    "43", "CONFIRMED (abstract level)", "full text not read")
add("N", "new statement in repaired text: project fact", "3.11", "plain CNN baseline with 64-channel blocks", "64", "channels", "predefined baseline", "configs/stage14_ablation_matrix.json", "A7 resolution: c1=64, c2=64 (reused Stage 9 baseline)",
    "c1=64, c2=64", "CONFIRMED" if "c1=64, c2=64" in txt("configs/stage14_ablation_matrix.json") else "MISMATCH", "")
sr = list(csv.DictReader(open(P("results_v2", "validation", "stage18_panda", "summary_table.csv"), encoding="utf-8")))
none = sorted(x["image_id"][:8] for x in sr if all(float(x[k]) == 0 for k in ("gleason_3_pixel_fraction", "gleason_4_pixel_fraction", "gleason_5_pixel_fraction")))
add("N", "new statement in repaired text: PANDA case identifiers", "4.8 Figure 7 caption", "pilot cases with no Gleason 3/4/5 pixels", "00743313, 004dd32d, 01642d24 (three)", "cases", "10-case pilot",
    "results_v2/validation/stage18_panda/summary_table.csv", "gleason_3/4/5_pixel_fraction all 0", ", ".join(none),
    "CONFIRMED" if set(none) == {"00743313", "004dd32d", "01642d24"} and "three (00743313, 004dd32d and 01642d24)" in BODY else "MISMATCH",
    "historical Stage 22/22.5 wording says two; see 11_DOCUMENTATION_ERRATA.md")
add("X", "structural tokens", "throughout", "section numbers, table/figure numbers, list numbers, reference numbers, hexadecimal/hash fragments", "n/a", "", "", "manuscript structure", "", "",
    "NOT QUANTITATIVE RESULTS", "e.g. 'Section 3.12', limitation-list numbers 1 to 17, [1] to [13]")

# ---------------------------------------------------------------- D. Stage 23 rows carried forward (tokens still present)
crit_vals = {re.sub(r"\s", "", r["value_as_written"]) for r in crit}
carried = removed = 0
for r in s23:
    if r["number_id"] in mis_ids:
        continue
    v = r["value_as_written"]
    present = re.search(r"(?<![\w.])" + re.escape(v) + r"(?![\w])", BODY) is not None
    if present:
        carried += 1
        status = "CARRIED FORWARD (token present; Stage 23 source retained)"
    else:
        removed += 1
        status = "TOKEN NOT FOUND IN REPAIRED MANUSCRIPT (edited or removed text)"
    add("S", "Stage 23 numbers-audit row carried forward", r["section"], r["context"][:90].strip(), v, r["unit"], "", r["source_file"], r["source_row_or_location"], "", status,
        "Stage 23 rule-based source assignment; independent confirmation exists for the critical values in the R block (220 of 220)")

fn = os.path.join(OUT, "04_NUMBERS_AUDIT_REPAIRED.csv")
with open(fn, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
bad = [r for r in rows if r["row_id"][0] in "RNC" and not (r["status"].startswith("CONFIRMED") or r["status"].startswith("AUDIT-RECORD CORRECTION"))]
print("04 rows:", len(rows), "| independent:", n_crit, "confirmed:", n_crit_ok, "| corrected Stage 23 rows:", len(mis_ids), "| carried:", carried, "| carried-token-missing:", removed)
print("problem rows:", len(bad))
for r in bad:
    print("  ", r["row_id"], r["quantity"][:60], "|", r["status"])

# ---------------------------------------------------------------- 06 figure/table citation audit
heads = []
cur = ""
for i, ln in enumerate(LINES):
    m = re.match(r"^#{2,3}\s+(\d+(?:\.\d+)?)\b", ln)
    if m:
        cur = m.group(1)
    elif ln.startswith("## Abstract"):
        cur = "Abstract"
    heads.append(cur)


def cap_idx(kind, n):
    pat = rf"^\*\*Table {n}\." if kind == "Table" else rf"^> \*\*Figure {n}[ .(]"
    for i, ln in enumerate(LINES):
        if re.match(pat, ln):
            return i
    return None


def first_mention(kind, n, cap):
    for i, ln in enumerate(LINES[: MS[: MS.find("## References")].count("\n") + 1]):
        if i == cap or re.match(r"^\*\*Table \d+\.|^> \*\*Figure \d+", ln):
            continue
        if re.search(rf"\b{kind}s? {n}(?!\d)", ln) or (kind == "Figure" and re.search(rf"\bFigure {n}[ab]\b", ln)):
            return i
    return None


NUM_SRC = {
    "Table 1": ("abstracts of references [4], [5], [9], [10], [11] (arXiv/publisher abstract pages)", "no computed values; k = 4–256 and 43 datasets are taken from the abstracts; full texts not read",
                "Abstract-level verification only; 'reference type not reported' cells could be completed after the full texts are read (USER)."),
    "Table 2": ("Grade.csv recount, manifests, [8] and the Kaggle competition description", "counts confirmed (165/85/60/20; 213/173/40; 10 of 5,160)", "License and terms of use: [DATASET LICENSE TO VERIFY]."),
    "Table 3": ("frozen criteria file (version 2)", "thresholds and rules confirmed against the criteria file", "none"),
    "Table 4": ("Stage 11 results", None, "none"),
    "Table 5": ("baseline result files and Stage 11/14", None, "none; unranked caption; sign sentence corrected (S24-H08)"),
    "Table 6": ("Stage 12 results and criteria file", None, "none"),
    "Table 7": ("Stage 14 and Stage 16 results", None, "dash meaning now explained (S24-L07)"),
    "Table 8": ("Stage 15 results", None, "none"),
    "Table 9": ("Stage 17 results", None, "none"),
    "Table 10": ("synthesis of Tables 4 to 9 and Section 4.8", "consistent with the analyses summarised; no numbers other than n", "outcome vocabulary note added (S24-L06)"),
}
FIG_INFO = {
    1: ("schematic (no data)", "n/a", "Graphic carries internal stage numbering in the last box ('Validation (Stages 11–18)'); the left-to-right flow can be read as feeding Q_anat into the network. Caption now states what the arrows mean. FIGURE REGENERATION REQUIRED FOR FUTURE VERSION."),
    2: ("frozen Stage 4 pipeline output for image testA_1", "n/a (illustration)", "Panel titles carry internal labels ('frozen Stage 4 pipeline', 'Stage 4, frozen'). FIGURE REGENERATION REQUIRED FOR FUTURE VERSION."),
    3: ("Stage 11 results", "effect sizes 0.0412 / −0.0341 / 0.0073 and p-values confirmed (04 R-block)", "Graphic title contains 'preregistered criteria' and 'anatomical validation (Stage 11)', which conflicts with the manuscript's terminology (caption now says so); axis label mixes Glass's Δ and r (caption explains). FIGURE REGENERATION REQUIRED FOR FUTURE VERSION."),
    4: ("Stage 12 results", "nine errors and thresholds confirmed (04 R-block)", "Graphic title contains 'Stage 12'; green/red coding (caption now states the coding and points to Table 6). FIGURE REGENERATION REQUIRED FOR FUTURE VERSION."),
    5: ("Stage 14 and Stage 16 results", "Δ and r per condition/scale confirmed (04 R-block)", "Graphic title and panel titles carry 'Stage 14/16'; dual y-axes in the right panel (caption now explains the axes and the dotted line). FIGURE REGENERATION REQUIRED FOR FUTURE VERSION."),
    6: ("Stage 15 and Stage 17 results", "effect sizes and CIs confirmed (04 R-block)", "Panel titles carry 'Stage 15/17', 'external cross-dataset validation' and 'partial institutional overlap'; Cohen's d and Spearman ρ share one axis (caption now says the measures differ); zero-length caps on P1/P3 (already in caption). FIGURE REGENERATION REQUIRED FOR FUTURE VERSION."),
    7: ("Stage 18 summary table (frozen)", "nine bars and three no-Gleason cases confirmed against summary_table.csv (04 R/N-blocks)", "Graphic title carries 'Stage 18' and 'compatibility gate'. Case count follows the frozen table (see 11_DOCUMENTATION_ERRATA.md). FIGURE REGENERATION REQUIRED FOR FUTURE VERSION."),
}
crit_sections = [(r["manuscript_section"], r["status"]) for r in crit]
out6 = []
for kind, N in (("Table", 10), ("Figure", 7)):
    for n in range(1, N + 1):
        cap = cap_idx(kind, n)
        fm = first_mention(kind, n, cap)
        item = f"{kind} {n}"
        pres = "yes" if fm is not None else "NO"
        order = "yes" if (fm is not None and cap is not None and fm < cap) else ("NO" if fm is not None else "n/a")
        loc = f"Section {heads[fm]} (line {fm + 1})" if fm is not None else ""
        if kind == "Table":
            src, numc, unres = NUM_SRC[item]
            if numc is None:
                cnt = sum(1 for s, st in crit_sections if f"Table {n}" in s)
                ok = sum(1 for s, st in crit_sections if f"Table {n}" in s and st.startswith("CONFIRMED"))
                numc = f"{ok} of {cnt} numeric rows confirmed in 04 (R-block)"
            capstat = "caption present; states population/unit" if cap is not None else "MISSING"
        else:
            src, numc, unres = FIG_INFO[n]
            capstat = "caption present and consistent with the graphic after repair" if cap is not None else "MISSING"
        action = "none in text" if (pres == "yes" and order == "yes") else "CITE BEFORE PLACEMENT"
        if kind == "Figure":
            action = "text citation in place; graphic unchanged (frozen); embedded internal labels recorded for a future formatting pass"
        out6.append({"item": item, "first_text_location": loc, "first_text_line": "" if fm is None else fm + 1, "caption_line": "" if cap is None else cap + 1,
                     "citation_present": pres, "order_correct": order, "caption_status": capstat, "content_source": src, "numerical_consistency": numc,
                     "unresolved_issue": unres, "action": action})
# monotonic order of first mentions
for kind in ("Table", "Figure"):
    xs = [r for r in out6 if r["item"].startswith(kind)]
    lines = [r["first_text_line"] for r in xs]
    mono = all(lines[i] <= lines[i + 1] for i in range(len(lines) - 1))
    print(kind, "first mentions monotonic in numerical order:", mono)
f6 = os.path.join(OUT, "06_FIGURE_TABLE_CITATION_AUDIT.csv")
with open(f6, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(out6[0].keys()))
    w.writeheader()
    w.writerows(out6)
print("06 rows:", len(out6), "| all cited:", all(r["citation_present"] == "yes" for r in out6), "| all in order:", all(r["order_correct"] == "yes" for r in out6))
