"""
STAGE 22.5 -- builds the machine-generated parts of the corrected manuscript
package:

  results_v2/manuscript_stage21_corrected/12_claim_evidence_matrix.csv
  results_v2/manuscript_stage21_corrected/13_numbers_and_sources.csv
  results_v2/tables/stage20_corrected/supplementary_full_statistical_evidence_corrected.csv
  results_v2/tables/stage20_corrected/supplementary_evidence_domain_summary_corrected.csv
  results_v2/tables/stage20_corrected/supplementary_provenance_corrected.csv   (with --finalize)
  the checksum block of 15_STAGE21_CORRECTED_AUDIT.md                          (with --finalize)

Usage:  python stage225_build_corrected_package.py [--finalize] [--verify]

READ-ONLY with respect to every frozen artifact. No statistic, p-value or
confidence interval is computed: values are read from frozen result files or
transcribed with the source named; min/max are taken over frozen per-angle
values only; patient-overlap counts are simple counts of Grade.csv rows.
Formatting-only changes: CI endpoints rounded to 4 decimals; "FAIL" -> "NOT MET".
"""
import csv
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
OUT = P("results_v2", "manuscript_stage21_corrected")
TAB = P("results_v2", "tables", "stage20_corrected")
ORIG = P("results_v2", "manuscript_stage21")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
SCRIPT_REL = "code_v2/stage225_build_corrected_package.py"

# ------------------------------------------------------------------ helpers
_cache = {}


def load(rel):
    if rel not in _cache:
        with open(P(*rel.split("/")), encoding="utf-8") as f:
            _cache[rel] = json.load(f)
    return _cache[rel]


def resolve(obj, path):
    cur = obj
    for tok in path.split("."):
        m = re.fullmatch(r"(\w+)\[(\w+)\]", tok)
        if m:
            cur = [e for e in cur[m.group(1)] if isinstance(e, dict) and e.get("id") == m.group(2)][0]
            continue
        if isinstance(cur, list):
            cur = cur[int(tok)]
        else:
            cur = cur[tok] if tok in cur else cur[int(tok)]
    return cur


def num_of(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        return float(m.group(0)) if m else None
    return None


def read_text(rel):
    with open(P(*rel.split("/")), encoding="utf-8") as f:
        return f.read()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------ GlaS patient counts
def patient_counts():
    with open(P("data", "glas", "Warwick_QU_Dataset", "Grade.csv"), encoding="latin1", newline=None) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    rec = []
    for ln in lines[1:]:
        parts = [x.strip() for x in ln.split(",")]
        name, pid = parts[0], int(parts[1])
        rec.append((re.sub(r"_\d+$", "", name), pid))
    train = {p for s, p in rec if s == "train"}
    held = [(s, p) for s, p in rec if s in ("testA", "testB")]
    held_p = {p for s, p in held}
    overlap_p = held_p & train
    overlap_img = sum(1 for s, p in held if p in train)
    return {"held_patients": len(held_p), "held_patients_with_train": len(overlap_p),
            "held_images": len(held), "held_images_with_train_patient": overlap_img,
            "all_patients": len({p for s, p in rec})}


PC = patient_counts()
assert PC["held_patients"] == 12 and PC["held_patients_with_train"] == 11
assert PC["held_images"] == 80 and PC["held_images_with_train_patient"] == 79 and PC["all_patients"] == 16

# ------------------------------------------------------ text-source checks
TEXT_CHECKS = {
    "STAGE8_EXTENDED_AUDIT_REPORT.md|0.0736": ("reports/STAGE8_EXTENDED_AUDIT_REPORT.md", r"0\.07364"),
    "STAGE7_AUDIT_REPORT.md|15": ("reports/STAGE7_AUDIT_REPORT.md", r"15/15 tests pass"),
    "stage18_metadata_audit.md|10": ("data/PANDA/stage18_metadata_audit.md", r"radboud_pilot_manifest"),
    "stage18_metadata_audit.md|5160": ("data/PANDA/stage18_metadata_audit.md", r"5,160"),
    "stage18_metadata_audit.md|10616": ("data/PANDA/stage18_metadata_audit.md", r"10,616"),
    "STAGE17_AUDIT_REPORT.md|29399": ("reports/STAGE17_AUDIT_REPORT.md", r"29,399"),
    "STAGE7_AUDIT_REPORT.md|34": ("reports/STAGE7_AUDIT_REPORT.md", r"~34%"),
    "STAGE15_AUDIT_REPORT.md|4": ("reports/STAGE15_AUDIT_REPORT.md", r"only 4 benign images"),
    "STAGE15_AUDIT_REPORT.md|16": ("reports/STAGE15_AUDIT_REPORT.md", r"11 of 16 patients"),
    "STAGE15_AUDIT_REPORT.md|11": ("reports/STAGE15_AUDIT_REPORT.md", r"11 of 16 patients"),
    "STAGE8_EXTENDED_AUDIT_REPORT.md|8": ("reports/STAGE8_EXTENDED_AUDIT_REPORT.md", r"identical 8"),
    "STAGE8_EXTENDED_AUDIT_REPORT.md|1000": ("reports/STAGE8_EXTENDED_AUDIT_REPORT.md", r"500\s*(?:→|->)\s*1000"),
    "STAGE8_EXTENDED_AUDIT_REPORT.md|500": ("reports/STAGE8_EXTENDED_AUDIT_REPORT.md", r"500\s*(?:→|->)\s*1000"),
}
summary_rows = list(csv.DictReader(open(P("results_v2", "validation", "stage18_panda", "summary_table.csv"), encoding="utf-8")))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def verify_original_row(r):
    """Re-derive an original (Stage 21) row's number from its named source. Returns (ok, method)."""
    num = float(r["number"])
    sf, fld = r["source_file"], r["source_field"]
    dec = len(r["number"].split(".")[1]) if "." in r["number"] else 0
    tol = 0.5 * 10 ** (-dec) + 1e-12
    if sf.endswith(".json"):
        path = fld.split(" ")[0]
        v = resolve(load(sf), path)
        if "D_rotation_consistency" in path and isinstance(v, str):
            m = re.search(r"at least (\d+) of the (\d+) non-zero", v)
            return bool(m) and int(m.group(1)) == int(num), "json rule text"
        if "gt30pct" in fld and int(num) == 30 and "dominant" in path:
            return True, "json field name (30 % encoded)"
        fv = num_of(v)
        return fv is not None and abs(fv - num) <= tol, "json field"
    if sf.endswith("summary_table.csv"):
        lab = r["label"]
        v4 = [fnum(x["gleason_4_largest_vs_benign_median_ratio"]) for x in summary_rows]
        v4 = [v for v in v4 if v is not None]
        v5 = [fnum(x["gleason_5_largest_vs_benign_median_ratio"]) for x in summary_rows]
        v5 = [v for v in v5 if v is not None]
        nc = [fnum(x["benign_epithelium_n_components"]) for x in summary_rows]
        nc = [v for v in nc if v and v > 0]
        g4 = [fnum(x["gleason_4_pixel_fraction"]) for x in summary_rows]
        if "smallest Gleason-4" in lab:
            ok = min(v4) == num
        elif "largest Gleason-5" in lab:
            ok = max(v5) == num
        elif "with Gleason 4 and a computable" in lab:
            ok = len(v4) == num
        elif "containing Gleason 4" in lab:
            ok = sum(1 for v in g4 if v and v > 0) == num
        elif "PANDA Gleason-4 ratio" in lab:
            case = re.search(r"case (\w+)", lab).group(1)
            hit = [x for x in summary_rows if x["image_id"].startswith(case)]
            ok = bool(hit) and fnum(hit[0]["gleason_4_largest_vs_benign_median_ratio"]) == num
        elif "minimum" in lab:
            ok = min(nc) == num
        elif "maximum" in lab:
            ok = max(nc) == num
        else:
            ok = False
        return ok, "summary_table.csv"
    chk = TEXT_CHECKS.get(f"{os.path.basename(sf)}|{r['number']}")
    if chk:
        return re.search(chk[1], read_text(chk[0]), re.I) is not None, "report text"
    return False, "no check"


# ------------------------------------------------------------ numbers table
def fix_text(s):
    s = s.replace("preregistered per-angle threshold", "frozen per-angle threshold")
    s = s.replace("preregistration", "frozen criteria file")
    s = s.replace("Preregistered", "Frozen").replace("preregistered", "frozen")
    return s


HEADER13 = ["number", "label", "unit", "dataset", "analysis", "source_file", "source_field", "stage",
            "interpretation", "allowed_use", "verified_against_source", "stage22_5_change"]
ALLOW_MAIN = "may be reported with the stated unit, dataset, and effect-size context"
ALLOW_CTX = "context/descriptive only; must not be used to override the frozen pooled result"


def build_numbers():
    rows = []
    for r in csv.DictReader(open(P("results_v2", "manuscript_stage21", "13_numbers_and_sources.csv"), encoding="utf-8")):
        ok, how = verify_original_row(r)
        assert ok, ("original row failed re-verification", r["label"], how)
        new = {k: r[k] for k in ("number", "label", "unit", "dataset", "analysis", "source_file", "source_field", "stage", "interpretation", "allowed_use")}
        changed = False
        for k in ("label", "analysis", "interpretation", "allowed_use"):
            f = fix_text(new[k])
            if f != new[k]:
                new[k] = f
                changed = True
        new["verified_against_source"] = f"yes ({how})"
        new["stage22_5_change"] = "wording only" if changed else "none (carried over from Stage 21; re-verified)"
        rows.append(new)

    def add(number, label, unit, dataset, analysis, source_file, source_field, stage, interp, allowed, value_fn=None, text=None):
        if value_fn is not None:
            v = value_fn()
            ok = abs(float(v) - float(number)) <= 0.5 * 10 ** (-(len(str(number).split(".")[1]) if "." in str(number) else 0)) + 1e-12
            how = "json field (min/max over frozen per-angle values where stated)"
        else:
            ok = re.search(text[1], read_text(text[0]), re.I) is not None
            how = "source text/code"
        assert ok, ("new row failed verification", label)
        rows.append({"number": number, "label": label, "unit": unit, "dataset": dataset, "analysis": analysis,
                     "source_file": source_file, "source_field": source_field, "stage": stage,
                     "interpretation": interp, "allowed_use": allowed, "verified_against_source": f"yes ({how})",
                     "stage22_5_change": "added in Stage 22.5 (existing frozen value or simple count; no new statistic)"})

    S12 = "results_v2/validation/stage12/stage12_results.json"
    S16 = "results_v2/validation/stage16/stage16_results.json"
    S11 = "results_v2/validation/stage11/stage11_results.json"
    S15 = "results_v2/validation/stage15/stage15_results.json"
    ang = [15, 30, 37, 45, 60, 90, 123, 150, 173]
    ctxs = "descriptive, non-gating additional check of the frozen rotation criterion; not a criterion"
    for i, a in enumerate(ang, start=1):
        sv = load(S12)["per_angle"][i]["S_abs_diff_mean"]
        pv = load(S12)["per_angle"][i]["phi_deviation_from_expected_shift_deg_mean"]
        add(round(sv, 4), f"H2 scalar-invariance check: mean |S(R x) - S(x)| at {a} deg", "S units", "GlaS", "Stage 12 non-gating additional check", S12,
            f"per_angle.{i}.S_abs_diff_mean", "12", "non-gating additional check", ctxs, lambda i=i: load(S12)["per_angle"][i]["S_abs_diff_mean"])
        add(round(pv, 2), f"phi-shift check: mean deviation from expected shift at {a} deg", "degrees", "GlaS", "Stage 12 non-gating additional check", S12,
            f"per_angle.{i}.phi_deviation_from_expected_shift_deg_mean", "12", "non-gating additional check", ctxs, lambda i=i: load(S12)["per_angle"][i]["phi_deviation_from_expected_shift_deg_mean"])
    pa = load(S12)["per_angle"][1:]
    add(round(min(x["S_abs_diff_mean"] for x in pa), 4), "Minimum over 9 angles of mean |delta S|", "S units", "GlaS", "Stage 12 non-gating additional check", S12,
        "min over per_angle[1..9].S_abs_diff_mean", "12", "range endpoint (frozen per-angle values)", ctxs, lambda: min(x["S_abs_diff_mean"] for x in pa))
    add(round(max(x["S_abs_diff_mean"] for x in pa), 4), "Maximum over 9 angles of mean |delta S|", "S units", "GlaS", "Stage 12 non-gating additional check", S12,
        "max over per_angle[1..9].S_abs_diff_mean", "12", "range endpoint (frozen per-angle values)", ctxs, lambda: max(x["S_abs_diff_mean"] for x in pa))
    add(round(min(x["phi_deviation_from_expected_shift_deg_mean"] for x in pa), 2), "Minimum over 9 angles of mean phi deviation", "degrees", "GlaS", "Stage 12 non-gating additional check", S12,
        "min over per_angle[1..9].phi_deviation_from_expected_shift_deg_mean", "12", "range endpoint (frozen per-angle values)", ctxs, lambda: min(x["phi_deviation_from_expected_shift_deg_mean"] for x in pa))
    add(round(max(x["phi_deviation_from_expected_shift_deg_mean"] for x in pa), 2), "Maximum over 9 angles of mean phi deviation", "degrees", "GlaS", "Stage 12 non-gating additional check", S12,
        "max over per_angle[1..9].phi_deviation_from_expected_shift_deg_mean", "12", "range endpoint (frozen per-angle values)", ctxs, lambda: max(x["phi_deviation_from_expected_shift_deg_mean"] for x in pa))
    add(round(min(x["model_error_median"] for x in pa), 4), "Minimum over 9 angles of median rotation error", "normalized error", "GlaS", "Stage 12", S12,
        "min over per_angle[1..9].model_error_median", "12", "range endpoint (frozen per-angle values)", ALLOW_CTX, lambda: min(x["model_error_median"] for x in pa))
    add(round(max(x["model_error_median"] for x in pa), 4), "Maximum over 9 angles of median rotation error", "normalized error", "GlaS", "Stage 12", S12,
        "max over per_angle[1..9].model_error_median", "12", "range endpoint (frozen per-angle values)", ALLOW_CTX, lambda: max(x["model_error_median"] for x in pa))
    for i, a in enumerate(ang, start=1):
        v = load(S16)["rotation_by_scale"]["full"]["per_angle"][i]["pooled_model_error_mean"]
        add(round(v, 4), f"Full-field rotation error at {a} deg (diagnostic; thresholds calibrated for center-pixel convention)", "normalized error", "GlaS", "Stage 16 full-field scale",
            S16, f"rotation_by_scale.full.per_angle.{i}.pooled_model_error_mean", "16", "diagnostic reduction; not a criterion result", ALLOW_CTX,
            lambda i=i: load(S16)["rotation_by_scale"]["full"]["per_angle"][i]["pooled_model_error_mean"])
    add(2083, "Stage 16 rotation analysis patches (all pooled held-out valid patches)", "patches", "GlaS", "Stage 16 rotation", S16,
        "rotation_by_scale.full.per_angle.1.n_patches", "16", "pooled testA+testB, n=2083", ALLOW_MAIN, lambda: load(S16)["rotation_by_scale"]["full"]["per_angle"][1]["n_patches"])
    add(2083, "Stage 14 A6 rotation analysis patches", "patches", "GlaS", "Stage 14 A6", "results_v2/validation/stage14/A6_spatial_reduction/a6_results.json",
        "n_patches", "14", "pooled testA+testB, n=2083", ALLOW_MAIN, lambda: load("results_v2/validation/stage14/A6_spatial_reduction/a6_results.json")["n_patches"])
    add(44.5046, "Criterion A individual cross-pairing null mean angular error", "degrees", "GlaS", "Stage 11", S11,
        "pooled_testA_testB_80_images.metrics.A_angular_correspondence.individual_null_mean_deg", "11", "null mean beside model mean 43.38", ALLOW_MAIN,
        lambda: load(S11)["pooled_testA_testB_80_images"]["metrics"]["A_angular_correspondence"]["individual_null_mean_deg"])
    add(0.5108, "Criterion C null-of-mean mean D_Q", "D_Q", "GlaS", "Stage 11", S11,
        "pooled_testA_testB_80_images.metrics.C_tensor_similarity.null_of_mean_mean", "11", "null mean beside observed mean 0.5092", ALLOW_MAIN,
        lambda: load(S11)["pooled_testA_testB_80_images"]["metrics"]["C_tensor_similarity"]["null_of_mean_mean"])
    add(1000, "Permutation re-pairings for Criteria A and C", "permutations", "GlaS", "Stage 11", "code_v2/stage11_geometric_validation.py", "N_PERM",
        "11", "a permutation count of 0 of 1000 means p < 0.001", ALLOW_MAIN, text=("code_v2/stage11_geometric_validation.py", r"N_PERM = 1000"))
    add(128, "Patch size", "pixels", "GlaS", "patch extraction", "code/data_prep.py", "PATCH_SIZE", "3", "patch edge length", ALLOW_MAIN, text=("code/data_prep.py", r"PATCH_SIZE = 128"))
    add(96, "Patch stride", "pixels", "GlaS", "patch extraction", "code/data_prep.py", "STRIDE", "3", "stride < patch size, so adjacent patches overlap", ALLOW_MAIN, text=("code/data_prep.py", r"STRIDE = 96"))
    add(0.0293, "Mean S_DL, malignant images (image-level)", "S units", "GlaS", "Stage 15 P1", S15, "primary_pooled.P1.mean_malignant", "15",
        "context for small-S denominators in the rotation error", ALLOW_CTX, lambda: load(S15)["primary_pooled"]["P1"]["mean_malignant"])
    add(0.0311, "Mean S_DL, benign images (image-level)", "S units", "GlaS", "Stage 15 P1", S15, "primary_pooled.P1.mean_benign", "15",
        "context for small-S denominators in the rotation error", ALLOW_CTX, lambda: load(S15)["primary_pooled"]["P1"]["mean_benign"])
    add(43, "Malignant images among the 80 held-out images", "images", "GlaS", "Stage 15", S15, "primary_pooled.P1.n_malignant", "15", "image-level label", ALLOW_MAIN,
        lambda: load(S15)["primary_pooled"]["P1"]["n_malignant"])
    add(37, "Benign images among the 80 held-out images", "images", "GlaS", "Stage 15", S15, "primary_pooled.P1.n_benign", "15", "image-level label", ALLOW_MAIN,
        lambda: load(S15)["primary_pooled"]["P1"]["n_benign"])
    add(11, "Held-out patients (of 12) that also contribute training images", "patients", "GlaS", "patient structure (count only)",
        "data/glas/Warwick_QU_Dataset/Grade.csv", "count of distinct patient IDs in testA/testB rows that also occur in train rows", "22",
        "canonical split is image-level; not patient-independent", ALLOW_MAIN, lambda: PC["held_patients_with_train"])
    add(79, "Held-out images (of 80) from patients that also contribute training images", "images", "GlaS", "patient structure (count only)",
        "data/glas/Warwick_QU_Dataset/Grade.csv", "count of testA/testB rows whose patient ID also occurs in train rows", "22",
        "overlap limits independence; cannot explain the negative effect sizes as positive bias", ALLOW_MAIN, lambda: PC["held_images_with_train_patient"])
    add(0.139, "Stage 8 tiny-set L_final/L_initial, original 500-epoch run", "ratio", "GlaS (8 train patches)", "Stage 8 original run", "reports/STAGE8_EXTENDED_AUDIT_REPORT.md",
        "Section 4 table (ratio 0.13902)", "8", "criterion <= 0.10 not met before the extension", "engineering/optimization statement only", text=("reports/STAGE8_EXTENDED_AUDIT_REPORT.md", r"0\.13902"))
    add(1816, "Model 3 train_inner patches", "patches", "GlaS", "Stage 10", "reports/STAGE10_AUDIT_REPORT.md", "Section 3 ('1816 train_inner / 433 dev')", "10",
        "image-level 68/17 split of the 85 training images", ALLOW_MAIN, text=("reports/STAGE10_AUDIT_REPORT.md", r"1816 train_inner / 433 dev"))
    add(433, "Model 3 development patches", "patches", "GlaS", "Stage 10", "reports/STAGE10_AUDIT_REPORT.md", "Section 3 ('1816 train_inner / 433 dev')", "10",
        "image-level 68/17 split of the 85 training images", ALLOW_MAIN, text=("reports/STAGE10_AUDIT_REPORT.md", r"1816 train_inner / 433 dev"))
    for case, val in (("00bbc148", 28.7), ("018eabc8", 96.1), ("00951a7f", 112.0), ("0068d4c7", 0.2)):
        hit = [x for x in summary_rows if x["image_id"].startswith(case)][0]
        add(val, f"PANDA Gleason-3 ratio (case {case})", "ratio", "PANDA Radboud pilot", "Stage 18 compatibility gate", "results_v2/validation/stage18_panda/summary_table.csv",
            "gleason_3_largest_vs_benign_median_ratio", "18", "largest same-class connected region / same-case benign median component area; semantic regions, not gland instances",
            ALLOW_CTX, lambda hit=hit: float(hit["gleason_3_largest_vs_benign_median_ratio"]))
    add(1, "PANDA pilot cases containing Gleason 5", "cases", "PANDA Radboud pilot", "Stage 18 compatibility gate", "results_v2/validation/stage18_panda/summary_table.csv",
        "rows with gleason_5_pixel_fraction > 0", "18", "case 00928370", ALLOW_CTX, lambda: sum(1 for x in summary_rows if (fnum(x["gleason_5_pixel_fraction"]) or 0) > 0))
    add(1, "PANDA pilot Gleason-4 cases with no benign reference", "cases", "PANDA Radboud pilot", "Stage 18 compatibility gate", "results_v2/validation/stage18_panda/summary_table.csv",
        "rows with gleason_4_pixel_fraction > 0 and benign_epithelium_n_components = 0", "18", "case 006f6aa3; cannot contribute a ratio", ALLOW_CTX,
        lambda: sum(1 for x in summary_rows if (fnum(x["gleason_4_pixel_fraction"]) or 0) > 0 and (fnum(x["benign_epithelium_n_components"]) or 0) == 0))
    return rows


# ----------------------------------------------------------- claim matrix
HEADER12 = ["claim", "evidence_source", "dataset", "sample_unit", "sample_size", "statistic", "effect_size", "p_value",
            "confidence_interval", "frozen_criterion_status", "status", "allowed_manuscript_wording", "forbidden_overclaim", "stage22_5_change"]
FROZEN = "yes (prospectively frozen internal criteria, v2)"


def build_claims():
    C = []
    C.append(["The typed rank-2 (D8.irrep(1,2)) representation and the Q construction are mathematically and numerically implemented as specified",
              "Stage 7 (15/15 tests); Stage 4 reference generation", "N/A / GlaS", "architecture tests; 165 images", "15 tests; 165 images",
              "irrep matrix equals R(2a) to 0.00e+00", "NA", "NA", "NA", "engineering criterion", "SUPPORTED",
              "The Q construction and the typed representation were implemented and verified against the analytic transformation law.",
              "That the whole network is exactly equivariant on a finite pixel grid; that the representation is anatomically meaningful.",
              "wording (anatomical reference terminology)"])
    C.append(["Model 3 was trained once (seed 42); the checkpoint reloads bit-identically and the controlled tiny-set criterion was met after one extension of the epoch limit",
              "Stage 8 extended run; Stage 10", "GlaS train", "patch (8 tiny-set patches); 1816/433 train/dev patches", "8; 1816/433",
              "L_final/L_initial = 0.0736 after extension (0.1390 in the original 500-epoch run); checkpoint reload bit-identical (Stage 10)", "NA", "NA", "NA", "engineering criterion", "SUPPORTED",
              "The full Model 3 training was performed once with seed 42. Checkpoint reload produced bit-identical outputs, and the controlled tiny-set run was reproduced exactly.",
              "Any claim of deterministic or multi-run-reproducible full training; that optimization success implies anatomical correspondence or generalization.",
              "earlier reproducibility wording replaced; original 500-epoch ratio and single-run status added"])
    C.append(["Criterion A: Model 3 predicted orientation corresponds to the anatomical reference orientation", "results_v2/validation/stage11/stage11_results.json", "GlaS testA+testB (canonical image-level held-out set)", "patch (overlapping)", "2083",
              "Glass's delta = 0.0412 (threshold >= 0.5); mean error 43.38 deg vs individual null mean 44.50 deg", "0.0412",
              "permutation p = 0.002 (secondary, non-binding for Criterion A; descriptive because patches overlap)", "NA", FROZEN, "NOT DEMONSTRATED",
              "Criterion A was not met (Glass's delta 0.0412 vs the frozen 0.5); the permutation p-value of 0.002 is a secondary, non-binding check and is not evidence of agreement.",
              "That a small permutation p indicates anatomical agreement; that the model 'partially learned' orientation.", "effect size first; p labelled secondary/non-binding; patch dependence stated"])
    C.append(["Criterion B: Model 3 predicted order magnitude corresponds to the anatomical reference order magnitude", "results_v2/validation/stage11/stage11_results.json", "GlaS testA+testB", "patch (overlapping)", "2083",
              "Pearson r = -0.0341 (threshold >= 0.30)", "-0.0341", "p = 0.120 (descriptive)", "[-0.076, 0.009] (patch bootstrap; descriptive)", FROZEN, "NOT DEMONSTRATED",
              "Criterion B was not met (r = -0.0341; 95% CI includes zero).", "Any positive-correlation language; per-split values (testB r = 0.308) as evidence.", "patch dependence stated"])
    C.append(["Criterion C (permutation p < 0.01 AND Glass's delta >= 0.5): predicted tensor vs anatomical reference tensor", "results_v2/validation/stage11/stage11_results.json", "GlaS testA+testB", "patch (overlapping)", "2083",
              "Glass's delta = 0.0073 (threshold >= 0.5); mean D_Q 0.5092 vs null-of-mean 0.5108", "0.0073", "permutation p < 0.001 (0 of 1000 re-pairings; descriptive)", "NA", FROZEN, "NOT DEMONSTRATED",
              "Criterion C was not met: the permutation condition was satisfied (p < 0.001) but the effect size (Glass's delta 0.0073) was negligible against the required 0.5.",
              "That 'the tensor test is significant'; that the permutation p supports correspondence; reporting the permutation p as zero.", "claim restated as the two-part criterion; p < 0.001 instead of 0; effect size first"])
    C.append(["Criterion D: Model 3 meets the frozen empirical center-pixel, pixel-domain rotation-consistency criterion", "results_v2/validation/stage12/stage12_results.json", "GlaS testA+testB", "patch", "2083",
              "4/9 non-zero angles below threshold (>= 7/9 required)", "NA", "NA", "NA", FROZEN, "NOT DEMONSTRATED",
              "Empirical center-pixel, pixel-domain rotation consistency did not meet the frozen criterion: four of nine non-zero angles (90, 123, 150, 173 deg) were below their thresholds; seven were required. The measurement samples dense pixel (32,32) (rotation center approximately (31.5,31.5)); only 90 deg of the nine angles is an exact pixel-lattice rotation, the others involve interpolation; the normalized error is affected by small-S denominators.",
              "Any general conclusion about the network's equivariance drawn from this measurement; that passing large angles is meaningful; continuous equivariance of the whole network.",
              "wording per Correction 7; measurement scope added"])
    C.append(["H2 scalar-invariance and phi-shift observations (non-gating additional checks of Criterion D)", "results_v2/validation/stage12/stage12_results.json (per_angle S_abs_diff_mean, phi_deviation_from_expected_shift_deg_mean)", "GlaS testA+testB", "patch", "2083",
              "mean |delta S| 0.031-0.039; mean phi deviation 16.6-22.2 deg across the nine angles (context: image-level mean predicted S about 0.03)", "NA", "NA", "NA", "non-gating additional checks in the frozen file", "DESCRIPTIVE",
              "As non-gating additional checks, mean |S(R x) - S(x)| ranged from 0.031 to 0.039 and the mean deviation of phi from the expected shift ranged from 16.6 to 22.2 degrees across the nine angles.",
              "Treating these as a criterion or as a pass/fail result; describing |delta S| as small without the scale of S.", "new claim (Correction 8)"])
    C.append(["The Model 2 channel-to-angle assignment differs from randomized assignments", "results_v2/validation/stage13/stage13_results.json", "GlaS testA / testB", "patch", "1470 / 613",
              "z = -1.7106 (testA), -1.7044 (testB); 100 permutations, seed 0", "NA", "one-sided 0.09; two-sided 0.14 (both splits)", "NA", "descriptive (not a frozen criterion)", "INCONCLUSIVE",
              "For Model 2 only, the observed order magnitude lay below the permutation-null mean in both splits, without reaching p < 0.05.", "Anti-nematic behaviour; any statement about Model 3; biological interpretation of the sign.", "frozen-status label reworded"])
    C.append(["None of the tested ablation conditions (A0, A1, A3, A4, A5) met the frozen anatomical criteria, under the fixed single-seed configuration", "results_v2/validation/stage14/stage14_all_evaluations.json", "GlaS testA+testB", "patch", "2083",
              "Glass's delta_A across A0/A1/A3/A4/A5 ranges -0.044 to 0.046; r ranges -0.210 to 0.220 (one seed per condition)", "see 13_numbers_and_sources.csv", "NA", "NA", "frozen diagnostic matrix (not a criterion)", "NOT DEMONSTRATED",
              "Under the fixed single-seed configuration, none of the tested design variations produced the frozen anatomical correspondence; conditions were not ranked.",
              "That any ablation is 'better'; that equivariance is unnecessary or harmful; A1's r = 0.220 as learned correspondence (its Q head was untrained); any claim about variability across seeds.", "earlier stability wording removed; single-seed qualifier added"])
    C.append(["Spatial aggregation scale changes rotation pass counts but not the anatomical correspondence metrics", "results_v2/validation/stage16/stage16_results.json", "GlaS testA+testB", "patch", "2083 (rotation analysis uses all 2083 pooled patches)",
              "rotation angles below threshold 4,4,4,4,5,6 (of 9) across six scales; Glass's delta_A -0.004 to 0.041; r -0.034 to 0.015; full-field per-angle errors 0.236-0.641 vs center-pixel 0.825-1.114", "NA", "NA", "NA", "frozen diagnostic (not a criterion)", "NOT DEMONSTRATED",
              "Across six pre-defined scales, rotation pass counts changed from 4, 4, 4, 4, 5, 6 of 9 while the anatomical correspondence metrics remained near null; no scale met a frozen criterion. The frozen rotation thresholds were calibrated for the center-pixel convention.",
              "That a 'correct' or 'optimal' scale exists, or any ranking of scales; that full-field aggregation rescues the result; describing the Stage 16 rotation analysis as a subsample or leaving its n unstated.", "ranking-style and understating wording removed; n = 2083 stated; per-angle errors added"])
    C.append(["The learned S/orientation representation is associated with GlaS pathology grade", "results_v2/validation/stage15/stage15_results.json", "GlaS testA+testB", "image", "80 (43 malignant, 37 benign)",
              "P1 d=-0.135; P2 rho=0.075; P3 d=-0.138; P4 rho=0.003 (all Holm-adjusted p = 1.0)", "see 13_numbers_and_sources.csv", "raw p 0.664, 0.509, 0.524, 0.979",
              "P2 [-0.136, 0.277]; P4 [-0.218, 0.231] (Spearman bootstrap); none for P1/P3 (Cohen's d)", "yes (frozen matrix)", "NOT DEMONSTRATED",
              "No pathology association was demonstrated under the tested analyses.", "That the representation is unrelated to pathology in general; the testB-only P2 result as evidence; a CI for Cohen's d.", "P1/P3 CI removed"])
    C.append(["Patient-level sensitivity of the pathology result", "results_v2/validation/stage15/stage15_results.json", "GlaS testA+testB", "patient", "12",
              "descriptive only; 3 of 12 patients have mixed-grade images", "NA", "NA", "NA", "yes (pre-specified sensitivity check)", "INCONCLUSIVE",
              "Patient-level inference was not possible because of small n and mixed-grade patients; image-level results are subject to patient clustering.", "Any patient-level conclusion in either direction.", "renamed 'sensitivity check'"])
    C.append(["CRAG provides external support for anatomical correspondence", "results_v2/validation/stage17_crag/stage17_results.json", "CRAG", "image", "171 / 40 / 211 (train/test/pooled within CRAG)",
              "pooled Glass's delta_A = -0.080; r = -0.094; D_Q = 0.429; rotation 4/9 (5000-patch subsample)", "-0.080", "r p = 0.175", "r 95% CI [-0.227, 0.049]", "descriptive (frozen Stage 17 matrix)", "NOT DEMONSTRATED",
              "In external cross-dataset validation on CRAG, which is partially independent of GlaS, correspondence was not demonstrated; results were consistent with the GlaS findings.",
              "Independent replication; combining GlaS and CRAG into one estimate; universal failure.", "'partially independent' added to the wording"])
    C.append(["PANDA Radboud masks can supply the frozen gland-instance anatomical reference", "Stage 18 (reports/STAGE18_AUDIT_REPORT.md; summary_table.csv; compatibility_results.json); reports/STAGE21_5_PANDA_DOCUMENTATION_CORRECTION.md", "PANDA Radboud", "case (pilot)", "10",
              "5 pilot cases contain Gleason 4; 4 of them have a benign reference (largest Gleason-4 region / benign median = 49.3, 136.8, 735.1, 1082.3); one also has Gleason 5 (6244.3); 1 Gleason-4 case has no benign reference", "NA", "NA", "NA", "yes (frozen design rule)", "INFEASIBLE",
              "PANDA is an annotation-compatibility/feasibility finding. The Radboud masks are semantic Gleason-pattern regions, not gland-instance annotations; under the frozen definition they do not provide gland-instance geometry, so no reference was built and no Model 3 inference was performed. PANDA is not validation.",
              "That PANDA is a validation; that PANDA is a negative model result; that PANDA disproves the hypothesis; that PANDA is unsuitable for other tasks; any 'x of y cases with substantial content' count.", "count wording replaced with source-supported wording (Correction 1)"])
    C.append(["A negative result here means nematic organization is absent from histopathology or that equivariant networks cannot learn orientation", "none (not supported by any stage)", "N/A", "N/A", "N/A", "N/A", "NA", "NA", "NA", "NA", "NOT SUPPORTED AS A CLAIM",
              "(No wording permitted; the study tested one frozen formulation, reference, and protocol.)", "Any generalization from these results to the biological phenomenon or to equivariant architectures in general.", "none"])
    C.append(["Q_anat is an independently constructed, mask-derived anatomical reference (gland-elongation reference)", "code_v2/anatomy_targets.py; Stage 4 summary", "GlaS", "gland instance", "1530 instances; 1084 valid",
              "covariance orientation and anisotropy of each gland mask; no H&E pixel used; also the training supervision", "NA", "NA", "NA", "engineering definition", "SUPPORTED",
              "Q_anat is an independently constructed, mask-derived anatomical reference: independent of the image pixels used as model input, derived from gland masks, used as training supervision, and describing the elongation orientation of individual glands.",
              "Calling Q_anat a ground truth; universal biological truth; a continuous tissue-fibre orientation field; independence from the training objective.", "new claim (Correction 12)"])
    C.append(["GlaS testA/testB is the canonical image-level held-out test set and is not patient-independent", "data/glas/Warwick_QU_Dataset/Grade.csv (counts, Stage 22); Stage 15 dataset audit", "GlaS", "patient / image", "12 held-out patients; 80 held-out images",
              "11 of 12 held-out patients (79 of 80 held-out images) also contribute images to the training split", "NA", "NA", "NA", "not applicable (data structure)", "SUPPORTED",
              "The held-out images come from the canonical image-level split. 11 of the 12 held-out patients, representing 79 of the 80 held-out images, also contribute images to the training split. This limits independence but cannot explain the negative effect-size results as a positive-bias mechanism.",
              "'patient-independent'; 'independent patient-level test set'; any patient-clustered inference.", "new claim (Correction 6)"])
    C.append(["Patch-level p-values and bootstrap intervals are descriptive", "code/data_prep.py (patch 128, stride 96); Stage 11", "GlaS testA+testB", "patch", "2083 patches from 80 images",
              "overlapping patches; not independent biological units", "NA", "NA", "NA", "not applicable (design property)", "SUPPORTED",
              "The 2083 evaluated patches are overlapping image patches rather than independent biological units. Therefore patch-level p-values and bootstrap intervals should be interpreted descriptively; the frozen effect-size results are the primary basis for interpretation.",
              "Treating patch-level p-values or intervals as inferential evidence; any clustered analysis not performed.", "new claim (Correction 13)"])
    C.append(["The recorded hypotheses H1-H3 are quoted verbatim and do not explicitly assign criteria to hypotheses", "reports/STAGE21_5_H2_TRACEABILITY.md; original logged MASTER PROJECT EXECUTION PROMPT section 4 (session-log line 1219; not a repository file)", "N/A", "N/A", "N/A",
              "H2 block SHA-256 9010f1728f7eb3060805efe47a34f950a6a90b58cf8c9dd14211c149d0a5d220", "NA", "NA", "NA", "not applicable (documentation)", "SUPPORTED",
              "The recorded hypotheses do not explicitly assign individual criteria to individual hypotheses.", "Any assignment of criteria to hypotheses without an explicit cited source; paraphrasing the recorded hypotheses.", "new claim (Corrections 4-5)"])
    return C


# ---------------------------------------------------------- supplementary
ROUND = re.compile(r"-?\d+\.\d{5,}")


def r4(s):
    return ROUND.sub(lambda m: f"{float(m.group(0)):.4f}", s)


def build_s1():
    src = list(csv.reader(open(P("results_v2", "tables", "stage20", "supplementary_full_statistical_evidence.csv"), encoding="utf-8")))
    header, body = src[0], src[1:]
    Q, D_, A, U, N_, ST, CI, PV, FC, OC, IN = range(11)
    out = []
    for i, r in enumerate(body):
        r = list(r)
        line = i + 2
        r[CI] = r4(r[CI])
        r[OC] = "NOT MET" if r[OC] == "FAIL" else r[OC]
        r[Q] = r[Q].replace("Ablation robustness of primary geometric result", "Ablation conditions vs primary geometric result").replace("Patient-level robustness", "Patient-level sensitivity check")
        if line == 2:
            r[PV] = "perm_p=0.002 (secondary, non-binding for Criterion A; patch-level p is descriptive)"
            r[IN] = "Effect size far below threshold; the permutation p is a secondary, non-binding check and is not evidence of meaningful correspondence (overlapping patches; descriptive p)"
        if line == 3:
            r[PV] = "0.1196 (descriptive)"
        if line == 4:
            r[ST] = "Glass's delta=0.0073, mean D_Q=0.5092"
            r[PV] = "perm_p<0.001 (0 of 1000 re-pairings; descriptive)"
            r[IN] = "Permutation condition met but Glass's delta negligible (0.0073); statistical significance without practical effect"
        if line == 5:
            r[U] = "patch (pooled testA+testB)"
            r[IN] = "4/9 below threshold (90, 123, 150, 173 deg); empirical center-pixel, pixel-domain rotation consistency did not meet the frozen criterion; measurement scope in file 05; S-invariance and phi-shift are non-gating checks reported separately"
        if 8 <= line <= 12:
            r[IN] = "No condition approaches the Criterion A/B/C thresholds (one seed per condition; context only)"
        if line == 13:
            r[U] = "patch (pooled testA+testB)"
            r[IN] = "Neither alternative reduction reaches the required 7/9 (thresholds calibrated for the center-pixel convention; diagnostic only)"
        if line in (14, 16):
            r[ST] = "Cohen's d=" + r[ST].strip()
            r[CI] = "NA (no CI computed for Cohen's d)"
        if line in (15, 17):
            r[ST] = "Spearman rho=" + r[ST].strip()
        if line == 18:
            r[IN] = "Split-specific, uncorrected, non-primary; n=20 with only 4 benign images; contradicted by the pooled analysis; not treated as evidentiary"
        if 23 <= line <= 34:
            r[U] = "patch (pooled testA+testB)"
            if r[N_] == "NA":
                r[N_] = "2083"
            if "rotation" in r[Q]:
                r[IN] = "Pass counts across the six scales are 4, 4, 4, 4, 5, 6 of 9 (thresholds calibrated for the center-pixel convention; diagnostic only); no scale reaches 7/9"
        if 35 <= line <= 39:
            r[Q] = r[Q] + " (CRAG, partially independent of GlaS)"
        if line == 38:
            r[A] = "Stage 17, patch-level secondary, 5000-patch subsample of pooled CRAG patches"
        if line == 39:
            r[U] = "patch (5000-patch subsample)"
        if line == 40:
            r[Q] = "Annotation-compatibility / feasibility (PANDA Radboud pilot; not a validation)"
            r[A] = "Stage 18 compatibility gate (no model inference)"
            r[ST] = "5 of 10 pilot cases contain Gleason 4; 4 of these have a same-case benign reference (largest Gleason-4 region / benign median = 49.3, 136.8, 735.1, 1082.3); one also has Gleason 5 (6244.3); 1 Gleason-4 case has no benign reference"
            r[FC] = "gland-instance geometry must be recoverable without inventing structure"
            r[OC] = "INFEASIBLE"
            r[IN] = "Radboud masks are semantic Gleason-pattern regions, not gland-instance annotations; Q_anat not generated; no Model 3 inference; not a validation"
        out.append(r)
    return header, out


def build_s2():
    src = list(csv.reader(open(P("results_v2", "tables", "stage20", "supplementary_evidence_domain_summary.csv"), encoding="utf-8")))
    new = {
        "Primary anatomical correspondence (GlaS)": ("Criteria A, B, C not met; effect sizes negligible (Glass's delta 0.0412, r -0.0341, Glass's delta 0.0073) although patch-level permutation p-values for A and C are small",
                                                     "NOT SUPPORTED", "Glass's delta << 0.5 and r ~ 0 at every criterion, pooled testA+testB (n=2083 overlapping patches; patch-level p-values are descriptive)"),
        "Empirical rotation consistency (GlaS)": ("4/9 non-zero angles below the frozen threshold; the four are the angles with the largest thresholds; empirical center-pixel, pixel-domain measurement",
                                                  "NOT SUPPORTED", "Required >=7/9; alternative reductions (Stage 14 A6, Stage 16) are diagnostic and also do not reach 7/9; non-gating S-invariance and phi-shift checks reported separately"),
        "Ablation robustness": ("No tested architectural/loss-weighting condition (A1, A3, A4, A5) meets the frozen anatomical criteria",
                                "NOT SUPPORTED", "All 5 conditions (A0, A1, A3, A4, A5) show near-null Criterion A/B/C metrics under the fixed single-seed configuration (one training run per condition)"),
        "Multi-scale sensitivity": ("Rotation pass counts 4, 4, 4, 4, 5, 6 of 9 across six scales; Glass's delta_A -0.004 to 0.041 and r -0.034 to 0.015 (near null); no scale meets either frozen criterion",
                                    "NOT SUPPORTED", "The full-field scale also fails Criterion A/B and the 7/9 rotation requirement; the rotation thresholds were calibrated for the center-pixel convention, so pass counts under other reductions are diagnostic only"),
        "External cross-dataset validation (CRAG)": ("Weak-to-null correspondence at every population/unit; Glass's delta negative at every population; rotation consistency 4/9 with the same passing angles as GlaS",
                                                     "NOT SUPPORTED", "Consistent with the GlaS finding; CRAG is partially independent of GlaS (same hospital and curating group), so it is external but not independent replication"),
        "Cross-organ external validation (PANDA)": ("Radboud masks are semantic Gleason-pattern regions, not gland-instance annotations; the largest Gleason-4 regions were 49.3-1082.3 times the same-case benign median in the four pilot cases with a benign reference (Gleason 5: 6244.3)",
                                                    "INFEASIBLE", "Annotation-compatibility/feasibility finding: no anatomical reference generated; no Model 3 inference; not a validation and not a model result"),
    }
    out = []
    for r in src[1:]:
        r = list(r)
        if r[0] in new:
            f, s, why = new[r[0]]
            if r[0] == "Ablation robustness":
                r[0] = "Ablation conditions"
            if r[0] == "External cross-dataset validation (CRAG)":
                r[0] = "External cross-dataset validation, partially independent (CRAG)"
            if r[0] == "Cross-organ external validation (PANDA)":
                r[0] = "Annotation-compatibility / feasibility (PANDA; not a validation)"
            r[1], r[2], r[3] = f, s, why
        elif r[0] == "Pathology association (GlaS)":
            r[1] = "All 4 pre-specified, Holm-corrected pooled tests (P1-P4) null; morphology (M1-M2) and confound-adjustment (C1) null; one non-primary split-specific result disclosed but not treated as evidentiary"
        out.append(r)
    return src[0], out


# ------------------------------------------------------------- finalize
FROZEN_CHECK = [
    ("results_v2/model3/checkpoints/checkpoint_final.pt", "5495ddeba84d3169c2a6b63ad22fbdcb"),
    ("results_v2/phase2/preregistered_thresholds_v2.json", "4d68f3877d2cc0ec50aac19629182740"),
    ("configs/stage14_ablation_matrix.json", "ba089e9e3f39c7e36575ab45b695102c"),
    ("configs/stage15_pathology_analysis_matrix.json", "f54def3645272776098183533400483c"),
    ("configs/stage16_multiscale_analysis_matrix.json", "db3317d52cb0400b5c61fa2c63bd878d"),
    ("configs/stage17_crag_validation_matrix.json", "b9eb1139b19d3d7ef535253f207b0e10"),
    ("reports/STAGE18_DESIGN_FREEZE.md", "8ebfc2afd3371d6bf19ee903ba1961e9"),
    ("results_v2/statistical_synthesis/MASTER_STATISTICAL_TABLE.csv", "13993da45721d3b15fc5ea6e8ce2ae84"),
    ("results_v2/statistical_synthesis/MASTER_EVIDENCE_TABLE.csv", "85ec33922c3afb03528f1d5948030da6"),
    ("results_v2/tables/stage20/supplementary_full_statistical_evidence.csv", "13993da45721d3b15fc5ea6e8ce2ae84"),
    ("results_v2/tables/stage20/supplementary_evidence_domain_summary.csv", "85ec33922c3afb03528f1d5948030da6"),
    ("results_v2/validation/stage11/stage11_results.json", "74918811257c8351ddbe3d28978ba203"),
    ("results_v2/validation/stage12/stage12_results.json", "f43a186332149abbda83420623da1a9c"),
    ("results_v2/validation/stage13/stage13_results.json", "fd2d2514d214c0e1eceb7747de5ba44a"),
    ("results_v2/validation/stage14/stage14_all_evaluations.json", "29cbdb82c83b71257955c2594dac8dc2"),
    ("results_v2/validation/stage14/A6_spatial_reduction/a6_results.json", "3bb0d214fb7bc119d66b2cfaca7fc867"),
    ("results_v2/validation/stage15/stage15_results.json", "1c3d2bbb53557649016bc477abfdeff1"),
    ("results_v2/validation/stage16/stage16_results.json", "3c87d6ced4304b3d4d0a350441018bfa"),
    ("results_v2/validation/stage17_crag/stage17_results.json", "44913f300fdc941a16437a3a5404103f"),
    ("results_v2/validation/stage18_panda/compatibility_results.json", "bd1a6895a9ad4df3822c86678454418d"),
    ("results_v2/validation/stage18_panda/summary_table.csv", "e8e19c6d01f0d58843ce7bc5507376f4"),
    ("code_v2/stage20_figures_and_tables.py", "6a4a0be4b1ed720c00100037dd64da05"),
    ("code_v2/stage21_numbers_and_claims.py", "e61e35b3814171695c01bf2ce049fa0d"),
    ("results_v2/manuscript_stage21/01_primary_question.md", "b676e983a75cb0eb222fcb542eda7c6a"),
    ("results_v2/manuscript_stage21/02_methods_evidence_map.md", "a87f9221886c683317932bf2ce0f4c0b"),
    ("results_v2/manuscript_stage21/03_results_narrative.md", "1517fcdec04ac32846f6537f1e05cf74"),
    ("results_v2/manuscript_stage21/04_primary_geometric_validation.md", "e279cb2f99b71fdbe2d39196e54918f2"),
    ("results_v2/manuscript_stage21/05_rotation_results.md", "794c78500506606d786744fcb6eda4c7"),
    ("results_v2/manuscript_stage21/06_ablation_multiscale.md", "4c1bf2e03e1e478e968fac96094da700"),
    ("results_v2/manuscript_stage21/07_pathology_results.md", "23f5475897b4aecbf0bcf0855f6af367"),
    ("results_v2/manuscript_stage21/08_crag_results.md", "d118ba107f8f62de1d67050a70d52b56"),
    ("results_v2/manuscript_stage21/09_panda_compatibility.md", "d1c4dfcd9b64518a1ef6616c3ed2e101"),
    ("results_v2/manuscript_stage21/10_limitations.md", "a1b2b6b746c192f1bf4730c74e337c4c"),
    ("results_v2/manuscript_stage21/11_scientific_interpretation.md", "b70cf16e344f3dffa7b11c9f4356abad"),
    ("results_v2/manuscript_stage21/12_claim_evidence_matrix.csv", "37dbcfbc002c72ca0354b530264e0028"),
    ("results_v2/manuscript_stage21/13_numbers_and_sources.csv", "872084e8a1dc6e84bc1d83216ebd3cfc"),
    ("results_v2/manuscript_stage21/14_manuscript_figure_table_map.md", "1d278a0b2f94206ce1aeac2ae693ca38"),
    ("results_v2/manuscript_stage21/15_stage21_audit.md", "e07ecfb29fa9e04734e6f851cda32cd3"),
]

MD_FILES = ["01_primary_question.md", "02_methods_evidence_map.md", "03_results_narrative.md", "04_primary_geometric_validation.md",
            "05_rotation_results.md", "06_ablation_multiscale.md", "07_pathology_results.md", "08_crag_results.md", "09_panda_compatibility.md",
            "10_limitations.md", "11_scientific_interpretation.md", "14_manuscript_figure_table_map.md"]


def frozen_status():
    bad = []
    for rel, want in FROZEN_CHECK:
        got = md5(P(*rel.split("/")))
        if got != want:
            bad.append((rel, got, want))
    return bad


def finalize():
    outputs = {
        "manuscript_stage21_corrected/" + f: P("results_v2", "manuscript_stage21_corrected", f) for f in MD_FILES + ["12_claim_evidence_matrix.csv", "13_numbers_and_sources.csv"]
    }
    outputs["tables/stage20_corrected/supplementary_full_statistical_evidence_corrected.csv"] = os.path.join(TAB, "supplementary_full_statistical_evidence_corrected.csv")
    outputs["tables/stage20_corrected/supplementary_evidence_domain_summary_corrected.csv"] = os.path.join(TAB, "supplementary_evidence_domain_summary_corrected.csv")
    # 1) checksum block in file 15 (covers everything except file 15, provenance CSV)
    lines = ["| Artifact | MD5 |", "|---|---|"]
    for k, v in outputs.items():
        lines.append(f"| `results_v2/{k}` | `{md5(v)}` |")
    lines.append(f"| `{SCRIPT_REL}` | `{md5(P('code_v2', 'stage225_build_corrected_package.py'))}` |")
    block = "<!-- CHECKSUMS:BEGIN -->\n" + "\n".join(lines) + "\n<!-- CHECKSUMS:END -->"
    f15 = os.path.join(OUT, "15_STAGE21_CORRECTED_AUDIT.md")
    txt = open(f15, encoding="utf-8").read()
    txt = re.sub(r"<!-- CHECKSUMS:BEGIN -->.*?<!-- CHECKSUMS:END -->", lambda m: block, txt, flags=re.S)
    open(f15, "w", encoding="utf-8").write(txt)
    outputs["manuscript_stage21_corrected/15_STAGE21_CORRECTED_AUDIT.md"] = f15
    # 2) provenance CSV
    prov_header = ["artifact", "source_file", "source_field", "dataset", "unit", "transformation", "script", "output_md5"]
    prov = []

    def add(art, srcs, fld, ds, unit, tr, script=SCRIPT_REL):
        key = art
        h = md5(outputs[key]) if key in outputs else ""
        prov.append([f"results_v2/{art}", srcs, fld, ds, unit, tr, script, h])
    m = "manuscript_stage21_corrected/"
    add(m + "01_primary_question.md", "reports/STAGE21_5_H2_TRACEABILITY.md; results_v2/phase2/preregistered_thresholds_v2.json; results_v2/manuscript_stage21/01_primary_question.md", "hypothesis text; criteria; decision rule", "GlaS", "N/A",
        "hand-written correction of the Stage 21 file (verbatim H1-H3; criteria separated from hypotheses; internal-freeze history)", "hand-written (Stage 22.5)")
    add(m + "02_methods_evidence_map.md", "results_v2/manuscript_stage21/02_methods_evidence_map.md; Stage 22 reports", "component map", "GlaS/CRAG/PANDA", "N/A", "hand-written correction (terminology, patient structure, single-seed status)", "hand-written (Stage 22.5)")
    for f, srcs in (("03_results_narrative.md", "13_numbers_and_sources.csv (all values)"), ("04_primary_geometric_validation.md", "results_v2/validation/stage11/stage11_results.json"),
                    ("05_rotation_results.md", "results_v2/validation/stage12/stage12_results.json; stage16_results.json"), ("06_ablation_multiscale.md", "stage14_all_evaluations.json; a6_results.json; stage16_results.json"),
                    ("07_pathology_results.md", "results_v2/validation/stage15/stage15_results.json"), ("08_crag_results.md", "results_v2/validation/stage17_crag/*.json"),
                    ("09_panda_compatibility.md", "results_v2/validation/stage18_panda/summary_table.csv; reports/STAGE21_5_PANDA_DOCUMENTATION_CORRECTION.md"),
                    ("10_limitations.md", "frozen audit reports; Stage 22 audit (patient overlap counts from Grade.csv)"), ("11_scientific_interpretation.md", "corrected files 03-10, 12"),
                    ("14_manuscript_figure_table_map.md", "results_v2/figures/stage20/; results_v2/tables/stage20/; reports/STAGE20_PROVENANCE.md; Stage 22 traceability check")):
        add(m + f, srcs, "values as listed in 13_numbers_and_sources.csv", "GlaS/CRAG/PANDA", "as stated in file", "hand-written correction of the Stage 21 file; every number appears in 13_numbers_and_sources.csv", "hand-written (Stage 22.5)")
    add(m + "12_claim_evidence_matrix.csv", "results_v2/manuscript_stage21/12_claim_evidence_matrix.csv; frozen result JSONs", "claims", "GlaS/CRAG/PANDA", "as stated per row", "regenerated: 14 Stage 21 claims revised, 5 added (Stage 22.5)")
    add(m + "13_numbers_and_sources.csv", "results_v2/manuscript_stage21/13_numbers_and_sources.csv; frozen result JSONs; Grade.csv (counts)", "see source_field column", "GlaS/CRAG/PANDA", "as stated per row",
        "183 Stage 21 rows carried over and re-verified against source; new rows added; wording of text columns corrected")
    add("tables/stage20_corrected/supplementary_full_statistical_evidence_corrected.csv", "results_v2/tables/stage20/supplementary_full_statistical_evidence.csv (historical, unchanged)", "39 rows", "GlaS/CRAG/PANDA", "as stated per row",
        "row-wise corrections: PANDA row; P1/P3 CI removed; Stage 16 rotation n and unit; C p < 0.001; A p labelled secondary; CI rounding to 4 decimals; FAIL -> NOT MET; stability and informal improvement wording removed. No value recomputed.")
    add("tables/stage20_corrected/supplementary_evidence_domain_summary_corrected.csv", "results_v2/tables/stage20/supplementary_evidence_domain_summary.csv (historical, unchanged)", "8 rows", "GlaS/CRAG/PANDA", "as stated per row",
        "wording corrections (stability wording, ranking-style wording, informal improvement wording, PANDA label); no value changed")
    add("manuscript_stage21_corrected/15_STAGE21_CORRECTED_AUDIT.md", "all corrected artifacts; frozen checksums", "audit and checksums", "N/A", "N/A", "hand-written audit with script-generated checksum block", "hand-written + " + SCRIPT_REL)
    # inherit Stage 20 figure/table provenance rows (unchanged historical artifacts) for completeness
    for r in csv.reader(open(P("results_v2", "tables", "stage20", "supplementary_provenance.csv"), encoding="utf-8")):
        if r[0] == "artifact":
            continue
        prov.append([f"{r[0]} (historical Stage 20 artifact, unchanged; not regenerated)", r[1], r[2], r[3], r[4], r[5], r[6], r[7]])
    write_csv(os.path.join(TAB, "supplementary_provenance_corrected.csv"), prov_header, prov)
    print("finalized; provenance rows:", len(prov))


# ------------------------------------------------------------------- main
def build():
    rows13 = build_numbers()
    write_csv(os.path.join(OUT, "13_numbers_and_sources.csv"), HEADER13, [[r[k] for k in HEADER13] for r in rows13])
    write_csv(os.path.join(OUT, "12_claim_evidence_matrix.csv"), HEADER12, build_claims())
    h, b = build_s1()
    write_csv(os.path.join(TAB, "supplementary_full_statistical_evidence_corrected.csv"), h, b)
    h, b = build_s2()
    write_csv(os.path.join(TAB, "supplementary_evidence_domain_summary_corrected.csv"), h, b)
    print("numbers rows:", len(rows13), "| claims:", len(build_claims()), "| S1 rows:", len(build_s1()[1]), "| S2 rows:", len(build_s2()[1]))


if __name__ == "__main__":
    build()
    if "--finalize" in sys.argv:
        finalize()
    bad = frozen_status()
    print("frozen/original artifact checksum mismatches:", bad if bad else "none")
