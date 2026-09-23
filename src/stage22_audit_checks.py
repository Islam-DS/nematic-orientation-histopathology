"""Stage 22 — read-only number-consistency audit.

Reads frozen artifacts only. Performs NO new statistical analysis, model run,
resampling, or estimation: it only (a) resolves every row of the Stage 21
numbers table back to its named source field and compares the stored value,
and (b) checks that every numeric token displayed in the Stage 19 master
statistical table, the Stage 20 tables, and the Stage 21 narrative files
appears (at the displayed precision) among the values in the frozen source
files. Writes reports/STAGE22_NUMBER_CONSISTENCY.csv.
"""
import csv
import glob
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "STAGE22_NUMBER_CONSISTENCY.csv")
NUM_TABLE = os.path.join(ROOT, "results_v2", "manuscript_stage21", "13_numbers_and_sources.csv")


def p(*a):
    return os.path.join(ROOT, *a)


# ---------------------------------------------------------------- resolver
def resolve(obj, path):
    cur = obj
    for tok in path.split("."):
        m = re.fullmatch(r"(\w+)\[(\w+)\]", tok)
        if m:
            cur = cur[m.group(1)]
            key = m.group(2)
            hit = [e for e in cur if isinstance(e, dict) and e.get("id") == key]
            if not hit:
                raise KeyError(tok)
            cur = hit[0]
            continue
        if isinstance(cur, list):
            cur = cur[int(tok)]
        elif isinstance(cur, dict):
            if tok in cur:
                cur = cur[tok]
            elif tok.isdigit() and int(tok) in cur:
                cur = cur[int(tok)]
            else:
                raise KeyError(tok)
        else:
            raise KeyError(tok)
    return cur


def to_float(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"-?\d+(?:\.\d+)?", x)
        return float(m.group(0)) if m else None
    return None


# text-source rows: (file, regex that must be present, note)
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

rows = list(csv.DictReader(open(NUM_TABLE, encoding="utf-8")))
summary_rows = list(csv.DictReader(open(p("results_v2", "validation", "stage18_panda", "summary_table.csv"), encoding="utf-8")))

json_cache = {}


def load_json(rel):
    if rel not in json_cache:
        json_cache[rel] = json.load(open(p(*rel.split("/")), encoding="utf-8"))
    return json_cache[rel]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


out = []
disc = 0
for i, r in enumerate(rows, start=2):
    num = float(r["number"])
    sf, fld = r["source_file"], r["source_field"]
    status, resolved, note = "", "", ""
    if sf.endswith(".json"):
        path = fld.split(" ")[0]
        try:
            v = resolve(load_json(sf), path)
            resolved = repr(v)
            fv = to_float(v)
            if "D_rotation_consistency" in path and isinstance(v, str):
                m7 = re.search(r"at least (\d+) of the (\d+) non-zero", v)
                status = "MATCH" if m7 and int(m7.group(1)) == int(num) else "DISCREPANCY"
                note = "required-angle count parsed from rule text ('at least 7 of the 9 non-zero')"
            elif "gt30pct" in fld and int(num) == 30 and "dominant" in path:
                status, note = "MATCH_BY_FIELD_NAME", "threshold 30 (percent) is encoded in the field name; field value is 70 images"
            elif fv is None:
                status, note = "UNRESOLVED_NONNUMERIC", "resolved value is not numeric"
            else:
                dec = len(r["number"].split(".")[1]) if "." in r["number"] else 0
                if round(fv, dec) == round(num, dec) or math.isclose(fv, num, rel_tol=0, abs_tol=0.5 * 10 ** (-dec) + 1e-12):
                    status = "MATCH"
                else:
                    status = "DISCREPANCY"
                    note = f"stored {r['number']} vs source {fv}"
        except Exception as e:  # noqa: BLE001
            status, note = "RESOLVER_ERROR", f"{type(e).__name__}: {e}"
    elif sf.endswith("summary_table.csv"):
        lab, fld_l = r["label"], fld
        ratio_cols = {"gleason_4_largest_vs_benign_median_ratio": "gleason_4_largest_vs_benign_median_ratio"}
        vals4 = [fnum(x["gleason_4_largest_vs_benign_median_ratio"]) for x in summary_rows]
        vals4 = [v for v in vals4 if v is not None]
        vals5 = [fnum(x["gleason_5_largest_vs_benign_median_ratio"]) for x in summary_rows]
        vals5 = [v for v in vals5 if v is not None]
        ncomp = [fnum(x["benign_epithelium_n_components"]) for x in summary_rows]
        ncomp_pos = [v for v in ncomp if v and v > 0]
        g4frac = [fnum(x["gleason_4_pixel_fraction"]) for x in summary_rows]
        if "smallest Gleason-4" in lab:
            ok = min(vals4) == num; resolved = str(min(vals4))
        elif "largest Gleason-5" in lab:
            ok = max(vals5) == num; resolved = str(max(vals5))
        elif "with Gleason 4 and a computable" in lab:
            ok = len(vals4) == num; resolved = str(len(vals4))
        elif "containing Gleason 4" in lab:
            ok = sum(1 for v in g4frac if v and v > 0) == num; resolved = str(sum(1 for v in g4frac if v and v > 0))
        elif "PANDA Gleason-4 ratio" in lab:
            case = re.search(r"case (\w+)", lab).group(1)
            hit = [x for x in summary_rows if x["image_id"].startswith(case)]
            resolved = hit[0]["gleason_4_largest_vs_benign_median_ratio"] if hit else "NOCASE"
            ok = bool(hit) and fnum(resolved) == num
        elif "minimum" in lab:
            ok = min(ncomp_pos) == num; resolved = str(min(ncomp_pos))
        elif "maximum" in lab:
            ok = max(ncomp_pos) == num; resolved = str(max(ncomp_pos))
        else:
            ok, resolved = False, "unhandled"
        status = "MATCH" if ok else "DISCREPANCY"
    else:
        key = f"{os.path.basename(sf)}|{r['number']}"
        chk = TEXT_CHECKS.get(key)
        if chk:
            txt = open(p(*chk[0].split("/")), encoding="utf-8").read()
            hit = re.search(chk[1], txt, re.I) is not None
            status = "TEXT_SOURCE_PRESENT" if hit else "TEXT_SOURCE_NOT_FOUND"
            resolved = f"regex {chk[1]!r}"
        else:
            status, note = "NO_CHECK_DEFINED", "text source without a defined check"
    if status in ("DISCREPANCY", "RESOLVER_ERROR", "UNRESOLVED_NONNUMERIC", "TEXT_SOURCE_NOT_FOUND", "NO_CHECK_DEFINED"):
        disc += 1
    out.append(
        {
            "check_type": "numbers_table_row",
            "location": f"13_numbers_and_sources.csv row {i}",
            "item": f"{r['label']} = {r['number']}",
            "reference": f"{sf} :: {fld}",
            "resolved_source_value": resolved,
            "result": status,
            "note": note,
        }
    )

# ------------------------------------------------- downstream token checks
leaves = []


def collect(o):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        if not (isinstance(o, float) and (math.isnan(o) or math.isinf(o))):
            leaves.append(float(o))
    elif isinstance(o, dict):
        for v in o.values():
            collect(v)
    elif isinstance(o, list):
        for v in o:
            collect(v)


json_files = []
for pat in ("results_v2/validation/**/*.json", "results_v2/phase2/*.json", "results_v2/anatomy_targets/*.json",
            "results_v2/model3/metrics/*.json", "configs/*.json"):
    json_files += glob.glob(p(*pat.split("/")), recursive=True)
for jf in json_files:
    if os.path.getsize(jf) > 30_000_000:
        continue
    try:
        collect(json.load(open(jf, encoding="utf-8")))
    except Exception:  # noqa: BLE001
        pass
for cf in glob.glob(p("results_v2", "validation", "**", "*.csv"), recursive=True) + [NUM_TABLE]:
    if os.path.getsize(cf) > 30_000_000:
        continue
    try:
        for rr in csv.reader(open(cf, encoding="utf-8")):
            for c in rr:
                x = fnum(c)
                if x is not None:
                    leaves.append(x)
    except Exception:  # noqa: BLE001
        pass
leaves = sorted(set(leaves))
import bisect  # noqa: E402


def in_leaves(t):
    """token t (string) matches a frozen value at displayed precision (also abs / percent / 1-p forms)."""
    m = re.fullmatch(r"-?(\d+)(?:\.(\d+))?(?:e(-?\d+))?", t)
    if not m:
        return False
    x = float(t)
    dec = len(m.group(2)) if m.group(2) else 0
    tol = 0.5 * 10 ** (-dec) + 1e-12
    for cand, scale in ((x, 1), (-x, 1), (x / 100.0, 100), (-x / 100.0, 100)):
        lo, hi = cand - tol / scale, cand + tol / scale
        j = bisect.bisect_left(leaves, lo)
        if j < len(leaves) and leaves[j] <= hi:
            return True
    return False


STRIP = [
    r"Stage\s*\d+(?:\.\d+)?(?:[–-]\d+)?", r"Stages\s*\d+(?:[–,\s-]+\d+)*", r"Figure\s*\d+", r"Table\s*\d+", r"Section\s*\d+", r"§\s*\d+(?:\.\d+)?",
    r"\b[A-Za-z]+\d+[A-Za-z0-9]*\b", r"\b\d+[A-Za-z]+\d*\b(?<!\d%)", r"MD5\s*`?[0-9a-f]{32}`?", r"`[0-9a-f]{16,}`", r"\bfile\s*\d+", r"\bFiles?\s*\d+(?:[–,\s-]+\d+)*",
    r"\bNN?\s*=\s*\d+\b(?=\s*\?)", r"\d{4}-\d{2}-\d{2}", r"\bStage\s*\d+",
]
ALLOWED_INT = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "15", "30", "37", "45", "60", "90", "123", "150", "173", "95", "100", "10", "12", "16"}


def tokens_of(text):
    t = text
    for s in STRIP:
        t = re.sub(s, " ", t)
    return re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])", t.replace(",", "") if False else t)


def scan(path_rel, text, label, per_line=True):
    n_tok = n_bad = 0
    for ln, line in enumerate(text.splitlines(), start=1):
        for tk in tokens_of(line):
            if tk in ALLOWED_INT:
                continue
            n_tok += 1
            if not in_leaves(tk):
                # thousands separators e.g. 2,083 are handled below
                n_bad += 1
                out.append({"check_type": "downstream_token", "location": f"{path_rel}:{ln}", "item": tk,
                            "reference": "frozen JSON/CSV numeric leaves (displayed precision)", "resolved_source_value": "",
                            "result": "TOKEN_NOT_FOUND_IN_SOURCES", "note": line.strip()[:160]})
    return n_tok, n_bad


def scan_csv(path_rel, cols=None):
    txt = open(p(*path_rel.split("/")), encoding="utf-8").read()
    # strip thousands separators only for a lone "d,ddd" group (not for lists such as "90,123,150,173")
    txt = re.sub(r"(?<![\d,])(\d{1,3}),(\d{3})(?![\d,])", r"\1\2", txt)
    return scan(path_rel, txt, path_rel)


targets = ["results_v2/statistical_synthesis/MASTER_STATISTICAL_TABLE.csv"] + [
    x.replace(ROOT + os.sep, "").replace(os.sep, "/") for x in sorted(glob.glob(p("results_v2", "tables", "stage20", "table*.csv")))
] + [
    "results_v2/manuscript_stage21/" + os.path.basename(x) for x in sorted(glob.glob(p("results_v2", "manuscript_stage21", "0[1-9]_*.md")) + glob.glob(p("results_v2", "manuscript_stage21", "1[01]_*.md")) + glob.glob(p("results_v2", "manuscript_stage21", "14_*.md")))
] + ["results_v2/manuscript_stage21/12_claim_evidence_matrix.csv", "results_v2/statistical_synthesis/STAGE19_STATISTICAL_SYNTHESIS.md"]
tally = {}
for t in targets:
    tally[t] = scan_csv(t)

# ------------------------------------------------ Stage 19 vs Stage 20 tables
import filecmp  # noqa: E402
same = filecmp.cmp(p("results_v2", "statistical_synthesis", "MASTER_STATISTICAL_TABLE.csv"),
                   p("results_v2", "tables", "stage20", "supplementary_full_statistical_evidence.csv"), shallow=False)
out.append({"check_type": "cross_file", "location": "Stage 19 MASTER_STATISTICAL_TABLE.csv vs Stage 20 supplementary_full_statistical_evidence.csv",
            "item": "byte identity", "reference": "", "resolved_source_value": "", "result": "MATCH" if same else "DISCREPANCY", "note": ""})
same2 = filecmp.cmp(p("results_v2", "statistical_synthesis", "MASTER_EVIDENCE_TABLE.csv"),
                    p("results_v2", "tables", "stage20", "supplementary_evidence_domain_summary.csv"), shallow=False)
out.append({"check_type": "cross_file", "location": "Stage 19 MASTER_EVIDENCE_TABLE.csv vs Stage 20 supplementary_evidence_domain_summary.csv",
            "item": "byte identity", "reference": "", "resolved_source_value": "", "result": "MATCH" if same2 else "DISCREPANCY", "note": ""})

# Stage 20 Table 3 vs Stage 12 JSON
t3 = list(csv.DictReader(open(p("results_v2", "tables", "stage20", "table3_rotation_consistency.csv"), encoding="utf-8")))
s12 = load_json("results_v2/validation/stage12/stage12_results.json")["per_angle"]
for a, b in zip(t3, s12):
    ok = fnum(a["model_error_mean"]) is not None and abs(fnum(a["model_error_mean"]) - b["model_error_mean"]) < 5e-5
    thr = b.get("frozen_local_threshold")
    ok2 = (a["frozen_threshold"] in ("N/A", "") and thr is None) or (thr is not None and fnum(a["frozen_threshold"]) is not None and abs(fnum(a["frozen_threshold"]) - thr) < 5e-5)
    out.append({"check_type": "cross_file", "location": f"table3_rotation_consistency.csv angle {a['angle_deg']}",
                "item": f"error {a['model_error_mean']} / threshold {a['frozen_threshold']}", "reference": "stage12_results.json per_angle",
                "resolved_source_value": f"{b['model_error_mean']} / {thr}", "result": "MATCH" if ok and ok2 else "DISCREPANCY", "note": ""})

# ------------------------------------------- semantic discrepancies (recorded)
# These were identified by reading the frozen artifacts against their sources
# (values themselves match; the *statement*, label, scale or precision does not).
# Each is a documentation-level finding; no frozen file was changed.
SEMANTIC = [
    ("results_v2/statistical_synthesis/MASTER_STATISTICAL_TABLE.csv:40 (byte-identical copy: results_v2/tables/stage20/supplementary_full_statistical_evidence.csv:40)",
     "PANDA: '3/3 cases with substantial Gleason 4/5 content ... 49x-6244x'",
     "results_v2/validation/stage18_panda/summary_table.csv",
     "5 cases have Gleason-4 pixels; 4 have a computable benign reference (ratios 49.3, 136.8, 735.1, 1082.3); 1 has Gleason 5 (6244.3). The interval 49x-6244x spans four distinct cases, and 'substantial' is not defined in any frozen file.",
     "NOT REPRODUCIBLE FROM FROZEN DEFINITIONS"),
    ("reports/STAGE18_AUDIT_REPORT.md:104-107; results_v2/validation/stage18_panda/compatibility_report.md:43-45; results_v2/statistical_synthesis/STAGE19_STATISTICAL_SYNTHESIS.md:154-157",
     "PANDA '3 of 3' wording (same statement in three further frozen files)", "results_v2/validation/stage18_panda/summary_table.csv",
     "Same as above.", "NOT REPRODUCIBLE FROM FROZEN DEFINITIONS"),
    ("reports/STAGE18_AUDIT_REPORT.md:108 (also compatibility_report.md:48)",
     "PANDA Gleason 3 'similar pattern in 2 of 3 cases with substantial content (96x-112x)'", "results_v2/validation/stage18_panda/summary_table.csv",
     "Gleason-3 ratios are 28.7, 0.2 (640-px focus), 96.1, 112.0; a case with Gleason 3 but no benign reference (006f6aa3) also exists. Reproducible only under an undefined pixel-fraction threshold for 'substantial'.",
     "NOT REPRODUCIBLE FROM FROZEN DEFINITIONS"),
    ("MASTER_STATISTICAL_TABLE.csv rows 14 and 16 (P1, P3); same rows in Stage 20 supplementary S1",
     "Statistic = Cohen's d (-0.1348; -0.1378) paired with 95% CI [-0.0078, 0.0040] and [-0.1261, 0.0640]",
     "results_v2/validation/stage15/stage15_results.json primary_pooled.P1/P3 mean_diff_95CI_bootstrap",
     "The CI is a bootstrap interval for the raw between-group mean difference (units of S_DL / phi-dispersion), not for Cohen's d. Read as a CI on the listed statistic it is on a different scale.",
     "SCALE MISMATCH (documentation)"),
    ("MASTER_STATISTICAL_TABLE.csv rows 24, 26, 28, 30, 32, 34 (Stage 16 rotation rows)",
     "n = NA; unit 'patch (pooled subsample)'", "results_v2/validation/stage16/stage16_results.json rotation_by_scale.*.per_angle[*].n_patches",
     "Stage 16 rotation used n_patches = 2083 (the full pooled held-out population); no subsample. (The 5000-patch subsample belongs to CRAG, Stage 17.)",
     "LABEL/UNIT DISCREPANCY (documentation)"),
    ("MASTER_STATISTICAL_TABLE.csv row 4; table2_primary_criteria.csv row C; 04_primary_geometric_validation.md:13; 12_claim_evidence_matrix.csv row 6; figure3 title",
     "Criterion C permutation p reported as 0.0 / 0.000 / 'perm p=0'", "code_v2/stage11_geometric_validation.py N_PERM = 1000; stage11_results.json C_tensor_similarity.permutation_p_value = 0.0",
     "With 1000 re-pairings a count of zero supports 'p < 0.001' (below 1/1001), not p = 0.",
     "PRECISION/LABEL (documentation)"),
    ("MASTER_STATISTICAL_TABLE.csv '95% CI' column (rows 3, 14-21, 35-38) and supplementary S1 copy",
     "CI endpoints printed with 15-17 significant digits", "same source JSONs",
     "Values agree with sources; precision inconsistent with the 4-decimal Statistic column.", "ROUNDING (cosmetic)"),
    ("reports/STAGE15_AUDIT_REPORT.md:91-93",
     "testB P2 raw p 0.005 'would not survive any reasonable multiple-comparison adjustment across even just these 8'",
     "results_v2/validation/stage15/stage15_results.json primary_by_split (8 split-specific raw p-values)",
     "Holm across those 8 gives 0.0407 for testB.P2 (arithmetic on frozen p-values): below 0.05, not below 0.01. The other stated reasons (n=20 with 4 benign; contradicted by pooled n=80) are unaffected. Not repeated in the Stage 21 package.",
     "ARITHMETICALLY UNSUPPORTED SENTENCE (frozen report)"),
    ("results_v2/tables/stage20/table1_datasets_and_design.csv row 3; MASTER_EVIDENCE_TABLE.csv row 9 label",
     "PANDA role 'attempted cross-organ external validation'; evidence-row label 'Cross-organ external validation (PANDA)'", "Stage 21 file 09 / claim matrix row 14",
     "Elsewhere the package states PANDA is a compatibility finding, not a validation or model result; the labels read as a validation attempt.", "TERMINOLOGY (documentation)"),
    ("results_v2/figures/stage20/figure7_supplementary_panda_compatibility.png (title)",
     "Title says '10-case pilot'", "results_v2/validation/stage18_panda/summary_table.csv",
     "The figure shows nine bars from six cases (00bbc148, 0068d4c7, 0018ae58, 00928370, 00951a7f, 018eabc8); 006f6aa3 is omitted (no benign reference) and two cases have no G3/G4/G5 component.", "LABEL (documentation)"),
    ("results_v2/tables/stage20/table5_pathology_crag.csv column 'statistic'",
     "P1/P3 (Cohen's d) and P2/P4 (Spearman rho) share one 'statistic' column without a type label", "stage15_results.json",
     "Values match; effect-size type is recoverable only from the row label.", "LABEL (documentation)"),
    ("reports/STAGE12_AUDIT_REPORT.md:106",
     "|delta S| ~0.03-0.04 described as 'small on the [0,1] scale'", "stage12_results.json per_angle S_abs_diff_mean; stage15_results.json mean S 0.029-0.031",
     "Mean |delta S| (0.031-0.039) is about the same size as the mean predicted S (~0.03), so 'small' is scale-dependent. S-invariance and phi-shift results are non-gating and are not reported in the Stage 21 package.", "WORDING / OMISSION (documentation)"),
]
for loc, item, ref, note, res in SEMANTIC:
    out.append({"check_type": "semantic_discrepancy", "location": loc, "item": item, "reference": ref,
                "resolved_source_value": "", "result": res, "note": note})

fields = ["check_type", "location", "item", "reference", "resolved_source_value", "result", "note"]
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out)

from collections import Counter  # noqa: E402
print("rows in numbers table:", len(rows))
print(Counter((o["check_type"], o["result"]) for o in out))
print("token tally (tokens_checked, not_found):")
for k, v in tally.items():
    print(f"  {k}: {v}")
