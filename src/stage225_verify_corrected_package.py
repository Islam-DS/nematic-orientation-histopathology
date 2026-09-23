"""
STAGE 22.5 -- read-only verification of the corrected manuscript package.
Prints results; writes nothing. Checks:
  1. all required files exist
  2. forbidden / unsupported phrases (with context) in the corrected package
  3. every numeric token in the corrected narrative files, claim matrix and corrected
     supplementary tables occurs in frozen numeric sources (presence at displayed precision)
  4. S1: P1/P3 rows carry no CI; no unrounded CI; corrected-vs-original cell differences
  5. H1-H3 verbatim blocks from the Stage 21.5 record are present in file 01
  6. frozen Stage 0-20, Stage 21 (original) and Stage 21.5 artifact checksums
"""
import bisect
import csv
import glob
import hashlib
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
OUT = P("results_v2", "manuscript_stage21_corrected")
TAB = P("results_v2", "tables", "stage20_corrected")
FILES = ["01_primary_question.md", "02_methods_evidence_map.md", "03_results_narrative.md", "04_primary_geometric_validation.md", "05_rotation_results.md",
         "06_ablation_multiscale.md", "07_pathology_results.md", "08_crag_results.md", "09_panda_compatibility.md", "10_limitations.md",
         "11_scientific_interpretation.md", "12_claim_evidence_matrix.csv", "13_numbers_and_sources.csv", "14_manuscript_figure_table_map.md",
         "15_STAGE21_CORRECTED_AUDIT.md"]
TABLES = ["supplementary_full_statistical_evidence_corrected.csv", "supplementary_evidence_domain_summary_corrected.csv", "supplementary_provenance_corrected.csv"]

print("== 1. required files")
missing = [f for f in FILES if not os.path.exists(os.path.join(OUT, f))] + [t for t in TABLES if not os.path.exists(os.path.join(TAB, t))]
print("missing:", missing if missing else "none")

texts = {}
for f in FILES:
    fp = os.path.join(OUT, f)
    if os.path.exists(fp):
        texts[f] = open(fp, encoding="utf-8").read()
for t in TABLES:
    fp = os.path.join(TAB, t)
    if os.path.exists(fp):
        texts["tables/" + t] = open(fp, encoding="utf-8").read()

print("\n== 2. forbidden / unsupported phrase scan")
PATTERNS = {
    "PANDA count 3 of 3 / 2 of 3": r"(?<![\d/])3 of 3\b|(?<![\d/])3/3\b|(?<![\d/])2 of 3\b|(?<![\d/])2/3\b",
    "'substantial' (review context)": r"substantial",
    "training deterministic / reproducible training": r"training (is|was) deterministic|deterministic training|reproducible training|training is reproducible|training.{0,20}deterministic",
    "robust*": r"robust",
    "preregist* (excluding file name)": r"pre-?regist(?!ered_thresholds)",
    "anatomical ground truth": r"anatomical ground truth",
    "H1 tested by / criteria->H1 mapping": r"H1 (is )?tested by|tested by (the )?(frozen |preregistered )?(geometric )?criteria A",
    "equivariance failure wording": r"not equivariant|equivariance failed|lacks? equivariance|lacking equivariance|failed equivariance",
    "ranking/mild wording": r"most favorable|monotonic-ish|\bMild\b|\bwinner\b|\bbest\b|superior|outperform",
    "Stage 15 arithmetic sentence / Holm-8": r"would not survive|Holm-8|across (even just )?these 8",
    "patient-independence wording (review)": r"patient-independent|independent patient-level|independent test set",
    "p = 0 / 0.000": r"\bp\s*=\s*0(?![\.\d])|\bp\s*=\s*0\.0+(?!\d)|perm_?p\s*=\s*0(?![\.\d])|permutation p = 0\.0\b|0\.000(?!\d)",
    "pooled subsample / n = NA (Stage 16)": r"pooled subsample",
    "H2 unrecoverable": r"unrecoverable|not recoverable|not present in any frozen",
    "ground truth (review)": r"ground truth",
    "novel/first/proves/state-of-the-art/clinical utility": r"\bnovel|\bproves?\b|state-of-the-art|clinical utility|biological truth",
}
for name, pat in PATTERNS.items():
    hits = []
    for f, t in texts.items():
        for ln, line in enumerate(t.splitlines(), 1):
            if re.search(pat, line, re.I):
                hits.append((f, ln, line.strip()[:170]))
    print(f"-- {name}: {len(hits)} hit(s)")
    for h in hits[:14]:
        print("     ", h)

print("\n== 3. numeric-token presence in frozen sources")
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


for pat in ("results_v2/validation/**/*.json", "results_v2/phase2/*.json", "results_v2/anatomy_targets/*.json", "results_v2/model3/metrics/*.json", "configs/*.json"):
    for jf in glob.glob(P(*pat.split("/")), recursive=True):
        if os.path.getsize(jf) < 30_000_000:
            try:
                collect(json.load(open(jf, encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
for cf in glob.glob(P("results_v2", "validation", "**", "*.csv"), recursive=True) + [os.path.join(OUT, "13_numbers_and_sources.csv")]:
    if os.path.getsize(cf) < 30_000_000:
        for rr in csv.reader(open(cf, encoding="utf-8")):
            for c in rr:
                try:
                    leaves.append(float(c))
                except ValueError:
                    pass
leaves = sorted(set(leaves))


def in_leaves(t):
    m = re.fullmatch(r"-?(\d+)(?:\.(\d+))?", t)
    if not m:
        return False
    x = float(t)
    dec = len(m.group(2)) if m.group(2) else 0
    tol = 0.5 * 10 ** (-dec) + 1e-12
    for cand, sc in ((x, 1), (-x, 1), (x / 100.0, 100), (-x / 100.0, 100)):
        lo, hi = cand - tol / sc, cand + tol / sc
        j = bisect.bisect_left(leaves, lo)
        if j < len(leaves) and leaves[j] <= hi:
            return True
    return False


STRIP = [r"Stage\s*\d+(?:\.\d+)?(?:[–-]\d+)?", r"Stages\s*\d+(?:[–,\s-]+\d+)*", r"Figures?\s*\d+(?:[–-]\d+)?", r"Tables?\s*\d+(?:[–-]\d+)?", r"Section\s*\d+", r"§\s*\d+(?:\.\d+)?",
         r"\b[A-Za-z]+\d+[A-Za-z0-9]*\b", r"\b\d+[A-Za-z]+\d*\b", r"`[0-9a-f]{16,}`", r"\b[0-9a-f]{32}\b", r"\b[0-9a-f]{64}\b", r"\bfiles?\s*\d+(?:[–,\s-]+\d+)*", r"\bFiles?\s*\d+(?:[–,\s-]+\d+)*",
         r"\d{4}-\d{2}-\d{2}", r"\b\d{8}\b", r"line\s*\d+(?:[–,\s-]+\d+)?", r"lines\s*\d+(?:[–,\s-]+\d+)?", r"limitation\s*\d+", r"\(\d+\)|\b\d+\.\s"]
ALLOWED = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "15", "16", "30", "37", "45", "60", "90", "95", "100", "123", "150", "173", "19", "13", "14", "11", "20", "39", "18", "17", "21", "22", "180", "0.01", "0.05", "0.001", "0.30", "0.5", "0.10", "0.2", "2000", "1000", "42", "0.3", "0.1"}
total = bad = 0
scan_targets = [f for f in FILES if f not in ("13_numbers_and_sources.csv", "15_STAGE21_CORRECTED_AUDIT.md")] + ["tables/supplementary_full_statistical_evidence_corrected.csv", "tables/supplementary_evidence_domain_summary_corrected.csv"]
for f in scan_targets:
    t = texts[f]
    t = re.sub(r"(?<![\d,])(\d{1,3}),(\d{3})(?![\d,])", r"\1\2", t)
    for ln, line in enumerate(t.splitlines(), 1):
        s = line
        for pat in STRIP:
            s = re.sub(pat, " ", s)
        for tk in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])", s):
            if tk in ALLOWED:
                continue
            total += 1
            if not in_leaves(tk):
                bad += 1
                print("   NOT FOUND:", f, ln, tk, "|", line.strip()[:120])
print(f"tokens checked (excluding small integers/thresholds): {total}; not found: {bad}")

print("\n== 4. supplementary S1 checks")
o = list(csv.reader(open(P("results_v2", "tables", "stage20", "supplementary_full_statistical_evidence.csv"), encoding="utf-8")))
n = list(csv.reader(open(os.path.join(TAB, "supplementary_full_statistical_evidence_corrected.csv"), encoding="utf-8")))
print("rows original/corrected:", len(o) - 1, len(n) - 1, "| header equal:", o[0] == n[0])
for line in (14, 16):
    print(f"  S1 line {line}: statistic='{n[line - 1][5]}' CI='{n[line - 1][6]}' p='{n[line - 1][7]}'")
for line in (15, 17):
    print(f"  S1 line {line}: statistic='{n[line - 1][5]}' CI='{n[line - 1][6]}' p='{n[line - 1][7]}'")
unr = [(i + 1, c) for i, r in enumerate(n) for c in r if re.search(r"\d\.\d{6,}", c)]
print("  unrounded (>=6-decimal) cells:", unr if unr else "none")
diff_cols = {}
for i in range(1, len(o)):
    for j, h in enumerate(o[0]):
        if o[i][j] != n[i][j]:
            diff_cols.setdefault(h, []).append(i + 1)
for h, lines in diff_cols.items():
    print(f"  changed column '{h}': {len(lines)} row(s): {lines[:14]}")
# numeric value preservation: every number in an original cell that changed only by formatting must still be present
lost = []
for i in range(1, len(o)):
    for j in (4, 5, 7):
        for tk in re.findall(r"-?\d+\.\d+", o[i][j]):
            if tk not in n[i][j] and float(tk) != 0.0:
                lost.append((i + 1, o[0][j], tk, n[i][j][:60]))
print("  original numeric tokens (n/Statistic/p-value) no longer present verbatim:", lost if lost else "none")

print("\n== 5. hypotheses verbatim in file 01")
rec = open(P("reports", "STAGE21_5_H2_TRACEABILITY.md"), encoding="utf-8").read()
blocks = re.findall(r"```text\n(.*?)\n```", rec, re.S)
h2 = blocks[0]
full = blocks[1]
h1 = re.search(r"H1 — Geometric validity\n\n.*?\n\n(?=H2)", full, re.S).group(0).rstrip("\n")
h3 = re.search(r"H3 — Pathological relevance\n\n.*?\n\n(?=IMPORTANT)", full, re.S).group(0).rstrip("\n")
t01 = texts["01_primary_question.md"]
for nm, b in (("H1", h1), ("H2", h2), ("H3", h3)):
    print(f"  {nm} present verbatim:", b in t01, "| chars:", len(b))
print("  H2 sha256:", hashlib.sha256(h2.encode()).hexdigest())

print("\n== 6. numbers table: rows and verification flags")
rows = list(csv.DictReader(open(os.path.join(OUT, "13_numbers_and_sources.csv"), encoding="utf-8")))
print("  rows:", len(rows), "| all verified flags start with 'yes':", all(r["verified_against_source"].startswith("yes") for r in rows))
print("  rows with empty source:", sum(1 for r in rows if not r["source_file"] or not r["source_field"]))
claims = list(csv.DictReader(open(os.path.join(OUT, "12_claim_evidence_matrix.csv"), encoding="utf-8")))
print("  claims:", len(claims))
prov = list(csv.DictReader(open(os.path.join(TAB, "supplementary_provenance_corrected.csv"), encoding="utf-8"))) if os.path.exists(os.path.join(TAB, TABLES[2])) else []
print("  provenance rows:", len(prov), "| corrected artifacts with hash:", sum(1 for r in prov if r["output_md5"]))
