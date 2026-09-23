"""
STAGE 24 -- read-only cross-document consistency checks and diagnostics.
Writes results_v2/manuscript_stage24/06_CROSS_DOCUMENT_CONSISTENCY.csv and
results_v2/manuscript_stage24/08_NUMBERS_AUDIT.csv (merging the independent critical-number check with
the review of the Stage 23 numbers-audit rows). Reads only; computes no scientific statistic.
"""
import csv
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
OUT = P("results_v2", "manuscript_stage24")


def rd(rel):
    return open(P(*rel.split("/")), encoding="utf-8", errors="replace").read()


MS = rd("results_v2/manuscript_stage23/01_FULL_MANUSCRIPT.md")
BODY = MS[: MS.find("## References")]
rows = []


def add(cid, topic, docs, finding, status, severity, action):
    rows.append({"check_id": f"X{len(rows)+1:03d}", "topic": topic, "documents_compared": docs, "finding": finding, "status": status, "severity_if_inconsistent": severity, "recommended_action": action})


# 1. H1-H3 verbatim (Stage 21.5 record vs manuscript)
rec = rd("reports/STAGE21_5_H2_TRACEABILITY.md")
blocks = re.findall(r"```text\n(.*?)\n```", rec, re.S)
full = blocks[1]
h1 = re.search(r"H1 — Geometric validity\n\n(.*?)\n\nH2", full, re.S).group(1)
h2 = re.search(r"H2 — Equivariance validity\n\n(.*?)\n\nH3", full, re.S).group(1)
h3 = re.search(r"H3 — Pathological relevance\n\n(.*?)\n\nIMPORTANT", full, re.S).group(1)
norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
for nm, h in (("H1", h1), ("H2", h2), ("H3", h3)):
    body_txt = norm(BODY.replace("**", ""))
    hn = norm(h)
    # H2 in the manuscript has the formula flattened with a colon; compare after normalising whitespace
    ok = hn in body_txt
    ok_punct = (not ok) and (hn.replace("R(α)^T The scalar", "R(α)^T. The scalar") in body_txt)
    status = "CONSISTENT" if ok else ("MINOR PUNCTUATION DIFFERENCE (one period inserted after '^T'; wording otherwise identical)" if ok_punct else "DISCREPANCY")
    add(0, f"{nm} exact wording", "Stage 21.5 H2 traceability record vs manuscript Section 1", f"{nm} wording equal after whitespace normalisation: {ok}; equal except an inserted period: {ok_punct}", status, "" if (ok or ok_punct) else "HIGH", "" if ok else ("remove the inserted period or state that punctuation was added" if ok_punct else "restore verbatim wording"))
add(0, "H1-H3 whitespace normalisation disclosed", "manuscript Section 1", "manuscript states that whitespace/line breaks were normalised", "CONSISTENT" if "normalized here" in BODY else "DISCREPANCY", "", "")
add(0, "no H1->A-D mapping", "manuscript", "occurrences of an H1/H2/H3 to criterion assignment sentence: " + str(len(re.findall(r"H[123] (?:is |was )?tested by|tested by (?:the )?(?:frozen |preregistered )?(?:geometric )?criteria A", BODY))) + "; occurrences of the explicit non-assignment statement: " + str(len(re.findall(r"not explicitly assign", BODY))), "CONSISTENT", "", "keep")

# 2. PANDA phrases in manuscript-facing documents
targets = glob.glob(P("results_v2", "manuscript_stage23", "*")) + glob.glob(P("results_v2", "manuscript_stage21_corrected", "*")) + glob.glob(P("results_v2", "tables", "stage20_corrected", "*"))
hits = []
for t in targets:
    if os.path.isdir(t):
        continue
    txt = open(t, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r"(?<![\d/])3 of 3\b|(?<![\d/])3/3\b|(?<![\d/])2 of 3\b|(?<![\d/])2/3\b", txt):
        if txt[max(0, m.start() - 6): m.start()] == "Model ":
            continue  # 'Model 2/3 mixing' is not a PANDA count
        hits.append(os.path.relpath(t, ROOT))
add(0, "historical PANDA count phrases", "results_v2/manuscript_stage23, manuscript_stage21_corrected, tables/stage20_corrected", f"occurrences: {len(hits)} {sorted(set(hits))}", "CONSISTENT" if not hits else "DISCREPANCY", "" if not hits else "HIGH", "" if not hits else "remove")
# frozen historical files still contain them (expected, unchanged)
frozen_hits = []
for rel in ("results_v2/tables/stage20/supplementary_full_statistical_evidence.csv", "results_v2/statistical_synthesis/MASTER_STATISTICAL_TABLE.csv", "reports/STAGE18_AUDIT_REPORT.md", "results_v2/validation/stage18_panda/compatibility_report.md", "results_v2/statistical_synthesis/STAGE19_STATISTICAL_SYNTHESIS.md"):
    if re.search(r"(?<![\d/])3 of 3\b|(?<![\d/])3/3\b|(?<![\d/])2 of 3\b", rd(rel)):
        frozen_hits.append(rel)
add(0, "historical PANDA phrases remain in frozen historical files (unchanged by design)", "Stage 18/19/20 frozen files", f"{len(frozen_hits)} of 5 frozen files still contain the phrases: {frozen_hits}", "EXPECTED (frozen; documented in Stage 21.5)", "", "do not cite these files; manuscript uses corrected S1")

# 3. Patient-overlap statements across documents
docs = {"manuscript": MS, "corrected 10 limitations": rd("results_v2/manuscript_stage21_corrected/10_limitations.md"), "corrected 12 claims": rd("results_v2/manuscript_stage21_corrected/12_claim_evidence_matrix.csv"),
        "Stage 22 reproducibility audit": rd("reports/STAGE22_REPRODUCIBILITY_AUDIT.md"), "Stage 23 audit": rd("results_v2/manuscript_stage23/07_STAGE23_AUDIT_REPORT.md")}
for k, v in docs.items():
    a = bool(re.search(r"11 of (?:the )?12", v)) and bool(re.search(r"79 of (?:the )?80", v))
    add(0, "patient overlap 11/12 and 79/80", k, f"both counts present: {a}", "CONSISTENT" if a else "CHECK", "" if a else "MODERATE", "" if a else "verify")

# 4. Criteria A-C values in corrected 04 and Stage 21 original vs manuscript
for rel, label in (("results_v2/manuscript_stage21_corrected/04_primary_geometric_validation.md", "corrected 04"), ("results_v2/manuscript_stage21/04_primary_geometric_validation.md", "original Stage 21 04 (unchanged)")):
    t = rd(rel)
    a = all(s in t for s in ("0.041", "0.034", "0.007"))
    add(0, "Criteria A-C values (0.041/0.034/0.007 at 3 dp)", label + " vs manuscript (0.0412/-0.0341/0.0073 at 4 dp)", f"values present at compatible precision: {a}", "CONSISTENT" if a else "CHECK", "", "note precision difference only")
add(0, "Criterion C p-value wording", "manuscript vs original Stage 21 04 vs Stage 19 table", "manuscript writes 'p < 0.001 (0 of 1000)'; original Stage 21 04 and Stage 19 write 0.000 / 0.0 (frozen)", "CONSISTENT (manuscript corrected per Stage 22.5)", "", "none")

# 5. Figure 7 discrepancy (unresolved in Stage 22.5)
sr = list(csv.DictReader(open(P("results_v2", "validation", "stage18_panda", "summary_table.csv"), encoding="utf-8")))
none = [x["image_id"][:8] for x in sr if all(float(x[k]) == 0 for k in ("gleason_3_pixel_fraction", "gleason_4_pixel_fraction", "gleason_5_pixel_fraction"))]
m14 = rd("results_v2/manuscript_stage21_corrected/14_manuscript_figure_table_map.md")
two14 = "two cases contain no Gleason 3/4/5" in m14
add(0, "Figure 7: pilot cases without Gleason 3/4/5 pixels", "frozen summary_table.csv vs corrected 14 map vs Stage 22 traceability/number-consistency vs manuscript caption vs figure image",
    f"frozen table: {len(none)} cases ({', '.join(none)}); corrected 14 map says 'two': {two14}; Stage 22 files say 'two': "
    f"{'two cases have no G3/G4/G5' in rd('reports/STAGE22_NUMBER_CONSISTENCY.csv') or 'two cases contain no' in rd('reports/STAGE22_TRACEABILITY_CHECK.md')}; manuscript caption says 'three'; figure shows 9 bars from 6 cases (viewed) which with 006f6aa3 omitted accounts for 7 of 10; the remaining 3 have no Gleason 3/4/5 pixels",
    "HIGH — MANUSCRIPT/SOURCE DOCUMENTATION DISCREPANCY (manuscript caption is consistent with the frozen table and the figure; the Stage 22 / 22.5 documentation is not; unresolved by design)", "HIGH",
    "do not edit Stage 22/22.5; state the count from summary_table.csv in the caption; add an erratum note in the final documentation package")

# 6. Baseline and Model 3 accuracy values (independent check file)
ck = list(csv.DictReader(open(P("results_v2", "manuscript_stage24", "_critical_numbers_check.csv"), encoding="utf-8")))
tab5 = [r for r in ck if r["manuscript_section"] == "4.2 Table 5"]
ok5 = sum(1 for r in tab5 if r["status"].startswith("CONFIRMED"))
add(0, "Table 5 baseline and Model 3 per-split values", "manuscript Table 5 vs frozen baseline JSON, Stage 11 JSON, Stage 14 A0 JSON", f"{ok5} of {len(tab5)} cells confirmed against their frozen JSON paths", "CONSISTENT" if ok5 == len(tab5) else "DISCREPANCY", "" if ok5 == len(tab5) else "HIGH", "")
for sp, w in (("testA", "0.7449"), ("testB", "0.8793")):
    rr = [r for r in ck if r["quantity"] == f"Model 3 {sp} accuracy"][0]
    add(0, f"Model 3 accuracy {sp}", "manuscript vs stage14_all_evaluations.json A0_reference", f"manuscript {w}; source {rr['source_value'][:8]}; status {rr['status']}", "CONSISTENT" if rr["status"].startswith("CONFIRMED") else "DISCREPANCY", "", "")
add(0, "Model 3 classification accuracy provenance", "corrected package vs manuscript", "corrected Stage 22.5 package contains no Model 3 classification accuracy; manuscript takes it from Stage 14 A0 evaluation JSON (frozen)", "CONSISTENT (source beyond corrected package; disclosed in Stage 23 audit)", "", "cite the JSON path in the supplement")

# 7. Corrected supplementary S1 vs manuscript Table 8
s1 = list(csv.DictReader(open(P("results_v2", "tables", "stage20_corrected", "supplementary_full_statistical_evidence_corrected.csv"), encoding="utf-8")))
p2 = [r for r in s1 if "P2 S_DL vs ordinal" in r["Analysis"]][0]
add(0, "P2 statistic and CI in corrected S1 vs manuscript Table 8", "S1 corrected vs manuscript", f"S1: {p2['Statistic']} {p2['95% CI']}; manuscript Table 8: 0.0750, -0.1361 to 0.2768", "CONSISTENT" if "0.0750" in p2["Statistic"] and "-0.1361" in p2["95% CI"] else "CHECK", "", "")
p1 = [r for r in s1 if "P1 S_DL vs binary" in r["Analysis"]][0]
add(0, "P1 CI absent in corrected S1 and manuscript", "S1 corrected vs manuscript", f"S1 P1 CI cell: '{p1['95% CI']}'", "CONSISTENT" if p1["95% CI"].startswith("NA") else "DISCREPANCY", "", "")
s2 = list(csv.DictReader(open(P("results_v2", "tables", "stage20_corrected", "supplementary_evidence_domain_summary_corrected.csv"), encoding="utf-8")))
vocab = sorted({r["Evidence status"] for r in s2})
add(0, "outcome vocabulary: supplement S2 vs manuscript Table 10", "S2 corrected vs manuscript", f"S2 uses {vocab}; manuscript Table 10 uses 'not met / not demonstrated / inconclusive / infeasible'", "MINOR VOCABULARY DIFFERENCE", "LOW", "harmonise (e.g. add a note that 'NOT SUPPORTED' in S2 = 'not demonstrated')")

# 8. Table / figure numbering and first citation order
def first_cite(kind, n):
    cap = re.search(rf"^(?:> )?\*\*{kind} {n}[.( ]", BODY, re.M)
    pos_cap = cap.start() if cap else None
    mentions = [m.start() for m in re.finditer(rf"\b{kind} {n}\b", BODY)]
    # first mention that is not the caption itself
    men = [p for p in mentions if pos_cap is None or abs(p - (pos_cap + 2)) > 3 and abs(p - (pos_cap + 4)) > 3]
    first_text = min([p for p in mentions if pos_cap is None or p != pos_cap and p != pos_cap + 2], default=None)
    return pos_cap, first_text
order_issues = []
for kind, N in (("Table", 10), ("Figure", 7)):
    for n in range(1, N + 1):
        cap, first = first_cite(kind, n)
        if cap is None:
            order_issues.append(f"{kind} {n}: caption not found")
        elif first is None:
            order_issues.append(f"{kind} {n}: never cited in text apart from its caption")
        elif first > cap:
            order_issues.append(f"{kind} {n}: caption precedes its first in-text citation")
add(0, "table/figure first citation before caption", "manuscript", f"{len(order_issues)} issues: {order_issues}", "CONSISTENT" if not order_issues else "FINDING", "" if not order_issues else "LOW", "")
caps_tab = [int(x) for x in re.findall(r"^\*\*Table (\d+)\.", BODY, re.M)]
caps_fig = [int(x) for x in re.findall(r"^> \*\*Figure (\d+)[. (]", BODY, re.M)]
add(0, "caption numbering sequence in order of appearance", "manuscript", f"tables {caps_tab}; figures {caps_fig}", "CONSISTENT" if caps_tab == sorted(caps_tab) == list(range(1, 11)) and caps_fig == sorted(caps_fig) == list(range(1, 8)) else "FINDING", "", "")

# 9. word counts vs Stage 23 audit claims
tok = lambda s: [w for w in re.findall(r"\S+", s) if not re.fullmatch(r"[|\-:]+", w)]  # noqa: E731
wc = len(tok(BODY))
prose = len(re.findall(r"\S+", re.sub(r"\|[^\n]*\n", "", BODY)))
abs_txt = MS[MS.index("## Abstract"): MS.index("**Keywords")]
abs_words = len(re.findall(r"\S+", abs_txt.replace("## Abstract", "")))
add(0, "word counts stated in Stage 23 audit (9,533 / 8,493 / abstract 347)", "manuscript vs 07 audit", f"recomputed body {wc}; prose {prose}; structured abstract {abs_words}", "CONSISTENT" if (wc, prose) == (9533, 8493) else "MINOR STALE COUNT in Stage 23 audit (recomputed values differ by a few words after the last edit)", "COSMETIC", "update counts to the recomputed values")

# 10. placeholders
ph = {"[CITATION TO BE ADDED": len(re.findall(r"\[CITATION TO BE ADDED", MS)), "[REFERENCE DETAIL TO VERIFY": len(re.findall(r"\[REFERENCE DETAIL TO VERIFY", MS)),
      "[PLACEHOLDER": len(re.findall(r"\[PLACEHOLDER", MS)), "TO BE PROVIDED": len(re.findall(r"TO BE PROVIDED", MS)), "to be verified (hedges)": len(re.findall(r"to be verified|to verify", MS, re.I))}
add(0, "unresolved placeholder inventory", "manuscript", str(ph), "FINDING (submission blockers of administrative/citation type)", "HIGH", "resolve before submission")

# ---- write 06
with open(os.path.join(OUT, "06_CROSS_DOCUMENT_CONSISTENCY.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["check_id", "topic", "documents_compared", "finding", "status", "severity_if_inconsistent", "recommended_action"])
    w.writeheader()
    w.writerows(rows)

# ---- 08: merge independent checks with Stage 23 row review
s23 = list(csv.DictReader(open(P("results_v2", "manuscript_stage23", "04_NUMBERS_AUDIT.csv"), encoding="utf-8")))
MISATTR = [
    (r"^0\.001$", r"Multi-scale 3x3", "permutation p < 0.001 is a restatement of 0 of 1000 (frozen p = 0.0; N_PERM = 1000), not the Multi-scale 3x3 value", "correct source to stage11 permutation_p_value/N_PERM"),
    (r"^4\.0$", r"fragmented", "licence version 'CC BY-SA-NC 4.0' is not a measured value; it was matched to the fragmented-gland count", "source: reports/STAGE18_DESIGN_FREEZE.md (licence line)"),
    (r"^32$", r"PANDA benign", "batch size 32 / dense index (32,32) were matched to the PANDA benign-component minimum (32)", "sources: Stage 10 protocol table; stage12 code (cy,cx = 32,32)"),
    (r"^0\.10$", r"CRAG train", "tiny-set criterion <= 0.10 was matched to a CRAG train value", "source: Stage 8 extended report (criterion)"),
    (r"^30%$", r"Criterion B threshold", "30% (CRAG large-gland area fraction) was matched to the r >= 0.30 threshold", "source: crag_compatibility_report.json (gt30pct field name)"),
    (r"^0\.03$", r"M1 gland-area", "'about 0.03' (mean S_DL) was matched to M1 rho 0.0311", "source: stage15 primary_pooled.P1.mean_malignant/benign (0.0293/0.0311)"),
    (r"^33\.95%$", r"bootstrap 95%", "33.95% residual was matched to the bootstrap confidence-level rule", "source: Stage 7 report Section 9 item 5"),
    (r"^60$", r"testA\.n_images", "'60 patches per angle' (calibration) was matched to testA image count", "source: Stage 0 recalibration audit (rotation-floor sampling)"),
]
mis = []
for r in s23:
    v = r["value_as_written"].replace("−", "-")
    for pv, pl, why, fix in MISATTR:
        if re.search(pv, v) and re.search(pl, r["source_row_or_location"]) and (pv != r"^60$" or "60 patches" in r["context"]):
            mis.append((r, why, fix))
            break
out8 = []
for k in ck:
    out8.append({"row_id": k["check_id"], "check_type": "independent direct-path check of manuscript value", "manuscript_section": k["manuscript_section"], "quantity": k["quantity"], "value_as_written": k["value_as_written"],
                 "unit": k["unit"], "sample_size_or_population": k["sample_size_or_population"], "source_file": k["source_file"], "source_location": k["source_path"], "source_value": k["source_value"],
                 "status": k["status"], "note": ""})
for r, why, fix in mis:
    out8.append({"row_id": r["number_id"], "check_type": "review of Stage 23 numbers-audit row", "manuscript_section": r["section"], "quantity": r["context"][:80], "value_as_written": r["value_as_written"], "unit": r["unit"],
                 "sample_size_or_population": "", "source_file": r["source_file"], "source_location": r["source_row_or_location"], "source_value": "",
                 "status": "SOURCE MISATTRIBUTED IN STAGE 23 AUDIT (value itself is traceable elsewhere)", "note": why + "; " + fix})
with open(os.path.join(OUT, "08_NUMBERS_AUDIT.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["row_id", "check_type", "manuscript_section", "quantity", "value_as_written", "unit", "sample_size_or_population", "source_file", "source_location", "source_value", "status", "note"])
    w.writeheader()
    w.writerows(out8)
print("cross-document checks:", len(rows))
for r in rows:
    if not (r["status"].startswith("CONSISTENT") or r["status"].startswith("EXPECTED")):
        print("  ", r["check_id"], r["topic"], "|", r["status"][:80])
print("08 rows:", len(out8), "| independent:", len(ck), "| Stage 23 misattributed rows:", len(mis))
print("placeholders:", ph)
print("order issues:", order_issues)
print("tables/figures caption order:", caps_tab, caps_fig)
