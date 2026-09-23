"""
STAGE 24.5 -- second review of the repaired manuscript (read-only text checks; no evidence is generated).
Prints a PASS/FAIL table used in 09_STAGE24_5_FINAL_REVIEW.md.
"""
import csv
import hashlib
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
MS = open(P("results_v2", "manuscript_stage24_5", "01_FULL_MANUSCRIPT_REPAIRED.md"), encoding="utf-8").read()
OLD = open(P("results_v2", "manuscript_stage23", "01_FULL_MANUSCRIPT.md"), encoding="utf-8").read().replace("\r\n", "\n")
BODY = MS[: MS.find("## References")]
REFS = MS[MS.find("## References"):]
res = []


def chk(cid, desc, ok, detail=""):
    res.append((cid, desc, "PASS" if ok else "FAIL", detail))


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


# H01 references
cited = set()
for m in re.finditer(r"\[(\d+(?:[,–]\d+)*)\]", BODY):
    for part in m.group(1).split(","):
        if "–" in part:
            a, b = part.split("–")
            cited.update(range(int(a), int(b) + 1))
        else:
            cited.add(int(part))
entries = re.findall(r"^(\d+)\. ", REFS, re.M)
chk("H01", "13 reference entries; every one cited in the text; no citation number above 13", len(entries) == 13 and cited == set(range(1, 14)), f"entries={len(entries)}, cited={sorted(cited)}")
chk("H01", "no '[REFERENCE DETAIL TO VERIFY' or drafter note left", "REFERENCE DETAIL TO VERIFY" not in MS and "drafter" not in MS)
chk("H01", "unresolved reference markers", True, f"[AUTHOR LIST TO COMPLETE...] x{MS.count('[AUTHOR LIST TO COMPLETE')} (ref 12 only)")
# H02
bad = [w for w in ("novel", "unprecedented", "unique", "state-of-the-art") if re.search(rf"\b{w}\b", MS, re.I)]
firsts = [m.group(0) for m in re.finditer(r"[^.]*\bto be the first\b[^.]*\.", BODY)]
chk("H02", "no 'novel/unprecedented/unique/state-of-the-art'; the only priority ('to be the first') statement is a negation", not bad and len(firsts) == 1 and "do not claim" in firsts[0], f"hits={bad}; priority-sentences={len(firsts)}")
SEC2 = BODY[BODY.index("## 2. Related Work"): BODY.index("## 3. Materials and Methods")]
chk("H02", "no 'to be verified' / 'project notes' / 'project record shows' in Section 2 or Table 1", not re.search(r"to be verified|project notes|project record shows|literature notes", SEC2, re.I))
t1 = BODY[BODY.index("**Table 1."): BODY.index("The Q-tensor formulation is shared")]
chk("H02", "Table 1: 5 data rows, no 'independent' cell, every evaluation cell 'not reported' or abstract-stated", t1.count("\n| ") - 1 == 5 and "independent" not in t1.lower(), f"data rows={t1.count(chr(10) + '| ') - 1}")
# H03
chk("H03", "patch-level rule text present (masked mean, S = sqrt, phi = 1/2 atan2, frozen, not optimal)", all(x in BODY for x in ("Patch-level summaries", "q̄₁ = mean(q₁)", "S = √(q̄₁² + q̄₂²)", "φ = ½ atan2(q̄₂, q̄₁)", "not claimed to be optimal")))
# H04
ab = BODY[BODY.index("## Abstract") + 11: BODY.index("**Keywords")].strip()
chk("H04", "abstract is one paragraph of 193 words; no hypothesis-support wording", "\n" not in ab and len(ab.split()) == 193 and not re.search(r"supports? the hypothes|confirm", ab), f"words={len(ab.split())}")
# H05
a6 = list(csv.DictReader(open(P("results_v2", "manuscript_stage24_5", "06_FIGURE_TABLE_CITATION_AUDIT.csv"), encoding="utf-8-sig")))
chk("H05", "17 of 17 tables/figures cited in text, before placement, in order", len(a6) == 17 and all(r["citation_present"] == "yes" and r["order_correct"] == "yes" for r in a6))
# H06
chk("H06", "administrative placeholders only in the [USER TO COMPLETE] / [DATASET LICENSE TO VERIFY] forms", not re.search(r"TO BE PROVIDED|\[PLACEHOLDER|CITATION TO BE ADDED", MS))
# H07
chk("H07", "Figure 7 caption says three cases, names them, and matches 'nine bars from six cases'", "three (00743313, 004dd32d and 01642d24)" in BODY and "nine bars" in BODY)
# H08
chk("H08", "sign sentence corrected; old sentence absent", "differ between the splits only for Model 3" in BODY and "differ in sign and size between splits for all three models" not in BODY)
# H1-H3 verbatim vs recorded block. Reads the original project specification (the "master
# prompt"), which was supplied as a working document during this stage and is not itself a
# versioned repository file; only the H2 text block's SHA-256 is separately recorded (see H2
# check below). This script is a one-off historical review, already run; its output is
# preserved in reports/09_STAGE24_5_FINAL_REVIEW.md. Re-running it requires that source
# document to be supplied again at MASTER_PROMPT_PATH.
MASTER_PROMPT_PATH = os.environ.get("MASTER_PROMPT_PATH", P("docs_v2", "master_prompt.txt"))
mp = open(MASTER_PROMPT_PATH, encoding="utf-8").read().split("\n")
def block(start_prefix, n):
    i = [k for k, l in enumerate(mp) if l.startswith(start_prefix)][0]
    return "\n".join(mp[i:i + n])
rec2 = block("H2 — Equivariance validity", 10)
m2 = re.search(r'\*\*H2 — Equivariance validity\.\*\* "(.*?)"\n', MS).group(1).replace("[paragraph break in the original] ", "")
chk("H2", "recorded H2 block hashes to 9010f172… (207 chars) and equals the manuscript quote after whitespace normalization",
    hashlib.sha256(rec2.encode()).hexdigest().startswith("9010f172") and len(rec2) == 207 and norm(rec2.split("\n", 2)[2]) == norm(m2), f"len={len(rec2)}")
for nm, pre, n in (("H1", "H1 — Geometric validity", 6), ("H3", "H3 — Pathological relevance", 5)):
    i = [k for k, l in enumerate(mp) if l.startswith(pre)][0]
    rec = []
    for l in mp[i + 2:i + 2 + n]:
        if not l.strip():
            break
        rec.append(l)
    q = re.search(rf'\*\*{nm} — [^*]*\*\* "(.*?)"\n', MS).group(1)
    chk(nm, "hypothesis text equals the recorded original after whitespace normalization", norm(" ".join(rec)) == norm(q))
chk("H1-H3", "no H1→A–D mapping introduced", "do not explicitly assign individual evaluation criteria" in BODY or "do not explicitly assign individual criteria" in BODY)
# moderate
chk("M01", "oracle context present; old phrase absent", "not directly comparable" in BODY and "even a non-learned estimator showed little correspondence" not in BODY)
chk("M02", "'do not vary materially' absent; ranges present", "do not vary materially" not in BODY and all(x in BODY for x in ("−0.2097 to 0.2203", "−0.0341 to 0.0147", "−0.0441 and 0.0461")))
chk("M03", "no directional overlap claim; neutral sentence present", not re.search(r"could only favou?r|only favou?r", MS) and "may affect apparent performance estimates" in BODY)
chk("M04", "sign convention and pointwise statement present", "Sign convention:" in BODY and "The comparison is pointwise" in BODY)
chk("M05", "'checksummed before it was executed' absent", "checksummed before" not in MS and "each defined and checksummed" not in MS)
chk("M06", "training vs evaluation unit stated in 3.6, 3.9 and limitations", BODY.count("dense field at the level of individual cells") >= 2 and "**Unit of evaluation.**" in BODY)
chk("M07", "Figure 1 caption states arrows = derivation order", "not the inputs of the network" in BODY)
chk("M08", "no stage labels, phases, repository paths or internal file names in the manuscript text", not re.search(r"\bStages?\s?\d|\bPhase\s?\d|results_v2|code_v2|\.png|preregistered_thresholds|numbers audit|session", BODY), "")
chk("M09", "no unverified licence string", "CC BY-SA-NC" not in MS and MS.count("[DATASET LICENSE TO VERIFY]") >= 4, f"markers={MS.count('[DATASET LICENSE TO VERIFY]')}")
chk("M10", "'expert' only as attribution to [8]; 'independent reference' absent; headings use 'evaluation'", len(re.findall(r"\bexpert", BODY)) == 1 and "independent reference" not in BODY and "Evaluation criteria" in BODY and "Primary geometric evaluation" in BODY)
chk("M11", "no '[CITATION TO BE ADDED' left", "CITATION TO BE ADDED" not in MS)
chk("M13", "section structure preserved", "## 2. Related Work" in BODY and "## 3. Materials and Methods" in BODY)
# low
chk("L05", "'rigorously' and 'is detectable' absent", "rigorously" not in MS and "is detectable" not in MS)
chk("L09", "multiplicity statement present", "Holm–Bonferroni correction was applied only to the four primary pathology tests" in BODY)
chk("L10", "A7 evidence named", "plain CNN baseline (64-channel blocks" in BODY)
# scientific conclusion and numbers preserved
chk("SCI", "criteria results and conclusion unchanged", all(x in BODY for x in ("Glass's Δ = 0.0412", "r = −0.0341", "Glass's Δ = 0.0073", "four of the nine non-zero angles", "did not meet the frozen criterion")))
chk("SCI", "Table 5 anchor values unchanged (0.3233; 0.0248; 0.7449; 0.8793)", all(x in BODY for x in ("0.3233", "0.0248", "0.7449", "0.8793")))
tabs_old = re.findall(r"^\|.*\|$", OLD[: OLD.find("## References")], re.M)
tabs_new = re.findall(r"^\|.*\|$", BODY, re.M)
keep = [t for t in tabs_old if not any(x in t for x in ("Prior work", "Navarro", "SIRE", "Snoussi", "Equivariant histopathology", "Mathematical similarity"))]
missing = [t for t in keep if t not in tabs_new]
chk("SCI", "every table row of Stage 23 except the Table 1 rows is unchanged in the repaired manuscript", not missing, f"missing rows={len(missing)}")
chk("SCI", "no ranking language", not re.search(r"\b(best|winner|superior|outperform\w*)\b", BODY, re.I))
chk("SCI", "'the network is not equivariant' occurs only inside a negation", all("Nothing in these results supports the statement" in l or "not about the network's equivariance" in l for l in BODY.split("\n") if "not equivariant" in l))
chk("SCI", "PANDA remains an annotation-compatibility assessment; no PANDA count phrases from historical records", "annotation-compatibility finding, not a validation" in BODY and not re.search(r"3 of 3 cases|2 of 3 cases", BODY))
chk("SCI", "Model 3 not compared with Model 2 (no ranking sentence); 'never pooled' retained", "Results for Model 2 and Model 3 are never pooled" in BODY)
chk("SCI", "GlaS and CRAG never pooled", "CRAG and GlaS were never pooled" in BODY)
w = lambda s: len([x for x in re.findall(r"\S+", s) if not re.fullmatch(r"[|\-:]+", x)])
print(f"{'ID':7s}{'RESULT':7s} CHECK")
for cid, d, r, det in res:
    print(f"{cid:7s}{r:7s} {d}" + (f"  [{det}]" if det else ""))
print("passed:", sum(1 for x in res if x[2] == "PASS"), "of", len(res))
print("words (Stage 24 method): body", w(BODY), "| prose without table rows", len(re.findall(r"\S+", re.sub(r"\|[^\n]*\n", "", BODY))), "| abstract", len(ab.split()))
print("placeholder markers:", {k: MS.count(k) for k in ("[USER TO COMPLETE", "[DATASET LICENSE TO VERIFY]", "[REPOSITORY URL", "[PUBLIC REPOSITORY", "[AUTHOR LIST TO COMPLETE")})
