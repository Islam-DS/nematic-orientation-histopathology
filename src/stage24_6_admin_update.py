"""
STAGE 24.6 -- administrative update of the repaired manuscript (text/metadata only).

Reads results_v2/manuscript_stage24_5/01_FULL_MANUSCRIPT_REPAIRED.md (never modified) and writes
results_v2/manuscript_stage24_6/{01_FULL_MANUSCRIPT_ADMIN_UPDATED.md, 03_ADMINISTRATIVE_PLACEHOLDER_AUDIT.csv}.
Only two regions are replaced: the author block under the title and the administrative statements between the
Conclusions and the References. Everything else is asserted to be byte-identical.
No scientific value is read, computed or changed.
"""
import csv
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results_v2", "manuscript_stage24_5", "01_FULL_MANUSCRIPT_REPAIRED.md")
OUT = os.path.join(ROOT, "results_v2", "manuscript_stage24_6")
os.makedirs(OUT, exist_ok=True)
OLD = open(SRC, encoding="utf-8", newline="").read()
assert "\r" not in OLD

# ---------------------------------------------------------------- supplied administrative texts (verbatim)
HEADER_OLD = "**Authors:** [USER TO COMPLETE — author names, affiliations and ORCID iDs]\n\n**Corresponding author:** [USER TO COMPLETE]\n"
AUTHOR_LINES = ["Md. Mahbubul Islam¹˒²,*", "Ainur Yerkos¹˒²,*", "Nadezhda Kunicina¹˒²"]
AFF = ["¹ Al-Farabi Kazakh National University, Almaty, Kazakhstan",
       "² Institute of Industrial Electronics, Electrical Engineering and Energy, Riga Technical University, Riga, Latvia"]
ORCID = [("Md. Mahbubul Islam", "https://orcid.org/0009-0009-1910-1980"),
         ("Ainur Yerkos", "https://orcid.org/0000-0001-5949-6942"),
         ("Nadezhda Kunicina", "https://orcid.org/0000-0002-0980-0958")]
CORR = [("Ainur Yerkos", "yerkos.ainur@kaznu.kz"), ("Md. Mahbubul Islam", "islam_md_mahbubul@live.kaznu.kz")]
HEADER_NEW = (
    "**Authors:**  \n" + "  \n".join(AUTHOR_LINES) + "\n\n"
    "**Affiliations:**  \n" + "  \n".join(AFF) + "\n\n"
    "**ORCID:**  \n" + "  \n".join(f"{n}: {u}" for n, u in ORCID) + "\n\n"
    "**Correspondence:**  \n" + "  \n".join(f"{n}: {e}" for n, e in CORR) + "  \n\\* Corresponding authors.\n"
)
CONTRIB = ("Md. Mahbubul Islam: Conceptualization, methodology, software, data curation, formal analysis, investigation, validation, "
           "visualization, writing—original draft, and writing—review and editing. Ainur Yerkos: Supervision, conceptual guidance, "
           "methodology review, and writing—review and editing. Nadezhda Kunicina: Supervision, research guidance, and "
           "writing—review and editing. All authors have read and agreed to the published version of the manuscript.")
STATEMENTS = [
    ("Data Availability Statement", "The datasets analyzed in this study are publicly available from their respective original repositories or public data platforms. Dataset-specific access conditions and licensing terms apply."),
    ("Code Availability", "The code repository information will be provided in the final version of the manuscript."),
    ("Author Contributions", CONTRIB),
    ("Funding", "This research received no external funding."),
    ("Ethics Statement", "[TO BE VERIFIED BASED ON DATASET-SPECIFIC REQUIREMENTS]"),
    ("AI Use Disclosure", "[TO BE FINALIZED ACCORDING TO THE JOURNAL'S CURRENT AI-USE POLICY]"),
    ("Conflicts of Interest", "The authors declare no conflicts of interest."),
]
ADMIN_NEW = "\n\n".join(f"**{k}:** {v}" for k, v in STATEMENTS) + "\n\n"

# ---------------------------------------------------------------- locate the two regions
assert OLD.count(HEADER_OLD) == 1
i0 = OLD.index("## Data Availability Statement")
i1 = OLD.index("---\n\n## References")
ADMIN_OLD = OLD[i0:i1]
assert ADMIN_OLD.startswith("## Data Availability Statement") and ADMIN_OLD.rstrip().endswith("[USER TO COMPLETE]")
assert i0 > OLD.index("## 6. Conclusions")
NEW = OLD.replace(HEADER_OLD, HEADER_NEW, 1)
NEW = NEW.replace(ADMIN_OLD, ADMIN_NEW, 1)
# everything outside the two regions must be identical
assert NEW.replace(HEADER_NEW, HEADER_OLD, 1).replace(ADMIN_NEW, ADMIN_OLD, 1) == OLD
open(os.path.join(OUT, "01_FULL_MANUSCRIPT_ADMIN_UPDATED.md"), "w", encoding="utf-8", newline="\n").write(NEW)

# ---------------------------------------------------------------- verification
def orcid_ok(u):
    d = u.rsplit("/", 1)[1].replace("-", "")
    tot = 0
    for c in d[:15]:
        tot = (tot + int(c)) * 2
    r = (12 - tot % 11) % 11
    return d[15] == ("X" if r == 10 else str(r))


checks = []


def chk(desc, ok):
    checks.append((desc, bool(ok)))


for a in AUTHOR_LINES:
    chk(f"author line exact: {a}", a in NEW)
chk("author order preserved", NEW.index(AUTHOR_LINES[0]) < NEW.index(AUTHOR_LINES[1]) < NEW.index(AUTHOR_LINES[2]))
chk("affiliations exact", all(a in NEW for a in AFF))
chk("ORCID URLs exact and in author order", all(u in NEW for _, u in ORCID) and NEW.index(ORCID[0][1]) < NEW.index(ORCID[1][1]) < NEW.index(ORCID[2][1]))
chk("both correspondence emails present, no others", all(e in NEW for _, e in CORR) and len(re.findall(r"[\w.]+@[\w.]+", NEW)) == 2)
chk("Author Contributions exact", f"**Author Contributions:** {CONTRIB}" in NEW)
chk("Funding exact", "**Funding:** This research received no external funding." in NEW)
chk("Conflicts exact", "**Conflicts of Interest:** The authors declare no conflicts of interest." in NEW)
chk("Data availability exact", f"**Data Availability Statement:** {STATEMENTS[0][1]}" in NEW)
chk("Code availability exact; no URL", f"**Code Availability:** {STATEMENTS[1][1]}" in NEW and not re.search(r"github|https?://(?!orcid)", NEW[NEW.index("**Code Availability:**"):NEW.index("**Author Contributions:**")], re.I))
chk("Ethics placeholder exact", "**Ethics Statement:** [TO BE VERIFIED BASED ON DATASET-SPECIFIC REQUIREMENTS]" in NEW)
chk("AI placeholder exact", "**AI Use Disclosure:** [TO BE FINALIZED ACCORDING TO THE JOURNAL'S CURRENT AI-USE POLICY]" in NEW)
chk("no invented approval number, grant number, access date or dataset licence in the administrative block", not re.search(r"approval|grant|accessed|CC BY|licensed under", ADMIN_NEW + HEADER_NEW, re.I))
chk("old placeholders removed from the administrative regions", "[USER TO COMPLETE" not in NEW and "[REPOSITORY URL" not in NEW and "[PUBLIC REPOSITORY" not in NEW)
chk("ORCID checksums (ISO 7064 mod 11-2) valid", all(orcid_ok(u) for _, u in ORCID))
head_old_rest = OLD[len(HEADER_OLD):]
chk("text between the author block and the administrative block is byte-identical to Stage 24.5", NEW[NEW.index("---\n\n## Abstract"):NEW.index("**Data Availability Statement:**")] == OLD[OLD.index("---\n\n## Abstract"):i0])
chk("References section byte-identical", NEW[NEW.index("## References"):] == OLD[OLD.index("## References"):])
num = lambda s: sorted(re.findall(r"[−-]?\d[\d,]*\.?\d*", s))  # noqa: E731
sci_old = OLD[OLD.index("## Abstract"):i0]
sci_new = NEW[NEW.index("## Abstract"):NEW.index("**Data Availability Statement:**")]
chk("scientific body (Abstract to Conclusions): identical text and identical numeric tokens", sci_old == sci_new and num(sci_old) == num(sci_new))
chk("Stage 24.5 manuscript unchanged on disk", hashlib.md5(open(SRC, "rb").read()).hexdigest() == "fd8d446bc3482d544589ef0421f07c94")

# ---------------------------------------------------------------- 03 placeholder audit
MK = {"[DATASET LICENSE TO VERIFY]": NEW.count("[DATASET LICENSE TO VERIFY]"), "[AUTHOR LIST TO COMPLETE PER JOURNAL STYLE]": NEW.count("[AUTHOR LIST TO COMPLETE PER JOURNAL STYLE]"),
      "[TO BE VERIFIED BASED ON DATASET-SPECIFIC REQUIREMENTS]": NEW.count("[TO BE VERIFIED BASED ON DATASET-SPECIFIC REQUIREMENTS]"),
      "[TO BE FINALIZED ACCORDING TO THE JOURNAL'S CURRENT AI-USE POLICY]": NEW.count("[TO BE FINALIZED ACCORDING TO THE JOURNAL'S CURRENT AI-USE POLICY]")}
loc32 = "Section 3.2 (GlaS, CRAG and PANDA paragraphs)"
ROWS = [
    ("authors", "COMPLETE", "header (Authors)", "none", "Stage 24.6 author-supplied information"),
    ("affiliations", "COMPLETE", "header (Affiliations)", "none; no additional affiliations were added", "Stage 24.6 author-supplied information"),
    ("ORCID", "COMPLETE", "header (ORCID)", "none; identifiers reproduced as supplied (format check digits valid; not looked up online)", "Stage 24.6 author-supplied information"),
    ("corresponding authors", "JOURNAL POLICY VERIFICATION REQUIRED", "header (Correspondence; asterisks on two authors)", "Corresponding-author designation requires author confirmation. Both supplied emails are kept; whether the journal format accepts two corresponding authors is unverified", "Stage 24.6 author-supplied information; journal rule unverified"),
    ("author contributions", "COMPLETE", "Author Contributions", "none; text reproduced exactly as supplied", "Stage 24.6 author-supplied information"),
    ("funding", "COMPLETE", "Funding", "none", "Stage 24.6 author-supplied information"),
    ("conflicts of interest", "COMPLETE", "Conflicts of Interest", "none", "Stage 24.6 author-supplied information"),
    ("data availability", "COMPLETE", "Data Availability Statement", "statement is final as supplied; dataset-specific licence verification is tracked in the 'dataset licenses' row", "Stage 24.6 author-supplied information"),
    ("code availability", "USER VERIFICATION REQUIRED", "Code Availability", "Intentionally deferred: repository not yet created; provide repository information (URL/DOI, release, licence) at final submission. Section 3.9 still says the criteria file is 'available in the project repository' and must be reconciled then", "Stage 24.6 author instruction"),
    ("ethics", "USER VERIFICATION REQUIRED", "Ethics Statement", f"Verify dataset-specific requirements (GlaS, CRAG, PANDA) and replace the placeholder; do not state an approval number or an exemption unless documented. Informed-consent wording is covered by the same verification", "Stage 24.6 author instruction"),
    ("AI-use disclosure", "JOURNAL POLICY VERIFICATION REQUIRED", "AI Use Disclosure", "Finalize wording according to the journal's current AI-use policy; the placeholder makes no claim about AI", "Stage 24.6 author instruction; journal policy unverified"),
    ("repository", "USER VERIFICATION REQUIRED", "Code Availability", "Create the repository at the final submission stage; no URL has been invented; decide whether derived artifacts are shared with it", "Stage 24.6 author instruction"),
    ("dataset licenses", "USER VERIFICATION REQUIRED", loc32, f"Verify GlaS, CRAG and PANDA licences and terms of use on the official pages ({MK['[DATASET LICENSE TO VERIFY]']} markers remain); do not assume identical terms or infer from mirrors", "Stage 24.5 manuscript; Stage 24.6 author instruction"),
    ("additional: reference [12] author list", "USER VERIFICATION REQUIRED", "References, entry 12", "Complete the PANDA author list per journal style (marker retained; scientific reference not altered in Stage 24.6)", "Stage 24.5 reference audit"),
    ("additional: 'project repository' wording in Methods", "USER VERIFICATION REQUIRED", "Section 3.9 (criteria file)", "Reconcile with the final repository information when it exists; Methods text was not altered in Stage 24.6", "Stage 24.6 consistency note"),
]
with open(os.path.join(OUT, "03_ADMINISTRATIVE_PLACEHOLDER_AUDIT.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["item", "status", "manuscript_location", "action_required", "source"])
    w.writerows(ROWS)

print(f"{'RESULT':7s} CHECK")
for d, ok in checks:
    print(f"{'PASS' if ok else 'FAIL':7s} {d}")
print("passed:", sum(ok for _, ok in checks), "of", len(checks))
print("placeholder markers remaining:", MK)
print("audit rows:", len(ROWS))
wc = lambda s: len([x for x in re.findall(r"\S+", s) if not re.fullmatch(r"[|\-:]+", x)])  # noqa: E731
print("words old/new (body to References):", wc(OLD[:OLD.index('## References')]), wc(NEW[:NEW.index('## References')]))
sys.exit(0 if all(ok for _, ok in checks) else 1)
