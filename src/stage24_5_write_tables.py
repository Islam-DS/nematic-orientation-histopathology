"""
STAGE 24.5 -- writes the curated tables of the repair package (claim audit, reference verification) and the abstract file.
Documentation only: no evidence is generated or modified. Reads the Stage 24 claim audit (unmodified) and the repaired manuscript.
"""
import csv
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
OUT = P("results_v2", "manuscript_stage24_5")
MS = open(os.path.join(OUT, "01_FULL_MANUSCRIPT_REPAIRED.md"), encoding="utf-8").read()

# ------------------------------------------------------------------ 03 claim audit (repaired)
s24 = list(csv.DictReader(open(P("results_v2", "manuscript_stage24", "07_CLAIM_AUDIT.csv"), encoding="utf-8-sig")))
ISSUE = {"C003": "S24-H02", "C004": "S24-H01", "C005": "S24-H01/H02", "C007": "S24-L01", "C013": "S24-H01/H02", "C014": "S24-H01/H02", "C015": "S24-H01/H02",
         "C017": "S24-M11", "C018": "S24-H02", "C019": "S24-M05", "C030": "S24-M09", "C034": "S24-M10", "C048": "S24-M01", "C049": "S24-M04", "C064": "S24-H08",
         "C082": "S24-M11", "C083": "S24-H07", "C087": "S24-M02", "C088": "S24-C-optional", "C095": "S24-M03", "U01": "S24-M10", "U02": "S24-H03", "U03": "S24-M01",
         "U04": "S24-H02", "U05": "S24-M11", "U06": "optional", "U07": "S24-L05", "U08": "S24-L05"}
REPAIR = {
    "C003": ("The abstract and the Introduction make no statement about the state of the wider literature.", "RESOLVED (sentence removed)", "none"),
    "C004": ("Group-equivariant [1] and steerable [2,3] networks provide layers whose feature maps transform according to a chosen group representation.", "RESOLVED (references verified at abstract level)", "none"),
    "C005": ("Rotation-equivariant networks have been applied to digital pathology [4,5]: [4] tumor detection on a lymph-node metastasis dataset; [5] mitosis detection, nuclei segmentation and tumor classification (as stated in their abstracts).", "RESOLVED (abstract-level source)", "none"),
    "C007": ("H1–H3 are quoted verbatim; the one paragraph break inside H2 is marked in brackets instead of an inserted period.", "RESOLVED", "none"),
    "C013": ("Navarro and Wilkinson [9] (arXiv preprint, 26 May 2026): group-equivariant networks for the two-dimensional Q-tensor order parameter, cyclic groups C_k (k = 4 to 256), synthetic microscopic textures; the present study uses D8.", "RESOLVED (title, authors and description verified on the arXiv abstract page)", "none"),
    "C014": ("SIRE [10] (Alblas et al., arXiv preprint): SO(3)-equivariant estimation of artery orientations in 3D medical images with approximate scale invariance, evaluated on three datasets; the type of reference orientation is not reported in the abstract and no distinction is drawn on it.", "RESOLVED (independent-reference claim removed)", "none"),
    "C015": ("Snoussi and Karimi [11] (arXiv preprint): rotation-equivariant spherical CNN for fiber orientation distributions in neonatal diffusion MRI, compared with an MLP baseline; the type of reference is not reported in the abstract.", "RESOLVED (independent-reference claim removed)", "none"),
    "C017": ("Statement removed. Section 2.2 states only that gland segmentation has been the subject of dedicated challenges [7,8] and that the literature on gland-orientation descriptors was not reviewed.", "RESOLVED (statement removed)", "none"),
    "C018": ("Table 1 lists, for [4], [5], [9], [10], [11], what the abstracts report (equivariance/symmetry, predicted quantity, data, evaluation); 'not reported' where an abstract is silent.", "RESOLVED (every cell verified against the abstract or marked 'not reported')", "none"),
    "C019": ("Relevant frozen artifacts (criteria file, analysis matrices, Model 3 checkpoint, result files) were checksum-verified during the reproducibility audit.", "RESOLVED", "none"),
    "C030": ("PANDA: 10,616 slides, 5,160 Radboud, 10-case pilot (counts unchanged). License and terms of use: [DATASET LICENSE TO VERIFY] for GlaS, CRAG and PANDA.", "PARTLY RESOLVED (counts supported; licenses USER TO VERIFY)", "USER TO VERIFY licenses on the official data pages"),
    "C034": ("Q_anat is an independently constructed, mask-derived reference; independent of the image pixel values but not of the target definition, annotation, Q construction or training objective.", "RESOLVED (5.5 wording aligned)", "none"),
    "C048": ("Calibration values (Δ = 0.152; r = 0.0115 on 2249 training patches) were used only to derive the frozen thresholds, are not held-out model results and are not directly comparable with the Model 3 results.", "RESOLVED", "none"),
    "C049": ("Rotation protocol as before, plus: the comparison is pointwise (equal to the field-level relation only at the fixed point); sign convention: skimage rotate with positive angle argument (convention of the earlier synthetic benchmark), x = column, y = row, expected output R(2α) applied to (q1, q2); the structure-tensor estimator used the opposite sign in its own calibration.", "RESOLVED (documented convention reproduced from the Stage 12 report and code)", "none"),
    "C064": ("Values unchanged. Split-wise values differ in size for all three models; signs of Glass's Δ (A), r and Glass's Δ (C) are the same in both splits for the plain CNN and Model 2 and differ only for Model 3.", "RESOLVED (sentence verified against the frozen baseline values)", "none"),
    "C082": ("Statement removed; the annotation-compatibility finding rests on the measured component-size ratios only.", "RESOLVED (statement removed)", "none"),
    "C083": ("Figure 7 caption follows the frozen summary table: three pilot cases (00743313, 004dd32d, 01642d24) contain no Gleason 3/4/5 pixels. An earlier documentation record says two; see 11_DOCUMENTATION_ERRATA.md.", "RESOLVED (manuscript follows the frozen source; discrepancy documented, historical records unchanged)", "none"),
    "C087": ("Glass's Δ stayed between −0.0441 and 0.0461 across ablation conditions and scales; r varied across scales (−0.0341 to 0.0147) and more widely across ablation conditions (−0.2097 to 0.2203), none reaching 0.30.", "RESOLVED (ranges read from frozen values)", "none"),
    "C088": ("The small p-values for Criteria A and C are consistent with the large number of overlapping patches and are not evidence of agreement.", "RESOLVED", "none"),
    "C095": ("The image-level split permits patient overlap between training and held-out images, introducing dependence that may affect apparent performance estimates; direction and size not analyzed; not offered as an explanation of the negative result.", "RESOLVED", "none"),
    "U01": ("The reference is computed from gland-instance annotations; [8] describes the GlaS/CRAG boundaries as annotated by an expert pathologist (attributed, in Section 3.2 only).", "RESOLVED", "none"),
    "U02": ("Patch-level S and φ are the norm and half-angle of the masked mean of (q1, q2) over the valid cells (Section 3.9); the training objective operates on the dense field.", "RESOLVED (frozen implementation rule stated; not claimed optimal)", "none"),
    "U03": ("Statement deleted.", "RESOLVED (statement removed)", "none"),
    "U04": ("We have not established whether any equivariant histopathology study has evaluated a rank-2 orientation output against an anatomical reference, and make no claim on that point.", "RESOLVED (unsupported statement replaced)", "none"),
    "U05": ("In this study, gland size, count and elongation are computed from the masks (Sections 3.3, 3.12).", "RESOLVED (narrowed to what this study does)", "none"),
    "U06": ("Unchanged (optional).", "UNCHANGED (optional, class C)", "none"),
    "U07": ("The permutation p-value is consistent with the sample size; the difference between the two mean errors is about 1.1°; it does not indicate agreement.", "RESOLVED", "none"),
    "U08": ("'developed and evaluated'.", "RESOLVED", "none"),
}
NEW = [
    ("N01", "3.9", "Patch-level (S, φ) rule for Criteria A–C", "code_v2/stage9_train_baselines.py::per_patch_scalar_qs; code_v2/stage11_geometric_validation.py (docstring)", "Norm and half-angle of the masked mean of (q1, q2) over valid cells, for prediction and reference; not claimed optimal.", "LOW", "YES (code read; wording reproduces the frozen rule)"),
    ("N02", "3.10", "Rotation sign convention", "reports/STAGE12_AUDIT_REPORT.md (line 41); code_v2/stage12_rotation_validation.py; code_v2/nematic_math.py::rotate_Q_analytic; code_v2/run_synthetic_benchmark.py", "skimage rotate with positive angle argument; x = column, y = row; expected (q1', q2') = R(2α)(q1, q2).", "LOW", "YES (code and report read)"),
    ("N03", "2.1; 2.3; Table 1", "Descriptions of [4], [5], [9], [10], [11]", "arXiv abstract pages (fetched 2026-09-21): 1806.03962, 2002.08725, 2605.27679, 2311.05400, 2504.01925", "Only what the abstracts state; 'not reported' otherwise; full texts not read.", "LOW", "YES at abstract level"),
    ("N04", "3.2", "CRAG description per [8]: colorectal adenocarcinoma gland dataset; 213 images; 173/40 split; 38 whole-slide images of different patients; UHCW NHS Trust; expert-pathologist annotation", "Graham et al., Med Image Anal 2019, 52:199-211 (arXiv:1806.01963 PDF, read)", "Statements attributed to [8].", "LOW", "YES (primary text read)"),
    ("N05", "3.11; 4.4", "A7 evidence = plain CNN baseline (64-channel blocks) + A4", "configs/stage14_ablation_matrix.json (A7 resolution)", "A7 reused two existing artifacts without new training; descriptive only.", "LOW", "YES"),
    ("N06", "3.15", "Holm–Bonferroni applied only to P1–P4; other analyses descriptive", "Stage 15 design; manuscript Table 8", "No inference is drawn from p-values of other analyses.", "LOW", "YES"),
    ("N07", "5.2", "Ranges of Glass's Δ and r across ablations and scales", "results_v2/validation/stage14/stage14_all_evaluations.json; results_v2/validation/stage16/stage16_results.json", "As printed; min/max of frozen values.", "LOW", "YES (04 N-rows)"),
    ("N08", "4.2", "Sign behaviour across splits (Table 5)", "results_v2/baselines/*/held_out_test*_results.json; stage11_results.json", "Only Model 3 changes sign; sizes differ for all three models.", "LOW", "YES (04 N-rows)"),
    ("N09", "Figure captions 1, 3, 4, 5, 6, 7", "Descriptions of what each graphic shows", "results_v2/figures/stage20/*.png (viewed)", "Captions describe the graphic; embedded internal labels recorded in 06 for a future formatting pass.", "LOW", "YES (images inspected)"),
    ("N10", "4.8 Figure 7", "Case IDs without Gleason 3/4/5 pixels", "results_v2/validation/stage18_panda/summary_table.csv", "00743313, 004dd32d, 01642d24 (three).", "LOW", "YES (04 N-rows)"),
    ("N11", "Table 10", "Outcome vocabulary vs supplementary S2", "results_v2/tables/stage20_corrected/supplementary_evidence_domain_summary_corrected.csv", "'not met'/'not demonstrated' = analysis did not provide supporting evidence ('NOT SUPPORTED' in S2).", "LOW", "YES"),
]
fields = ["claim_id", "claim", "section", "source", "allowed_wording", "risk_level", "verified", "stage24_status", "stage24_issue", "stage24_5_status", "stage24_5_action"]
rows = []
for r in s24:
    cid = r["claim_id"]
    src = (r["stage23_source_file"] + (" ; " + r["stage23_source_location"] if r["stage23_source_location"] else "")).strip(" ;")
    if cid in REPAIR:
        aw, st, act = REPAIR[cid]
        rows.append({"claim_id": cid, "claim": r["claim"], "section": r["manuscript_section"], "source": src, "allowed_wording": aw,
                     "risk_level": "LOW (after repair)" if "USER" not in st else "MODERATE (license unverified)",
                     "verified": "USER TO VERIFY" if "USER" in st else "YES", "stage24_status": r["stage24_status"], "stage24_issue": ISSUE.get(cid, ""),
                     "stage24_5_status": st, "stage24_5_action": act})
    else:
        rows.append({"claim_id": cid, "claim": r["claim"], "section": r["manuscript_section"], "source": src, "allowed_wording": r["claim"],
                     "risk_level": r["stage23_risk_level"], "verified": "YES (Stage 24 check; unchanged text)", "stage24_status": r["stage24_status"],
                     "stage24_issue": "", "stage24_5_status": "SUPPORTED (unchanged)", "stage24_5_action": "none"})
for cid, sec, claim, src, aw, risk, ver in NEW:
    rows.append({"claim_id": cid, "claim": claim, "section": sec, "source": src, "allowed_wording": aw, "risk_level": risk, "verified": ver,
                 "stage24_status": "(new in Stage 24.5)", "stage24_issue": "", "stage24_5_status": "NEW CLAIM, SUPPORTED", "stage24_5_action": "none"})
with open(os.path.join(OUT, "03_CLAIM_EVIDENCE_AUDIT_REPAIRED.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("03 rows:", len(rows), "| repaired:", len(REPAIR), "| new:", len(NEW))

# ------------------------------------------------------------------ 05 reference verification (final)
RF = ["reference_number", "citation", "title", "authors", "venue", "year", "doi_or_identifier", "verification_source", "verified", "action", "notes"]
REFS = [
    ("1", "Cohen, T.S.; Welling, M. Group Equivariant Convolutional Networks. ICML 2016; PMLR 48:2990–2999.", "Group Equivariant Convolutional Networks", "Taco S. Cohen; Max Welling", "Proceedings of the 33rd International Conference on Machine Learning (ICML 2016); PMLR volume 48, pp. 2990–2999", "2016", "arXiv:1602.07576",
     "arXiv abstract page; PMLR proceedings page (proceedings.mlr.press/v48/cohenc16.html)", "YES", "KEEP", "Was [REFERENCE DETAIL TO VERIFY] in Stage 23. Cited in Sections 1 and 2.1; the statement (layers with weight sharing over discrete groups) matches the abstract."),
    ("2", "Weiler, M.; Cesa, G. General E(2)-Equivariant Steerable CNNs. NeurIPS 2019.", "General E(2)-Equivariant Steerable CNNs", "Maurice Weiler; Gabriele Cesa", "Advances in Neural Information Processing Systems 32 (NeurIPS 2019)", "2019", "arXiv:1911.08251",
     "arXiv abstract page; NeurIPS proceedings page", "YES", "KEEP", "Page numbers not shown on the sources and not printed. Cited in Sections 1 and 2.1."),
    ("3", "Cesa, G.; Lang, L.; Weiler, M. A Program to Build E(N)-Equivariant Steerable CNNs. ICLR 2022; escnn software.", "A Program to Build E(N)-Equivariant Steerable CNNs", "Gabriele Cesa; Leon Lang; Maurice Weiler", "International Conference on Learning Representations (ICLR 2022)", "2022", "https://github.com/QUVA-Lab/escnn",
     "Official escnn repository README (citation section)", "YES (title, authors, venue, year); OpenReview page not retrievable (bot check)", "KEEP", "Software version 1.0.11 is taken from the project environment, not from a release page (USER may confirm). No DOI or arXiv identifier printed."),
    ("4", "Veeling, B.S.; Linmans, J.; Winkens, J.; Cohen, T.; Welling, M. Rotation Equivariant CNNs for Digital Pathology. MICCAI 2018, pp. 210–218.", "Rotation Equivariant CNNs for Digital Pathology", "Bastiaan S. Veeling; Jasper Linmans; Jim Winkens; Taco Cohen; Max Welling", "Medical Image Computing and Computer Assisted Intervention – MICCAI 2018, Lecture Notes in Computer Science (Springer), pp. 210–218", "2018", "doi:10.1007/978-3-030-00934-2_24; arXiv:1806.03962",
     "arXiv abstract page; Crossref record", "YES", "KEEP", "LNCS volume number and place of publication not retrieved and not printed. Manuscript statement restricted to the abstract (tumor detection on a lymph-node metastasis dataset)."),
    ("5", "Lafarge, M.W.; Bekkers, E.J.; Pluim, J.P.W.; Duits, R.; Veta, M. Roto-Translation Equivariant Convolutional Networks: Application to Histopathology Image Analysis. Med. Image Anal. 2021, 68, 101849.", "Roto-Translation Equivariant Convolutional Networks: Application to Histopathology Image Analysis", "Maxime W. Lafarge; Erik J. Bekkers; Josien P.W. Pluim; Remco Duits; Mitko Veta", "Medical Image Analysis, volume 68, article 101849", "2021", "doi:10.1016/j.media.2020.101849; arXiv:2002.08725",
     "arXiv abstract page; Crossref record", "YES", "KEEP", "Manuscript statement restricted to the abstract (mitosis detection, nuclei segmentation, tumor classification)."),
    ("6", "de Gennes, P.G.; Prost, J. The Physics of Liquid Crystals, 2nd ed.; Oxford University Press: Oxford, UK, 1993.", "The Physics of Liquid Crystals", "P.G. de Gennes; J. Prost", "Oxford University Press (Clarendon Press), 2nd edition; International Series of Monographs on Physics 83", "1993", "doi:10.1093/oso/9780198520245.001.0001; ISBN 0-19-852024-7",
     "Crossref record (DOI, publisher, year, ISBN); edition, place and series number from bibliographic listings found by web search (publisher page not retrievable)", "PARTIAL (DOI, publisher, year registry-verified; edition and place from web-search listings)", "KEEP; confirm edition and place on the publisher page", "Series number is not printed in the manuscript."),
    ("7", "Sirinukunwattana, K.; et al. Gland Segmentation in Colon Histology Images: The GlaS Challenge Contest. Med. Image Anal. 2017, 35, 489–502.", "Gland Segmentation in Colon Histology Images: The GlaS Challenge Contest", "Korsuk Sirinukunwattana; Josien P.W. Pluim; Hao Chen; Xiaojuan Qi; Pheng-Ann Heng; Yun Bo Guo; Li Yang Wang; Bogdan J. Matuszewski; Elia Bruni; Urko Sanchez; Anton Böhm; Olaf Ronneberger; Bassem Ben Cheikh; Daniel Racoceanu; Philipp Kainz; Michael Pfeiffer; Martin Urschler; David R.J. Snead; Nasir M. Rajpoot", "Medical Image Analysis, volume 35, pp. 489–502", "2017", "doi:10.1016/j.media.2016.08.008; arXiv:1603.00275",
     "arXiv abstract page (authors); Crossref record (journal, volume, pages, DOI); reference list of [8]", "YES", "KEEP", "GlaS split counts (165 images; 85 training, 80 test with 37 benign and 43 malignant) also agree with the dataset description in [8]."),
    ("8", "Graham, S.; et al. MILD-Net: Minimal Information Loss Dilated Network for Gland Instance Segmentation in Colon Histology Images. Med. Image Anal. 2019, 52, 199–211.", "MILD-Net: Minimal Information Loss Dilated Network for Gland Instance Segmentation in Colon Histology Images", "Simon Graham; Hao Chen; Jevgenij Gamper; Qi Dou; Pheng-Ann Heng; David Snead; Yee Wah Tsang; Nasir Rajpoot", "Medical Image Analysis, volume 52, pp. 199–211", "2019", "doi:10.1016/j.media.2018.12.001; arXiv:1806.01963",
     "arXiv abstract page; Crossref record; full text (arXiv PDF) read for the dataset description", "YES", "KEEP", "Stage 23 text said CRAG 'version 2, released with the MILD-Net study'. The primary text says CRAG was originally used in Awan et al. (2017) and describes it as a second colon adenocarcinoma dataset (213 images, 173/40 split, 38 whole-slide images from different patients, UHCW). The manuscript now says only 'described in the MILD-Net study [8]'."),
    ("9", "Navarro, J.; Wilkinson, M. On the Equivariant Learning of the Q-tensor Order Parameter. arXiv 2026, arXiv:2605.27679.", "On the Equivariant Learning of the Q-tensor Order Parameter", "Julia Navarro; Mark Wilkinson", "arXiv preprint (submitted 26 May 2026)", "2026", "arXiv:2605.27679",
     "arXiv abstract page", "YES (abstract level)", "KEEP; check whether a peer-reviewed version exists at submission time", "Stage 23 title was wrong. Peer-review status not checked. Full text not read; the manuscript describes only the abstract (cyclic groups C_k, k = 4 to 256; synthetic microscopic textures)."),
    ("10", "Alblas, D.; Suk, J.; Brune, C.; Yeung, K.K.; Wolterink, J.M. SIRE: Scale-Invariant, Rotation-Equivariant Estimation of Artery Orientations Using Graph Neural Networks. arXiv 2023, arXiv:2311.05400.", "SIRE: scale-invariant, rotation-equivariant estimation of artery orientations using graph neural networks", "Dieuwertje Alblas; Julian Suk; Christoph Brune; Kak Khee Yeung; Jelmer M. Wolterink", "arXiv preprint (submitted 9 November 2023)", "2023", "arXiv:2311.05400",
     "arXiv abstract page", "YES (abstract level)", "KEEP; check whether a journal version exists at submission time", "Stage 23 entry had no authors or venue. The Stage 23 claim of 'independent anatomical vessel-direction validation' is not stated in the abstract and was removed."),
    ("11", "Snoussi, H.; Karimi, D. Equivariant Spherical CNNs for Accurate Fiber Orientation Distribution Estimation in Neonatal Diffusion MRI with Reduced Acquisition Time. arXiv 2025, arXiv:2504.01925.", "Equivariant Spherical CNNs for Accurate Fiber Orientation Distribution Estimation in Neonatal Diffusion MRI with Reduced Acquisition Time", "Haykel Snoussi; Davood Karimi", "arXiv preprint (submitted 2 April 2025)", "2025", "arXiv:2504.01925",
     "arXiv abstract page", "YES (abstract level)", "KEEP; check whether a journal version exists at submission time", "Stage 23 title was paraphrased. The 'independent reference' claim is not stated in the abstract and was removed."),
    ("12", "Bulten, W.; et al. Artificial Intelligence for Diagnosis and Gleason Grading of Prostate Cancer: The PANDA Challenge. Nat. Med. 2022, 28, 154–163.", "Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge", "Wouter Bulten et al. (120 authors per Crossref; full list not transcribed)", "Nature Medicine, volume 28, issue 1, pp. 154–163", "2022", "doi:10.1038/s41591-021-01620-2",
     "Crossref record (title, journal, volume, issue, pages, year, DOI, author count)", "YES (bibliographic data); author list incomplete", "COMPLETE AUTHOR LIST PER JOURNAL STYLE (USER)", "Marker [AUTHOR LIST TO COMPLETE PER JOURNAL STYLE] kept in the manuscript. Dataset license/terms: [DATASET LICENSE TO VERIFY]."),
    ("13", "Cohen, J. Statistical Power Analysis for the Behavioral Sciences, 2nd ed.; Lawrence Erlbaum Associates: Hillsdale, NJ, USA, 1988.", "Statistical Power Analysis for the Behavioral Sciences", "Jacob Cohen", "Lawrence Erlbaum Associates, Hillsdale, NJ, 2nd edition", "1988", "ISBN 978-0805802832",
     "Publisher and library listings found by web search (Routledge, Biblio, Open Library, WorldCat entries; pages not individually fetched)", "PARTIAL (web-search listings)", "KEEP; confirm on the publisher page", "Used only for the 0.5 medium-effect convention cited for Glass's Δ."),
    ("candidate (not added)", "Awan, R.; Sirinukunwattana, K.; Epstein, D.; Jefferyes, S.; Qidwai, U.; Aftab, Z.; Mujeeb, I.; Snead, D.; Rajpoot, N. Glandular Morphometrics for Objective Grading of Colorectal Adenocarcinoma Histology Images. Sci. Rep. 2017, 7, 16852.", "Glandular Morphometrics for Objective Grading of Colorectal Adenocarcinoma Histology Images", "Ruqayya Awan; Korsuk Sirinukunwattana; David Epstein; Samuel Jefferyes; Uvais Qidwai; Zia Aftab; Imaad Mujeeb; David Snead; Nasir Rajpoot", "Scientific Reports, volume 7, article 16852", "2017", "doi:10.1038/s41598-017-16516-w",
     "Crossref record; reference list of [8]", "YES (bibliographic data)", "OPTIONAL: cite as the original CRAG source (adding it renumbers later references)", "Not added in Stage 24.5: [8] is cited as the description of the dataset used. The title supports only the statement that glandular morphometrics were proposed for objective grading; the paper was not read."),
]
with open(os.path.join(OUT, "05_REFERENCE_VERIFICATION_FINAL.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(RF)
    w.writerows(REFS)
print("05 rows:", len(REFS))

# ------------------------------------------------------------------ 02 abstract
ab = MS[MS.index("## Abstract") + len("## Abstract"): MS.index("**Keywords")].strip()
kw = re.search(r"\*\*Keywords:\*\*[^\n]*", MS).group(0)
labelled = ab
parts = re.split(r"(?<=[.;]) ", ab)
n_words = len(ab.split())
bg = parts[0]
rest = " ".join(parts[1:])
i_res = rest.index("None was met")
i_con = rest.index("Under the tested protocol")
meth, res, con = rest[:i_res].strip(), rest[i_res:i_con].strip(), rest[i_con:].strip()
lab = f"**Background:** {bg} **Methods:** {meth} **Results:** {res} **Conclusions:** {con}"
n_lab = len(lab.replace("**", "").split())
doc = f"""# Repaired abstract (Stage 24.5)

The abstract below is the one used in `01_FULL_MANUSCRIPT_REPAIRED.md`. It replaces the 347-word four-paragraph abstract of Stage 23. No scientific content was changed; every number is unchanged and traceable in `04_NUMBERS_AUDIT_REPAIRED.csv`.

## A. Abstract used in the repaired manuscript (single paragraph, {n_words} words)

{ab}

{kw}

Content check: problem (head–tail-symmetric orientation; rank-2 nematic output; equivariance) · formulation (D8-equivariant network, `D8.irrep(1,2)` Q head) · data (GlaS, 85 training images) · reference (mask-derived gland-elongation orientation reference) · primary evaluation (four prospectively frozen internal criteria on 80 held-out images) · main numerical results (Glass's Δ 0.0412, r −0.0341, Glass's Δ 0.0073, 4 of 9 angles) · secondary results (ablations, scales, pathology, CRAG, PANDA) · negative conclusion with its limits. No sentence implies hypothesis support.

## B. Same text with section labels ({n_lab} words including the four labels)

For use if the journal requires a labelled (structured) abstract.

{lab}

## C. Journal requirement status

CURRENT JOURNAL REQUIREMENT REQUIRES VERIFICATION. The official *Symmetry* instructions page (https://www.mdpi.com/journal/symmetry/instructions), the MDPI layout page (https://www.mdpi.com/authors/layout) and the journal home page returned HTTP 403 when fetched in Stage 24.5 (as in Stage 24). A web search returned inconsistent summaries of the *Symmetry* instructions (about 200 words; "not exceed 250 words"), which are not official text and are not treated as verified. Both A ({n_words} words) and B ({n_lab} words) are under 200 words, so either fits the two figures that were indicated. The choice between a single paragraph and a labelled abstract must be made from the official instructions or the current MDPI template.
"""
open(os.path.join(OUT, "02_ABSTRACT_REPAIRED.md"), "w", encoding="utf-8", newline="\n").write(doc)
print("02 written; abstract words:", n_words, "labelled:", n_lab)
