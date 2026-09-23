"""
STAGE 27 TASK 2: patient-level sensitivity analysis.

Determines whether a patient-disjoint recreation of the GlaS canonical split is feasible, using the
patient IDs in the original GlaS archive (data/glas/Warwick_QU_Dataset/Grade.csv). Read-only with
respect to the canonical manifest and all frozen artifacts; writes only into
results_v2/stage27_robustness/patient_sensitivity/.
"""
import json
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results_v2", "stage27_robustness", "patient_sensitivity")
os.makedirs(OUT, exist_ok=True)

manifest = pd.read_csv(os.path.join(ROOT, "data_manifests", "glas_manifest_stage3_finalized.csv"))
grade = pd.read_csv(os.path.join(ROOT, "data", "glas", "Warwick_QU_Dataset", "Grade.csv"))
grade.columns = [c.strip() for c in grade.columns]
grade = grade.rename(columns={"name": "image_id", "patient ID": "patient_id"})

m = manifest.merge(grade[["image_id", "patient_id"]], on="image_id", how="left")
assert m["patient_id"].notna().all(), "some images have no patient ID"
m["patient_id"] = m["patient_id"].astype(int)

# ---------------------------------------------------------------- 1. current (canonical) patient/split table
by_patient = m.groupby("patient_id")["canonical_split"].apply(lambda s: sorted(s.unique())).reset_index()
by_patient["n_images"] = m.groupby("patient_id").size().values
by_patient["splits_touched"] = by_patient["canonical_split"].apply(len)
by_patient.to_csv(os.path.join(OUT, "patient_split_table.csv"), index=False)

n_patients = m["patient_id"].nunique()
multi_split_patients = by_patient[by_patient["splits_touched"] > 1]["patient_id"].tolist()
heldout_patients = sorted(m[m.canonical_split.isin(["testA", "testB"])]["patient_id"].unique().tolist())
heldout_only_patients = [p for p in heldout_patients if set(by_patient.loc[by_patient.patient_id == p, "canonical_split"].iloc[0]) == {"testA"} or
                          set(by_patient.loc[by_patient.patient_id == p, "canonical_split"].iloc[0]) == {"testB"}]
train_touching_heldout = [p for p in heldout_patients if "train" in by_patient.loc[by_patient.patient_id == p, "canonical_split"].iloc[0]]

# ---------------------------------------------------------------- 2. feasibility of a strictly patient-disjoint recreation
# Strategy: assign every patient wholly to either TRAIN or HELD-OUT (never both), by majority image count in
# their current canonical split (ties -> held-out, to keep the held-out evaluation population from shrinking
# further than necessary). This is the least-disruptive patient-disjoint partition of the SAME 165 images
# (no new data, no re-annotation); it is not the only possible one, and it is not claimed to be optimal.
patient_assign = {}
for pid, grp in m.groupby("patient_id"):
    counts = grp["canonical_split"].value_counts()
    train_n = counts.get("train", 0)
    held_n = counts.get("testA", 0) + counts.get("testB", 0)
    patient_assign[pid] = "train" if train_n > held_n else "held_out"

m["patient_disjoint_group"] = m["patient_id"].map(patient_assign)
disjoint_train = m[m.patient_disjoint_group == "train"]
disjoint_held = m[m.patient_disjoint_group == "held_out"]

moved_from_train_to_held = m[(m.canonical_split == "train") & (m.patient_disjoint_group == "held_out")]
moved_from_held_to_train = m[(m.canonical_split.isin(["testA", "testB"])) & (m.patient_disjoint_group == "train")]

summary = {
    "n_images_total": int(len(m)),
    "n_patients_total": int(n_patients),
    "n_patients_multi_split": int(len(multi_split_patients)),
    "multi_split_patient_ids": multi_split_patients,
    "n_heldout_patients_canonical": int(len(heldout_patients)),
    "n_heldout_patients_also_in_train": int(len(train_touching_heldout)),
    "patient_disjoint_partition": {
        "rule": "each patient assigned wholly to train or held-out by majority image count in the canonical split; ties assigned to held-out",
        "train_images": int(len(disjoint_train)),
        "held_out_images": int(len(disjoint_held)),
        "held_out_testA_like": None,
        "images_moved_train_to_held": int(len(moved_from_train_to_held)),
        "images_moved_held_to_train": int(len(moved_from_held_to_train)),
        "moved_train_to_held_image_ids": moved_from_train_to_held["image_id"].tolist(),
        "moved_held_to_train_image_ids": moved_from_held_to_train["image_id"].tolist(),
        "net_train_size_change": int(len(disjoint_train)) - int((m.canonical_split == "train").sum()),
        "net_heldout_size_change": int(len(disjoint_held)) - int((m.canonical_split != "train").sum()),
    },
    "feasibility_verdict": None,
}

# Feasibility verdict: a *fully* patient-disjoint reconstruction of the exact 85/60/20 canonical split is
# infeasible without discarding or reassigning images, because 11 of 12 held-out patients contribute training
# images. A patient-disjoint partition of the SAME 165 images (different train/held-out membership) IS feasible
# and is what "disjoint_train"/"disjoint_held" above construct.
summary["feasibility_verdict"] = (
    "A patient-disjoint split that reproduces the exact canonical 85/60/20 image-level partition is NOT feasible: "
    f"{len(train_touching_heldout)} of {len(heldout_patients)} held-out patients also contribute training images, "
    "so keeping the canonical held-out image set fixed and removing patient overlap would require deleting "
    f"{len(moved_from_train_to_held)} of the 85 canonical training images (those images cannot be kept in training "
    "without violating patient-disjointness), leaving only "
    f"{int((m.canonical_split=='train').sum()) - len(moved_from_train_to_held)} training images. "
    "A DIFFERENT patient-disjoint partition of the same 165 images IS feasible (constructed above) and is used "
    "for the sensitivity retraining in this stage; it is not a recreation of the canonical benchmark and must not "
    "be presented as one."
)

with open(os.path.join(OUT, "patient_split_feasibility.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ---------------------------------------------------------------- 3. write the patient-disjoint manifest for retraining
disjoint_manifest = m.copy()
disjoint_manifest["patient_disjoint_split"] = disjoint_manifest["patient_disjoint_group"].map(
    {"train": "train", "held_out": "held_out"})
disjoint_manifest.to_csv(os.path.join(OUT, "glas_manifest_patient_disjoint.csv"), index=False)

print(json.dumps(summary, indent=2))
print(f"\n[stage27-task2] patient-disjoint partition: {len(disjoint_train)} train images "
      f"(vs. 85 canonical), {len(disjoint_held)} held-out images (vs. 80 canonical)")
