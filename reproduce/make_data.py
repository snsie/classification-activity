"""Build the three activity datasets, their fixed splits, the dictionary and the manifest.

Cohorts and features follow the M11 lecture app (snsie/classification-app, pipeline/datasets.py):
readmission and diabetes are copied from it; heart disease is a new builder from the UCI
documentation. Differences from the lecture, all presentational:
  * categories are kept as readable text (one column each). The starter one-hot encodes them
    inside every model's pipeline, so the encoding is learned from training rows only;
  * readmission's admission and discharge codes are replaced by their UCI descriptions;
  * diabetes `race` is one text column instead of five 0/1 columns; `sbp` is rounded to 0.01;
  * heart `cp`, `restecg`, `slope` and `thal` are categories, and `ca` is excluded (leaked).

Splits: an 80/20 development/test split, stratified (train_test_split, random_state=0), then
five stratified folds of the development rows (StratifiedKFold(5, shuffle=True, random_state=0)).
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 0
N_FOLDS = 5
TEST_SHARE = 0.2
KEYS = ["readmission", "diabetes", "heart"]

# ---------------------------------------------------------------- dataset definitions
# Each column: name, meaning, units, kind ("number", "0/1" or "category"), categories (for
# "category"), missing (what a blank means) and known (the "When is it known?" entry).
COMMON = [
    {"name": "row_id", "meaning": "Row number in this file", "units": "", "kind": "id", "known": "Assigned by the course",
     "missing": "never blank"},
    {"name": "split", "meaning": "`dev` (development: train and choose) or `test` (locked test set, used once)", "units": "",
     "kind": "id", "known": "Assigned by the course", "missing": "never blank"},
    {"name": "fold", "meaning": "Cross-validation fold 1–5 for development rows; blank for test rows", "units": "",
     "kind": "id", "known": "Assigned by the course", "missing": "blank on test rows"},
]

READMIT_CITE = ("Strack B, DeShazo JP, Gennings C, et al. Impact of HbA1c measurement on hospital readmission rates: "
                "analysis of 70,000 clinical database patient records. BioMed Research International 2014;2014:781670.")
NHANES_CITE = ("Centers for Disease Control and Prevention, National Center for Health Statistics. "
               "National Health and Nutrition Examination Survey 2017–2018.")
HEART_CITE = ("Detrano R, Janosi A, Steinbrunn W, et al. International application of a new probability algorithm "
              "for the diagnosis of coronary artery disease. Am J Cardiol 1989;64(5):304–10.")

DATASETS = {
    "readmission": {
        "name": "30-day readmission",
        "unit": "discharge", "units": "discharges",
        "cohort": "UCI Diabetes 130-US Hospitals 1999–2008: inpatient stays of patients with diabetes. "
                  "First discharge per patient, excluding discharge to hospice or death, then a stratified random "
                  "sample of 20,000 (random_state=0).",
        "job": "Predict readmission within 30 days, at discharge.",
        "teaches": "The accuracy trap",
        "framing": "Results apply to this historical inpatient cohort involving diabetes, not to all discharges.",
        "positive": "Readmitted within 30 days", "negative": "Not readmitted within 30 days",
        "prediction_point": "At discharge",
        "source": "UCI Machine Learning Repository: Diabetes 130-US Hospitals for Years 1999–2008",
        "url": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
        "citation": READMIT_CITE, "licence": "CC BY 4.0",
        "one_per_patient": True,
        "columns": [
            {"name": "age", "meaning": "Age band midpoint (the file gives 10-year bands, e.g. [70-80) → 75)", "units": "years", "kind": "number", "known": "At admission"},
            {"name": "female", "meaning": "1 = female, 0 = male", "units": "0/1", "kind": "0/1", "known": "At admission"},
            {"name": "time_in_hospital", "meaning": "Length of stay", "units": "days", "kind": "number", "known": "At discharge"},
            {"name": "num_lab_procedures", "meaning": "Lab tests during the stay", "units": "count", "kind": "number", "known": "At discharge (during the stay)"},
            {"name": "num_procedures", "meaning": "Procedures other than lab tests during the stay", "units": "count", "kind": "number", "known": "At discharge (during the stay)"},
            {"name": "num_medications", "meaning": "Distinct medications given during the stay", "units": "count", "kind": "number", "known": "At discharge (during the stay)"},
            {"name": "number_outpatient", "meaning": "Outpatient visits in the year before this stay", "units": "count", "kind": "number", "known": "At admission (past year)"},
            {"name": "number_emergency", "meaning": "Emergency visits in the year before this stay", "units": "count", "kind": "number", "known": "At admission (past year)"},
            {"name": "number_inpatient", "meaning": "Inpatient admissions in the year before this stay", "units": "count", "kind": "number", "known": "At admission (past year)"},
            {"name": "number_diagnoses", "meaning": "Diagnoses recorded for this stay", "units": "count", "kind": "number", "known": "At discharge"},
            {"name": "race", "meaning": "Race as recorded by the hospital", "units": "", "kind": "category", "known": "At admission"},
            {"name": "admission_type", "meaning": "Admission type", "units": "", "kind": "category", "known": "At admission"},
            {"name": "admission_source", "meaning": "Where the patient was admitted from", "units": "", "kind": "category", "known": "At admission"},
            {"name": "discharge_disposition", "meaning": "Where the patient was discharged to (hospice and death excluded)", "units": "", "kind": "category", "known": "At discharge"},
            {"name": "a1c_result", "meaning": "HbA1c test result during the stay (`None` = not measured)", "units": "", "kind": "category", "known": "At discharge (during the stay)"},
            {"name": "max_glu_serum", "meaning": "Highest serum glucose result during the stay (`None` = not measured)", "units": "mg/dL band", "kind": "category", "known": "At discharge (during the stay)"},
            {"name": "insulin", "meaning": "Insulin during the stay: not prescribed, or dose steady, up or down", "units": "", "kind": "category", "known": "At discharge (during the stay)"},
            {"name": "med_change", "meaning": "1 = any diabetes medication changed during the stay", "units": "0/1", "kind": "0/1", "known": "At discharge"},
            {"name": "diabetes_med", "meaning": "1 = any diabetes medication prescribed", "units": "0/1", "kind": "0/1", "known": "At discharge"},
            {"name": "diag1_group", "meaning": "Primary diagnosis, in Strack et al.'s nine groups of ICD-9 codes", "units": "", "kind": "category", "known": "At discharge"},
        ],
        "target": {"meaning": "1 = readmitted within 30 days of discharge, 0 = not (readmitted after 30 days, or not at all)",
                   "known": "30 days after discharge (the outcome)"},
        "excluded": [],
    },
    "diabetes": {
        "name": "Diabetes screening",
        "unit": "adult", "units": "adults",
        "cohort": "NHANES 2017–2018: adults (18+) with an HbA1c measurement who answered “Has a doctor ever told you that "
                  "you have diabetes?” with yes, no or borderline.",
        "job": "Flag adults for follow-up testing; finding positives matters most.",
        "teaches": "Choosing a threshold for a sensitivity target",
        "framing": "The label is “reported doctor diagnosis or HbA1c ≥ 6.5%”. It includes adults who were already diagnosed. "
                   "It identifies the label at one time point and does not predict future diabetes. Results describe adults in this sample.",
        "positive": "Diabetes (reported doctor diagnosis or HbA1c ≥ 6.5%)", "negative": "No diabetes by this label",
        "prediction_point": "At the survey examination, before the blood test results",
        "source": "CDC/NCHS National Health and Nutrition Examination Survey (NHANES) 2017–2018, public-use files",
        "url": "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017",
        "citation": NHANES_CITE, "licence": "Public domain",
        "one_per_patient": True,
        "columns": [
            {"name": "age", "meaning": "Age (80 means 80 or older)", "units": "years", "kind": "number", "known": "At the interview"},
            {"name": "female", "meaning": "1 = female, 0 = male", "units": "0/1", "kind": "0/1", "known": "At the interview"},
            {"name": "bmi", "meaning": "Body mass index", "units": "kg/m²", "kind": "number", "known": "At the examination"},
            {"name": "waist", "meaning": "Waist circumference", "units": "cm", "kind": "number", "known": "At the examination"},
            {"name": "sbp", "meaning": "Systolic blood pressure, mean of up to 3 readings", "units": "mmHg", "kind": "number", "known": "At the examination"},
            {"name": "income_ratio", "meaning": "Family income ÷ the poverty line (5 means 5 or more)", "units": "ratio", "kind": "number", "known": "At the interview"},
            {"name": "race", "meaning": "Race/ethnicity (self-reported)", "units": "", "kind": "category", "known": "At the interview"},
            {"name": "hdl", "meaning": "HDL cholesterol", "units": "mg/dL", "kind": "number", "known": "From the blood sample taken at the examination"},
            {"name": "total_chol", "meaning": "Total cholesterol", "units": "mg/dL", "kind": "number", "known": "From the blood sample taken at the examination"},
        ],
        "target": {"meaning": "1 = told by a doctor they have diabetes, or HbA1c ≥ 6.5%; 0 = neither",
                   "known": "From the interview and the HbA1c result (the outcome)"},
        "excluded": [
            {"name": "hba1c", "meaning": "HbA1c (%)", "known": "From the same blood test that defines the label (HbA1c ≥ 6.5% is positive)"},
            {"name": "told_diabetes", "meaning": "“Has a doctor ever told you that you have diabetes?” (DIQ010)", "known": "At the interview; a yes defines the label"},
            {"name": "insulin", "meaning": "Now taking insulin (DIQ050)", "known": "At the interview; only people already treated for diabetes answer yes"},
            {"name": "pills", "meaning": "Now taking diabetes pills (DIQ070)", "known": "At the interview; only people already treated for diabetes answer yes"},
        ],
    },
    "heart": {
        "name": "Heart disease",
        "unit": "patient", "units": "patients",
        "cohort": "UCI Heart Disease, Cleveland Clinic Foundation: patients referred for coronary angiography (processed file, 14 attributes).",
        "job": "Classify disease within a diagnostic-workup cohort.",
        "teaches": "Small samples: wide fold SD and a noisy test set",
        "framing": "This is a selected cohort already in diagnostic workup, not the general population.",
        "positive": "Angiographic heart disease (any major vessel with > 50% narrowing; label > 0)",
        "negative": "No angiographic disease (label 0)",
        "prediction_point": "After the clinical and noninvasive tests, before angiography",
        "source": "UCI Machine Learning Repository: Heart Disease (Cleveland), processed.cleveland.data",
        "url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "citation": HEART_CITE, "licence": "CC BY 4.0",
        "one_per_patient": True,
        "columns": [
            {"name": "age", "meaning": "Age", "units": "years", "kind": "number", "known": "At the visit"},
            {"name": "male", "meaning": "1 = male, 0 = female", "units": "0/1", "kind": "0/1", "known": "At the visit"},
            {"name": "chest_pain", "meaning": "Chest pain type", "units": "", "kind": "category", "known": "At the visit (history)"},
            {"name": "rest_bp", "meaning": "Resting systolic blood pressure on admission", "units": "mmHg", "kind": "number", "known": "At the visit"},
            {"name": "chol", "meaning": "Serum cholesterol", "units": "mg/dL", "kind": "number", "known": "From a blood test before angiography"},
            {"name": "high_fasting_glucose", "meaning": "1 = fasting blood sugar > 120 mg/dL", "units": "0/1", "kind": "0/1", "known": "From a blood test before angiography"},
            {"name": "rest_ecg", "meaning": "Resting ECG result", "units": "", "kind": "category", "known": "At the visit"},
            {"name": "max_hr", "meaning": "Maximum heart rate on the exercise test", "units": "beats/min", "kind": "number", "known": "Exercise test, before angiography"},
            {"name": "exercise_angina", "meaning": "1 = chest pain brought on by the exercise test", "units": "0/1", "kind": "0/1", "known": "Exercise test, before angiography"},
            {"name": "st_depression", "meaning": "ST depression on exercise, relative to rest", "units": "mm", "kind": "number", "known": "Exercise test, before angiography"},
            {"name": "st_slope", "meaning": "Slope of the peak-exercise ST segment", "units": "", "kind": "category", "known": "Exercise test, before angiography"},
            {"name": "thal", "meaning": "Thallium stress scan result (blank = not recorded)", "units": "", "kind": "category", "known": "Thallium stress test, before angiography"},
        ],
        "target": {"meaning": "1 = angiographic disease (UCI `num` 1–4), 0 = none (`num` 0)",
                   "known": "At coronary angiography (the outcome)"},
        "excluded": [
            {"name": "ca", "meaning": "Number of major vessels (0–3) coloured by fluoroscopy", "known": "During the procedure that defines the outcome"},
        ],
    },
}

# Category value labels. Codes not listed keep their description from the source.
HEART_CODES = {
    "chest_pain": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "no symptoms"},
    "rest_ecg": {0: "normal", 1: "ST-T wave abnormality", 2: "left ventricular hypertrophy"},
    "st_slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
}
NHANES_RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black",
               6: "Non-Hispanic Asian", 7: "Other / multiracial"}
# Discharged to hospice or died: these patients cannot be readmitted (Strack et al. 2014; lecture app)
EXCLUDE_DISPOSITION = [11, 13, 14, 19, 20, 21]


# ---------------------------------------------------------------- builders
def _icd9_group(code) -> str:
    """Strack et al.'s nine groups for the primary diagnosis (copied from the lecture app)."""
    if not isinstance(code, str) or not code:
        return "Other"
    if code[0] in "VE":
        return "Other"
    v = float(code)
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def _ids_mapping(raw: Path) -> dict[str, dict[int, str]]:
    """IDS_mapping.csv holds three small tables separated by blank lines."""
    out, current = {}, None
    for line in (raw / "IDS_mapping.csv").read_text().splitlines():
        if not line.strip(",").strip():
            current = None
            continue
        head, _, desc = line.partition(",")
        if current is None:
            current = head
            out[current] = {}
            continue
        desc = desc.strip().strip('"').strip()
        out[current][int(head)] = "Not recorded" if desc in ("NULL", "Not Available", "Not Mapped", "Unknown/Invalid") else desc
    return out


def readmission(raw: Path) -> tuple[pd.DataFrame, list[dict]]:
    d = pd.read_csv(raw / "diabetic_data.csv", na_values="?", keep_default_na=False, low_memory=False)
    funnel = [{"step": "All encounters in the UCI file", "n": int(len(d))}]
    d = d[~d.discharge_disposition_id.isin(EXCLUDE_DISPOSITION)]
    funnel.append({"step": "Excluding discharge to hospice or death", "n": int(len(d))})
    d = d.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    funnel.append({"step": "First encounter per patient", "n": int(len(d))})
    y_all = (d.readmitted == "<30").astype(int).to_numpy()
    idx, _ = train_test_split(np.arange(len(d)), train_size=20000, stratify=y_all, random_state=SEED)
    d = d.iloc[idx]
    funnel.append({"step": "Stratified random sample (random_state=0)", "n": int(len(d))})

    ids = _ids_mapping(raw)
    X = pd.DataFrame(index=d.index)
    X["age"] = d.age.str.extract(r"\[(\d+)-").astype(int)[0] + 5
    X["female"] = (d.gender == "Female").astype(int)
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]:
        X[c] = d[c].astype(int)
    X["race"] = d.race.fillna("Unknown").replace({"AfricanAmerican": "African American", "Caucasian": "White"})
    X["admission_type"] = d.admission_type_id.map(ids["admission_type_id"]).fillna("Not recorded")
    X["admission_source"] = d.admission_source_id.map(ids["admission_source_id"]).fillna("Not recorded")
    X["discharge_disposition"] = d.discharge_disposition_id.map(ids["discharge_disposition_id"]).fillna("Not recorded")
    X["a1c_result"] = d.A1Cresult.replace({"Norm": "Normal"})
    X["max_glu_serum"] = d.max_glu_serum.replace({"Norm": "Normal"})
    X["insulin"] = d.insulin.replace({"No": "Not prescribed"})
    X["med_change"] = (d.change == "Ch").astype(int)
    X["diabetes_med"] = (d.diabetesMed == "Yes").astype(int)
    X["diag1_group"] = d.diag_1.map(_icd9_group)
    X["target"] = (d.readmitted == "<30").astype(int)
    return X.reset_index(drop=True), funnel


def diabetes(raw: Path) -> tuple[pd.DataFrame, list[dict]]:
    def read(f: str) -> pd.DataFrame:
        return pd.read_sas(raw / f"{f}.xpt").set_index("SEQN")

    d = (read("DEMO_J")[["RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDFMPIR"]]
         .join(read("BMX_J")[["BMXBMI", "BMXWAIST"]])
         .join(read("BPX_J")[["BPXSY1", "BPXSY2", "BPXSY3"]])
         .join(read("GHB_J")[["LBXGH"]])
         .join(read("DIQ_J")[["DIQ010", "DIQ050", "DIQ070"]])
         .join(read("HDL_J")[["LBDHDD"]])
         .join(read("TCHOL_J")[["LBXTC"]]))
    funnel = [{"step": "NHANES 2017–2018 participants", "n": int(len(d))}]
    d = d[d.RIDAGEYR >= 18]
    funnel.append({"step": "Adults (18+)", "n": int(len(d))})
    d = d[d.LBXGH.notna()]
    funnel.append({"step": "HbA1c measured", "n": int(len(d))})
    d = d[d.DIQ010.isin([1, 2, 3])].copy()
    funnel.append({"step": "Answered the diabetes question (yes / no / borderline)", "n": int(len(d))})
    X = pd.DataFrame(index=d.index)
    X["age"] = d.RIDAGEYR.astype(int)
    X["female"] = (d.RIAGENDR == 2).astype(int)
    X["bmi"] = d.BMXBMI
    X["waist"] = d.BMXWAIST
    X["sbp"] = d[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1).round(2)
    X["income_ratio"] = d.INDFMPIR
    X["race"] = d.RIDRETH3.map(NHANES_RACE)
    X["hdl"] = d.LBDHDD
    X["total_chol"] = d.LBXTC
    X["target"] = ((d.DIQ010 == 1) | (d.LBXGH >= 6.5)).astype(int)
    return X.reset_index(drop=True), funnel


def heart(raw: Path) -> tuple[pd.DataFrame, list[dict]]:
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    d = pd.read_csv(raw / "processed.cleveland.data", header=None, names=cols, na_values="?")
    X = pd.DataFrame(index=d.index)
    X["age"] = d.age.astype(int)
    X["male"] = d.sex.astype(int)
    X["chest_pain"] = d.cp.map(HEART_CODES["chest_pain"])
    X["rest_bp"] = d.trestbps.astype(int)
    X["chol"] = d.chol.astype(int)
    X["high_fasting_glucose"] = d.fbs.astype(int)
    X["rest_ecg"] = d.restecg.map(HEART_CODES["rest_ecg"])
    X["max_hr"] = d.thalach.astype(int)
    X["exercise_angina"] = d.exang.astype(int)
    X["st_depression"] = d.oldpeak
    X["st_slope"] = d.slope.map(HEART_CODES["st_slope"])
    X["thal"] = d.thal.map(HEART_CODES["thal"])  # 2 blanks: not recorded
    # `ca` is deliberately not copied: see DATASETS["heart"]["excluded"].
    X["target"] = (d.num > 0).astype(int)
    return X, [{"step": "Cleveland patients in the processed UCI file", "n": int(len(d))}]


BUILDERS = {"readmission": readmission, "diabetes": diabetes, "heart": heart}


# ---------------------------------------------------------------- splits and files
def add_splits(X: pd.DataFrame) -> pd.DataFrame:
    y = X.target.to_numpy()
    dev, test = train_test_split(np.arange(len(y)), test_size=TEST_SHARE, stratify=y, random_state=SEED)
    split = np.array(["dev"] * len(y), dtype=object)
    split[test] = "test"
    fold = pd.array([pd.NA] * len(y), dtype="Int64")
    for k, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(dev)), y[dev]), 1):
        fold[dev[va]] = k
    out = pd.DataFrame({"row_id": np.arange(1, len(y) + 1), "split": split, "fold": fold})
    return pd.concat([out, X.reset_index(drop=True)], axis=1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dictionary_entry(key: str, df: pd.DataFrame) -> dict:
    """datasets.json entry: the definition plus the categories found in the data."""
    d = json.loads(json.dumps(DATASETS[key]))
    for c in d["columns"]:
        c.setdefault("missing", "never blank")
        if df[c["name"]].isna().any():
            c["missing"] = f"blank = not recorded ({int(df[c['name']].isna().sum())} rows)"
        if c["kind"] == "category":
            counts = df[c["name"]].value_counts()
            c["categories"] = [str(v) for v in sorted(counts.index)]
    d["columns"] = COMMON + d["columns"]
    return d


def make_data(raw: Path, out: Path, raw_files: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_by": "reproduce/make_data.py in the M11 activity repository (instructor-only)",
        "dataset_definitions": "Readmission and diabetes cohorts from snsie/classification-app pipeline/datasets.py; heart from the UCI documentation",
        "raw_files": raw_files,
        "versions": {"python": "3.11", "numpy": version("numpy"), "pandas": version("pandas"), "scikit-learn": version("scikit-learn")},
        "split": {"seed": SEED, "test_share": TEST_SHARE, "n_folds": N_FOLDS,
                  "method": "train_test_split(test_size=0.2, stratify=target, random_state=0) → dev/test; "
                            "StratifiedKFold(5, shuffle=True, random_state=0) on the dev rows → fold 1–5"},
        "datasets": {},
    }
    dictionary = {}
    for key in KEYS:
        X, funnel = BUILDERS[key](raw)
        df = add_splits(X)
        path = out / f"{key}.csv"
        df.to_csv(path, index=False, lineterminator="\n")
        dev, test = df[df.split == "dev"], df[df.split == "test"]
        manifest["datasets"][key] = {
            "file": f"{key}.csv",
            "sha256": sha256(path),
            "n_rows": int(len(df)), "n_pos": int(df.target.sum()), "prevalence": round(float(df.target.mean()), 4),
            "n_dev": int(len(dev)), "n_dev_pos": int(dev.target.sum()),
            "n_test": int(len(test)), "n_test_pos": int(test.target.sum()),
            "folds": {str(k): {"n": int((dev.fold == k).sum()), "n_pos": int(dev[dev.fold == k].target.sum())} for k in range(1, N_FOLDS + 1)},
            "predictors": [c["name"] for c in DATASETS[key]["columns"]],
            "cohort": funnel,
            "source": DATASETS[key]["source"], "url": DATASETS[key]["url"], "licence": DATASETS[key]["licence"],
        }
        dictionary[key] = dictionary_entry(key, df)
        m = manifest["datasets"][key]
        print(f"  {key}: {m['n_rows']} rows, {m['n_pos']} positive ({m['prevalence']:.1%}); dev {m['n_dev']}, test {m['n_test']}")
    (out / "datasets.json").write_text(json.dumps(dictionary, indent=2, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (out / "README.md").write_text(readme(dictionary, manifest))
    return manifest


# ---------------------------------------------------------------- data/README.md
def _md(s: str) -> str:
    return s.replace("|", "\\|")


def readme(dictionary: dict, manifest: dict) -> str:
    lines = [
        "# The data",
        "",
        "Three public health datasets. One row is one patient or adult. Every file has the same layout:",
        "",
        "- `row_id`: a row number.",
        "- `split`: `dev` (development, 80%: train models and make choices) or `test` (20%: locked, used once by `python -m analysis test`).",
        "- `fold`: 1–5 for development rows. Every model uses these folds. Never reshuffle them.",
        "- the predictors, then `target` (1 = the positive class).",
        "",
        "Blank cells are missing values. **When is it known?** says when each value exists, relative to the prediction point. "
        "A column known only at or after the outcome would leak the answer, so those columns are left out (listed under **Excluded**).",
        "",
        "`manifest.json` records sources, licences, versions, row counts and SHA-256 fingerprints. `datasets.json` holds this dictionary as data.",
        "",
    ]
    for key in KEYS:
        d, m = dictionary[key], manifest["datasets"][key]
        lines += [
            f"## `{key}.csv`: {d['name'].lower()} ({m['n_rows']:,} {d['units']}, {m['n_pos']:,} positive, {m['prevalence']:.1%})",
            "",
            f"- **Cohort:** {d['cohort']}",
            f"- **Job:** {d['job']}",
            f"- **Prediction point:** {d['prediction_point']}",
            f"- **Positive class (`target` = 1):** {d['positive']}",
            f"- **Framing:** {d['framing']}",
            f"- **Split:** {m['n_dev']:,} development rows ({m['n_dev_pos']:,} positive) in 5 folds; {m['n_test']:,} test rows ({m['n_test_pos']:,} positive).",
            f"- **Source:** {d['source']} ({d['licence']}). {d['citation']}",
            "",
            "| Column | Meaning | Units | Categories / missing | When is it known? |",
            "|---|---|---|---|---|",
        ]
        for c in d["columns"]:
            extra = "; ".join(f"`{v}`" for v in c.get("categories", [])) if c["kind"] == "category" else ""
            if c.get("missing", "never blank") != "never blank":
                extra = (extra + "; " if extra else "") + c["missing"]
            lines.append(f"| `{c['name']}` | {_md(c['meaning'])} | {c['units']} | {_md(extra)} | {_md(c['known'])} |")
        lines.append(f"| `target` | **Target:** {_md(d['target']['meaning'])} | 0/1 |  | {_md(d['target']['known'])} |")
        if d["excluded"]:
            lines += ["", "**Excluded (leaked) columns, not in the file:**", "", "| Column | Meaning | When is it known? |", "|---|---|---|"]
            lines += [f"| `{c['name']}` | {_md(c['meaning'])} | {_md(c['known'])} |" for c in d["excluded"]]
        if d["one_per_patient"]:
            lines += ["", "One row per person, so each person's records are already kept together."]
        lines.append("")
    return "\n".join(lines)
