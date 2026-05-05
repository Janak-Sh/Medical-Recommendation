"""Demographic feature loader (age bucket + admit type per visit).

Reads PATIENTS.csv.gz and ADMISSIONS.csv.gz from a MIMIC-III root and returns a
{(patient_id, hadm_id) -> [age_bucket, admit_type]} lookup.

MIMIC shifts elderly DOBs by 100-300 years for de-identification, which causes
pandas datetime subtraction to overflow. We use year-only diffs to sidestep this,
and clamp age at 89 to match the MIMIC de-id rule.
"""
from pathlib import Path
from typing import Dict, List

import pandas as pd


AGE_BUCKETS = [
    ("under_40", 0, 40),
    ("40_to_55", 40, 55),
    ("55_to_70", 55, 70),
    ("70_to_85", 70, 85),
    ("over_85",  85, 200),
]


def age_to_bucket(age) -> str:
    if age is None or age < 0:
        return "unknown"
    for name, lo, hi in AGE_BUCKETS:
        if lo <= age < hi:
            return name
    return "over_85"


def make_key(patient_id, hadm_id) -> str:
    """Composite string key. PyHealth json-serializes task instance vars for
    caching and can't tolerate tuple keys, so we use "<pid>|<hadm>".
    """
    return f"{patient_id}|{hadm_id}"


def build_demographic_lookup(root: str) -> Dict[str, List[str]]:
    """Returns {"<patient_id>|<hadm_id>": [age_bucket, admit_type]}."""
    root = Path(root)

    patients = pd.read_csv(
        root / "PATIENTS.csv.gz", compression="gzip",
        usecols=["SUBJECT_ID", "DOB"], parse_dates=["DOB"],
    )
    pat_dob = dict(zip(patients["SUBJECT_ID"].astype(str), patients["DOB"]))

    adms = pd.read_csv(
        root / "ADMISSIONS.csv.gz", compression="gzip",
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "ADMISSION_TYPE"],
        parse_dates=["ADMITTIME"],
    )

    lookup: Dict[str, List[str]] = {}
    for _, row in adms.iterrows():
        pid = str(row["SUBJECT_ID"])
        hadm = str(row["HADM_ID"])
        admittime = row["ADMITTIME"]
        admit_type = row["ADMISSION_TYPE"] if pd.notna(row["ADMISSION_TYPE"]) else "UNKNOWN"

        dob = pat_dob.get(pid)
        if dob is None or pd.isna(admittime) or pd.isna(dob):
            age_bucket = "unknown"
        else:
            age = admittime.year - dob.year
            if age > 89:
                age = 89
            age_bucket = age_to_bucket(age)

        lookup[make_key(pid, hadm)] = [age_bucket, admit_type]
    return lookup
