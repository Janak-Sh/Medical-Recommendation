"""Admin / structured feature loader.

Builds a {(patient, hadm) -> feature_dict} lookup from PATIENTS + ADMISSIONS.
A pre-modelling data study showed that 5 admin features (age, los, n_prior,
admit_type, discharge_location) carry roughly as much next-visit signal as
the full ICD/proc/drug code vocabulary, so we expose them as a first-class
input path for the binary tasks (mortality / readmission).

Each entry contains:
    numeric: [age_z, los_log_z, n_prior_z]   continuous, z-score normalized
    admit_type: str                          categorical (EMERGENCY/ELECTIVE/...)
    discharge_location: str                  categorical (HOME/SNF/REHAB/...)
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from data.demographics import make_key


N_ADMIN_NUMERIC = 3   # age_z, los_z, nprior_z

DEFAULT_ADMIN: Dict = {
    "numeric": [0.0, 0.0, 0.0],
    "admit_type": "UNKNOWN",
    "discharge_location": "UNKNOWN",
}


def build_admin_lookup(root: str) -> Dict[str, Dict]:
    """Returns {"<pid>|<hadm>" -> {numeric, admit_type, discharge_location}}.

    The numeric vector is z-scored using population stats so the model sees
    roughly zero-mean / unit-variance inputs. n_prior is capped at 10 to bound
    the value range; los is z-scored on log(1+days) to handle the long tail;
    age uses a year-only diff to sidestep MIMIC's shifted-DOB datetime overflow.
    """
    root = Path(root)

    pat = pd.read_csv(
        root / "PATIENTS.csv.gz", compression="gzip",
        usecols=["SUBJECT_ID", "DOB"], parse_dates=["DOB"],
    )
    pat_dob = dict(zip(pat["SUBJECT_ID"].astype(str), pat["DOB"]))

    adm = pd.read_csv(
        root / "ADMISSIONS.csv.gz", compression="gzip",
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME",
                 "ADMISSION_TYPE", "DISCHARGE_LOCATION"],
        parse_dates=["ADMITTIME", "DISCHTIME"],
    )
    adm = adm.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)

    adm["n_prior"] = adm.groupby("SUBJECT_ID").cumcount()
    adm["n_prior_capped"] = adm["n_prior"].clip(upper=10)

    adm["los_days"] = (adm["DISCHTIME"] - adm["ADMITTIME"]).dt.total_seconds() / 86400.0
    adm["los_days"] = adm["los_days"].clip(lower=0).fillna(0)
    adm["los_log"] = np.log1p(adm["los_days"])

    def _age(row):
        dob = pat_dob.get(str(row["SUBJECT_ID"]))
        if dob is None or pd.isna(dob) or pd.isna(row["ADMITTIME"]):
            return -1
        age = row["ADMITTIME"].year - dob.year
        return min(age, 89) if age > 0 else -1

    adm["age"] = adm.apply(_age, axis=1)

    age_mean = float(adm.loc[adm["age"] >= 0, "age"].mean())
    age_std  = float(adm.loc[adm["age"] >= 0, "age"].std())
    los_mean = float(adm["los_log"].mean())
    los_std  = float(adm["los_log"].std())
    nprior_mean = float(adm["n_prior_capped"].mean())
    nprior_std  = float(adm["n_prior_capped"].std())

    eps = 1e-6
    lookup: Dict[str, Dict] = {}
    for _, row in adm.iterrows():
        pid = str(row["SUBJECT_ID"])
        hadm = str(row["HADM_ID"])

        age = row["age"]
        age_z = (age - age_mean) / max(age_std, eps) if age >= 0 else 0.0
        los_z = (row["los_log"] - los_mean) / max(los_std, eps)
        nprior_z = (row["n_prior_capped"] - nprior_mean) / max(nprior_std, eps)

        admit_type = (row["ADMISSION_TYPE"]
                      if pd.notna(row["ADMISSION_TYPE"]) else "UNKNOWN")
        discharge = (row["DISCHARGE_LOCATION"]
                     if pd.notna(row["DISCHARGE_LOCATION"]) else "UNKNOWN")

        lookup[make_key(pid, hadm)] = {
            "numeric": [float(age_z), float(los_z), float(nprior_z)],
            "admit_type": admit_type,
            "discharge_location": discharge,
        }

    return lookup
