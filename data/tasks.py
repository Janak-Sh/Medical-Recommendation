"""Custom PyHealth tasks that augment each sample with extra features.

For both mortality and readmission we provide two augmented variants:

    {Mortality,Readmission}WithDemoMIMIC3
        Adds age_bucket + admit_type per visit.

    {Mortality,Readmission}WithAdminMIMIC3
        Adds the 5 admin features the data study identified as dominant
        signals (age, los, n_prior, admit_type, discharge_location).

Note on visit-id keys: PyHealth's MortalityPredictionMIMIC3 emits `hadm_id`
while ReadmissionPredictionMIMIC3 emits `visit_id` (both are MIMIC's HADM_ID).
We normalize this with a small per-task helper so both tasks index into the
same lookup dict.
"""
from typing import Any, Dict, List

from pyhealth.tasks import (
    MortalityPredictionMIMIC3,
    ReadmissionPredictionMIMIC3,
)

from data.admin_features import DEFAULT_ADMIN
from data.demographics import make_key


def _mortality_visit_id(s: Dict[str, Any]) -> str:
    return s["hadm_id"]


def _readmission_visit_id(s: Dict[str, Any]) -> str:
    return s["visit_id"]


# ----------------------------------------------------------------------------
# Mortality
# ----------------------------------------------------------------------------
class MortalityWithDemoMIMIC3(MortalityPredictionMIMIC3):
    """MortalityPredictionMIMIC3 + age_bucket + admit_type."""

    task_name: str = "MortalityWithDemoMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs":      "sequence",
        "age_bucket": "sequence",
        "admit_type": "sequence",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __init__(self, demo_lookup: Dict[str, List[str]]):
        super().__init__()
        self.demo_lookup = demo_lookup

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = super().__call__(patient)
        for s in samples:
            key = make_key(s["patient_id"], _mortality_visit_id(s))
            age_bucket, admit_type = self.demo_lookup.get(key, ["unknown", "UNKNOWN"])
            s["age_bucket"] = [age_bucket]
            s["admit_type"] = [admit_type]
        return samples


class MortalityWithAdminMIMIC3(MortalityPredictionMIMIC3):
    """MortalityPredictionMIMIC3 + 5 admin features (age_z, los_z, n_prior_z,
    admit_type, discharge_location)."""

    task_name: str = "MortalityWithAdminMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions":         "sequence",
        "procedures":         "sequence",
        "drugs":              "sequence",
        "admin_numeric":      "tensor",
        "admit_type":         "sequence",
        "discharge_location": "sequence",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __init__(self, admin_lookup: Dict[str, Dict]):
        super().__init__()
        self.admin_lookup = admin_lookup

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = super().__call__(patient)
        for s in samples:
            key = make_key(s["patient_id"], _mortality_visit_id(s))
            entry = self.admin_lookup.get(key, DEFAULT_ADMIN)
            s["admin_numeric"]      = entry["numeric"]
            s["admit_type"]         = [entry["admit_type"]]
            s["discharge_location"] = [entry["discharge_location"]]
        return samples


# ----------------------------------------------------------------------------
# Readmission
# ----------------------------------------------------------------------------
class ReadmissionWithDemoMIMIC3(ReadmissionPredictionMIMIC3):
    """ReadmissionPredictionMIMIC3 + age_bucket + admit_type."""

    task_name: str = "ReadmissionWithDemoMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs":      "sequence",
        "age_bucket": "sequence",
        "admit_type": "sequence",
    }
    output_schema: Dict[str, str] = {"readmission": "binary"}

    def __init__(self, demo_lookup: Dict[str, List[str]]):
        super().__init__()
        self.demo_lookup = demo_lookup

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = super().__call__(patient)
        for s in samples:
            key = make_key(s["patient_id"], _readmission_visit_id(s))
            age_bucket, admit_type = self.demo_lookup.get(key, ["unknown", "UNKNOWN"])
            s["age_bucket"] = [age_bucket]
            s["admit_type"] = [admit_type]
        return samples


class ReadmissionWithAdminMIMIC3(ReadmissionPredictionMIMIC3):
    """ReadmissionPredictionMIMIC3 + 5 admin features."""

    task_name: str = "ReadmissionWithAdminMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions":         "sequence",
        "procedures":         "sequence",
        "drugs":              "sequence",
        "admin_numeric":      "tensor",
        "admit_type":         "sequence",
        "discharge_location": "sequence",
    }
    output_schema: Dict[str, str] = {"readmission": "binary"}

    def __init__(self, admin_lookup: Dict[str, Dict]):
        super().__init__()
        self.admin_lookup = admin_lookup

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = super().__call__(patient)
        for s in samples:
            key = make_key(s["patient_id"], _readmission_visit_id(s))
            entry = self.admin_lookup.get(key, DEFAULT_ADMIN)
            s["admin_numeric"]      = entry["numeric"]
            s["admit_type"]         = [entry["admit_type"]]
            s["discharge_location"] = [entry["discharge_location"]]
        return samples
