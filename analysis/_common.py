"""Shared helpers for analysis scripts.

Rebuilds the same dataset + split + model the trainer used, then loads a
saved checkpoint. Assumes the analysis script is invoked with the same seed
the run was trained with (the run's args.json captures this).
"""
import json
import tempfile
from pathlib import Path
from typing import Tuple

import torch

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import RNN, Transformer
from pyhealth.tasks import (
    DrugRecommendationMIMIC3,
    MortalityPredictionMIMIC3,
    ReadmissionPredictionMIMIC3,
)
from pyhealth.utils import set_seed

from data.admin_features import build_admin_lookup
from data.tasks import (
    MortalityWithAdminMIMIC3,
    ReadmissionWithAdminMIMIC3,
)
from models.hcat_binary import HCATBinary
from models.hcat_drugrec import HCATDrugRec, build_hist_to_label_map


def load_run_args(run_dir: Path) -> dict:
    with open(run_dir / "args.json") as f:
        return json.load(f)


def build_samples(args: dict):
    """Rebuild the same SampleDataset the trainer used.

    Uses a temp cache so this analysis script doesn't collide with concurrent
    training cache state — same input args produce identical samples.
    """
    base = MIMIC3Dataset(
        root=args["root"],
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=args.get("dev", False),
    )

    task = args["task"]
    variant = args["variant"]

    if task == "drug_rec":
        return base.set_task(DrugRecommendationMIMIC3())

    if task == "mortality":
        if variant == "baseline":
            return base.set_task(MortalityPredictionMIMIC3())
        admin_lookup = build_admin_lookup(args["root"])
        return base.set_task(MortalityWithAdminMIMIC3(admin_lookup))

    if task == "readmission":
        if variant == "baseline":
            return base.set_task(ReadmissionPredictionMIMIC3())
        admin_lookup = build_admin_lookup(args["root"])
        return base.set_task(ReadmissionWithAdminMIMIC3(admin_lookup))

    raise ValueError(f"unknown task: {task}")


def build_split(samples, args: dict):
    set_seed(args["seed"])
    train_ds, val_ds, test_ds = split_by_patient(samples, [0.8, 0.1, 0.1])
    return train_ds, val_ds, test_ds


def build_model(samples, args: dict):
    """Reconstruct the model architecture matching how the trainer built it."""
    task = args["task"]
    variant = args["variant"]
    embedding_dim = args["embedding_dim"]
    dropout = args["dropout"]

    if task == "drug_rec":
        if variant == "baseline":
            return Transformer(dataset=samples, embedding_dim=embedding_dim)
        use_focal    = variant in ("AB", "ABC")
        use_copy     = variant == "ABC"
        use_evidence = variant == "ABC"
        hist_to_label = build_hist_to_label_map(samples) if use_copy else None
        return HCATDrugRec(
            dataset=samples,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_focal=use_focal,
            use_copy=use_copy,
            use_evidence=use_evidence,
            focal_gamma=args["focal_gamma"],
            hist_to_label=hist_to_label,
        )

    if task in ("mortality", "readmission"):
        if variant == "baseline":
            return RNN(dataset=samples, embedding_dim=embedding_dim)
        use_codes = variant != "no_codes"
        use_admin = variant != "no_admin"
        return HCATBinary(
            dataset=samples,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_codes=use_codes,
            use_admin=use_admin,
            use_focal=args.get("use_focal", False),
            focal_gamma=args["focal_gamma"],
        )

    raise ValueError(f"unknown task: {task}")


def load_checkpoint(model, run_dir: Path, ckpt: str = "best.ckpt"):
    """Load PyHealth-saved state_dict into a freshly-built model."""
    ckpt_path = run_dir / "pyhealth" / ckpt
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_run(run_dir: Path) -> Tuple[dict, object, object, object, object]:
    """Convenience: rebuild everything for a run.

    Returns:
        (args, samples, splits, model, dataloaders)
        splits     = (train_ds, val_ds, test_ds)
        dataloaders= (train_loader, val_loader, test_loader)
    """
    args = load_run_args(run_dir)
    samples = build_samples(args)
    splits = build_split(samples, args)
    model = build_model(samples, args)
    load_checkpoint(model, run_dir)

    bs = args["batch_size"]
    loaders = (
        get_dataloader(splits[0], batch_size=bs, shuffle=False),
        get_dataloader(splits[1], batch_size=bs, shuffle=False),
        get_dataloader(splits[2], batch_size=bs, shuffle=False),
    )
    return args, samples, splits, model, loaders
