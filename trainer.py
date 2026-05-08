"""Unified trainer for all three MIMIC-III tasks: drug recommendation,
mortality prediction, and readmission prediction.

Usage:
    python trainer.py --task drug_rec    --variant ABC  --epochs 30
    python trainer.py --task mortality   --variant full --epochs 20
    python trainer.py --task readmission --variant full --epochs 20

Variants:
    drug_rec:   baseline | A | AB | ABC
        baseline -> PyHealth Transformer + BCE
        A        -> HCATDrugRec, BCE
        AB       -> HCATDrugRec, focal BCE
        ABC      -> HCATDrugRec, focal BCE + copy + evidence streams (full model)

    mortality / readmission:  baseline | full | no_admin | no_codes
        baseline -> PyHealth RNN
        full     -> HCATBinary, codes + admin features
        no_admin -> HCATBinary, codes only
        no_codes -> HCATBinary, admin features only

Each run writes per-epoch trajectory CSV, final test_metrics.json, args.json,
and (for drug_rec) appends a row to a master all-runs CSV. WandB logging is
on by default; pass --no_wandb to disable.
"""
import argparse
import csv
import json
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import time

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import RNN, Transformer
from pyhealth.tasks import DrugRecommendationMIMIC3
from pyhealth.trainer import Trainer, get_metrics_fn
from pyhealth.utils import set_seed

from data.admin_features import build_admin_lookup
from data.tasks import (
    MortalityWithAdminMIMIC3,
    ReadmissionWithAdminMIMIC3,
)
from models.hcat_binary import HCATBinary
from models.hcat_drugrec import HCATDrugRec, build_hist_to_label_map


# ============================================================================
# Task config
# ============================================================================
TASK_REGISTRY = {
    "drug_rec": {
        "label_key": "drugs",
        "variants": ["baseline", "A", "AB", "ABC"],
        "metrics": [
            "pr_auc_samples", "pr_auc_macro", "pr_auc_micro",
            "f1_samples", "f1_macro",
            "jaccard_samples",
            "roc_auc_samples",
        ],
        "monitor": "pr_auc_samples",
        "wandb_project_default": "mimic3-drug-recommendation",
    },
    "mortality": {
        "label_key": "mortality",
        "variants": ["baseline", "full", "no_admin", "no_codes"],
        "metrics": ["roc_auc", "pr_auc", "f1", "accuracy"],
        "monitor": "roc_auc",
        "wandb_project_default": "mimic3-mortality-prediction",
    },
    "readmission": {
        "label_key": "readmission",
        "variants": ["baseline", "full", "no_admin", "no_codes"],
        "metrics": ["roc_auc", "pr_auc", "f1", "accuracy"],
        "monitor": "roc_auc",
        "wandb_project_default": "mimic3-readmission-prediction",
    },
}


# ============================================================================
# precision@k / recall@k (drug_rec only)
# ============================================================================
def precision_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> Tuple[float, float]:
    """Sample-averaged precision@k and recall@k for a multi-label problem."""
    n_labels = y_true.shape[1]
    k_eff = min(k, n_labels)
    top_k = np.argpartition(-y_prob, kth=k_eff - 1, axis=1)[:, :k_eff]
    hits = np.take_along_axis(y_true, top_k, axis=1).sum(axis=1)
    n_pos = y_true.sum(axis=1).clip(min=1)
    return float((hits / k_eff).mean()), float((hits / n_pos).mean())


def make_extended_evaluate(trainer: Trainer, k_list=(10, 20, 30)):
    """Evaluate that computes PyHealth metrics + precision@k / recall@k in one pass."""
    def evaluate(dataloader):
        y_true, y_prob, loss_mean = trainer.inference(dataloader)
        if trainer.model.mode is not None:
            metrics_fn = get_metrics_fn(trainer.model.mode)
            scores = metrics_fn(y_true, y_prob, metrics=trainer.metrics)
        else:
            scores = {}
        scores["loss"] = float(loss_mean)
        for k in k_list:
            p_k, r_k = precision_recall_at_k(y_true, y_prob, k)
            scores[f"precision@{k}"] = p_k
            scores[f"recall@{k}"] = r_k
        return scores
    return evaluate


# ============================================================================
# WandB log handler - parses PyHealth Trainer's per-epoch eval output
# ============================================================================
class WandBHandler(logging.Handler):
    EPOCH_RE = re.compile(r"---\s*(Train|Eval)\s+epoch-(\d+),\s*step-(\d+)\s*---")
    METRIC_RE = re.compile(r"^([\w@]+):\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")

    def __init__(self, run, csv_path: Path):
        super().__init__()
        self.run = run
        self.csv_path = csv_path
        self._mode = None
        self._epoch = -1
        self._buffer = {}
        self._train_loss = None
        self._trajectory = []
        self._epoch_start = time.time()
        self._train_start = time.time()

        try:
            self.run.define_metric("epoch")
            self.run.define_metric("train/*", step_metric="epoch")
            self.run.define_metric("val/*",   step_metric="epoch")
            self.run.define_metric("test/*",  step_metric="epoch")
        except Exception:
            pass

    def emit(self, record):
        msg = record.getMessage()
        m_block = self.EPOCH_RE.match(msg)
        if m_block:
            self._flush_block()
            self._mode = m_block.group(1).lower()
            self._epoch = int(m_block.group(2))
            self._buffer = {}
            return
        m_metric = self.METRIC_RE.match(msg)
        if m_metric and self._mode is not None:
            self._buffer[m_metric.group(1)] = float(m_metric.group(2))

    def _flush_block(self):
        if self._mode is None or not self._buffer:
            return
        if self._mode == "train":
            self._train_loss = self._buffer.get("loss")
        elif self._mode == "eval":
            epoch_time = time.time() - self._epoch_start
            log_dict = {f"val/{k}": v for k, v in self._buffer.items()}
            if self._train_loss is not None:
                log_dict["train/loss"] = self._train_loss
            log_dict["epoch"] = self._epoch
            log_dict["epoch_time_seconds"] = epoch_time
            self._epoch_start = time.time()
            try:
                self.run.log(log_dict)
            except Exception:
                pass
            row = {"epoch": self._epoch, "train_loss": self._train_loss}
            for k, v in self._buffer.items():
                row[f"val_{k}"] = v
            self._trajectory.append(row)
        self._mode = None
        self._buffer = {}

    def close_and_save_csv(self):
        self._flush_block()
        total_time = time.time() - self._train_start
        try:
            self.run.summary["total_training_seconds"] = total_time  # add this
        except Exception:
            pass
        if not self._trajectory:
            return
        all_keys = sorted({k for r in self._trajectory for k in r.keys()})
        ordered = ["epoch"] + sorted(k for k in all_keys if k != "epoch")
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            for r in self._trajectory:
                w.writerow({k: r.get(k, "") for k in ordered})


# ============================================================================
# Per-task setup (samples, model, defaults)
# ============================================================================
def setup_drug_rec(args, base_dataset):
    samples = base_dataset.set_task(DrugRecommendationMIMIC3())
    print(f"  {len(samples)} samples")

    if args.variant == "baseline":
        model_factory = lambda ds: Transformer(dataset=ds, embedding_dim=args.embedding_dim)
        train_pos_rate = None
    else:
        use_focal    = args.variant in ("AB", "ABC")
        use_copy     = args.variant == "ABC"
        use_evidence = args.variant == "ABC"
        hist_to_label = build_hist_to_label_map(samples) if use_copy else None

        def model_factory(ds):
            return HCATDrugRec(
                dataset=ds,
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
                use_focal=use_focal,
                use_copy=use_copy,
                use_evidence=use_evidence,
                focal_gamma=args.focal_gamma,
                hist_to_label=hist_to_label,
            )
        train_pos_rate = None  # bias init not used for multi-label

    return samples, model_factory, train_pos_rate


def setup_binary(args, base_dataset, label_key, task_admin_cls):
    """Shared mortality / readmission setup."""
    if args.variant == "baseline":
        # Use the vanilla PyHealth task (no admin features needed)
        if label_key == "mortality":
            from pyhealth.tasks import MortalityPredictionMIMIC3
            task = MortalityPredictionMIMIC3()
        else:
            from pyhealth.tasks import ReadmissionPredictionMIMIC3
            task = ReadmissionPredictionMIMIC3()
        samples = base_dataset.set_task(task)
    else:
        print("  building admin lookup from PATIENTS + ADMISSIONS...")
        admin_lookup = build_admin_lookup(args.root)
        print(f"    {len(admin_lookup)} (patient_id, hadm_id) entries")
        samples = base_dataset.set_task(task_admin_cls(admin_lookup))

    pos = sum(1 for s in samples if s[label_key] == 1)
    rate = pos / max(len(samples), 1)
    print(f"  {len(samples)} samples ({pos} positive, {label_key} rate {rate:.4f})")

    if args.variant == "baseline":
        model_factory = lambda ds: RNN(dataset=ds, embedding_dim=args.embedding_dim)
    else:
        use_codes = args.variant != "no_codes"
        use_admin = args.variant != "no_admin"

        def model_factory(ds):
            return HCATBinary(
                dataset=ds,
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
                use_codes=use_codes,
                use_admin=use_admin,
                use_focal=args.use_focal,
                focal_gamma=args.focal_gamma,
            )

    return samples, model_factory, label_key


# ============================================================================
# Args
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASK_REGISTRY.keys()))
    p.add_argument("--variant", required=True,
                   help="See task-specific list in module docstring.")
    p.add_argument("--root", default="./physionet_data",
                   help="Path to MIMIC-III csv.gz root.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=None,
                   help="Early-stop patience. Default: 10 for non-baseline drug_rec, "
                        "no early stopping otherwise.")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Default: 32")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=None,
                   help="Weight decay.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--use_focal", action="store_true",
                   help="Binary tasks only: use focal BCE instead of vanilla BCE.")
    p.add_argument("--exp_name", default=None)
    p.add_argument("--output", default=None,
                   help="Output dir. Default: ./results/<task>/")
    p.add_argument("--dev", action="store_true",
                   help="Use a tiny subset of MIMIC-III for smoke testing.")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default=None)
    args = p.parse_args()

    cfg = TASK_REGISTRY[args.task]
    if args.variant not in cfg["variants"]:
        p.error(f"--variant must be one of {cfg['variants']} for --task={args.task}")

    # Variant-aware defaults that match what was tuned per task
    if args.task == "drug_rec":
        is_baseline = args.variant == "baseline"
        if args.batch_size is None:
            args.batch_size = 32 if is_baseline else 24
        if args.wd is None:
            args.wd = 0.0
        if args.patience is None and not is_baseline:
            args.patience = 10
    else:
        if args.batch_size is None:
            args.batch_size = 32
        if args.wd is None:
            args.wd = 1e-4

    if args.exp_name is None:
        args.exp_name = f"{args.task}_{args.variant}_seed{args.seed}_e{args.epochs}"
        if args.dev:
            args.exp_name += "_dev"

    if args.output is None:
        args.output = f"./results/{args.task}"

    if args.wandb_project is None:
        args.wandb_project = cfg["wandb_project_default"]

    return args


# ============================================================================
# Main
# ============================================================================
def main():
    load_dotenv()
    args = parse_args()
    cfg = TASK_REGISTRY[args.task]

    set_seed(args.seed)

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{args.exp_name}] === task={args.task} variant={args.variant} epochs={args.epochs} ===")

    # 1) Load MIMIC-III
    print(f"loading MIMIC-III from {args.root}")
    base_dataset = MIMIC3Dataset(
        root=args.root,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=args.dev,
    )

    # 2) Task setup
    if args.task == "drug_rec":
        samples, model_factory, _ = setup_drug_rec(args, base_dataset)
    elif args.task == "mortality":
        samples, model_factory, label_key = setup_binary(
            args, base_dataset, "mortality", MortalityWithAdminMIMIC3,
        )
    else:  # readmission
        samples, model_factory, label_key = setup_binary(
            args, base_dataset, "readmission", ReadmissionWithAdminMIMIC3,
        )

    # 3) Patient-level split
    train_ds, val_ds, test_ds = split_by_patient(samples, [0.8, 0.1, 0.1])
    print(f"  split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = get_dataloader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # 4) Build model
    model = model_factory(samples)

    # Bias init from train positive rate is helpful for the imbalanced binary tasks.
    if args.task in ("mortality", "readmission") and args.variant != "baseline":
        train_pos = sum(1 for s in train_ds if s[cfg["label_key"]] == 1)
        train_pos_rate = train_pos / max(len(train_ds), 1)
        print(f"  train {cfg['label_key']} rate: {train_pos_rate:.4f} -> bias init = "
              f"log({train_pos_rate:.4f}/(1-{train_pos_rate:.4f}))")
        model.init_bias_from_marginal(train_pos_rate)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {model.__class__.__name__}, {n_params/1e6:.2f}M params")

    # 5) WandB init (must come before Trainer so the handler can attach)
    use_wandb = (not args.no_wandb) and _WANDB_AVAILABLE
    wandb_run = None
    wandb_handler = None
    if use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config={
                **vars(args),
                "model_class": model.__class__.__name__,
                "n_params": n_params,
            },
            tags=[
                f"task={args.task}",
                f"variant={args.variant}",
                f"seed={args.seed}",
                model.__class__.__name__,
            ],
            reinit=True,
        )
        wandb_handler = WandBHandler(wandb_run, run_dir / "trajectory.csv")
        logging.getLogger("pyhealth.trainer").addHandler(wandb_handler)

    # 6) Train
    trainer = Trainer(
        model=model,
        metrics=cfg["metrics"],
        output_path=str(run_dir),
        exp_name="pyhealth",
    )
    if args.task == "drug_rec":
        # Add precision@k / recall@k to per-epoch eval
        trainer.evaluate = make_extended_evaluate(trainer)

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor=cfg["monitor"],
        monitor_criterion="max",
        patience=args.patience,
        optimizer_params={"lr": args.lr},
        weight_decay=args.wd,
    )

    # 7) Test
    print(f"\n[{args.exp_name}] evaluating on test set")
    test_metrics = trainer.evaluate(test_loader)
    print(f"  test: {json.dumps(test_metrics, indent=2)}")

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(run_dir / "model_info.json", "w") as f:
        json.dump({"class": model.__class__.__name__, "n_params": n_params}, f, indent=2)

    # 8) WandB: log test metrics + close trajectory CSV
    if use_wandb and wandb_run is not None:
        wandb_handler.close_and_save_csv()
        for k, v in test_metrics.items():
            wandb_run.summary[f"test/{k}"] = v
        wandb_run.log({
            **{f"test/{k}": v for k, v in test_metrics.items()},
            "epoch": args.epochs - 1,
        })
        wandb_run.finish()

    # 9) Append a row to the master CSV for this task
    master_csv = out_root / "all_runs_test_metrics.csv"
    csv_row = {
        "exp_name": args.exp_name,
        "task": args.task,
        "variant": args.variant,
        "epochs": args.epochs,
        "seed": args.seed,
        "lr": args.lr,
        "wd": args.wd,
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "model_class": model.__class__.__name__,
        "n_params": n_params,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    csv_exists = master_csv.exists()
    fieldnames = list(csv_row.keys())
    if csv_exists:
        with open(master_csv, "r", newline="") as f:
            existing_keys = next(csv.reader(f), [])
        union = list(existing_keys)
        for k in fieldnames:
            if k not in union:
                union.append(k)
        fieldnames = union
    with open(master_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            w.writeheader()
        w.writerow({k: csv_row.get(k, "") for k in fieldnames})

    print(f"\n[{args.exp_name}] DONE. Results in {run_dir}/")
    print(f"  master CSV: {master_csv}")


if __name__ == "__main__":
    main()
