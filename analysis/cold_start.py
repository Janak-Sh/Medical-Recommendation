"""A1 — Cold-start / short-history degradation curve (drug recommendation).

Buckets test patients by `n_visits` (the number of valid visits in the input
history, including the current visit being predicted) and reports per-bucket:
    pr_auc_samples, roc_auc_samples, jaccard_samples, n_samples


Usage:
    python -m analysis.cold_start --run results/drug_rec/drug_rec_ABC_seed42_e20

Outputs:
    <run>/cold_start.csv   one row per visit-count bucket
    <run>/cold_start.png   bar chart (if matplotlib is available)
"""
import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from analysis._common import load_run

# Default visit-count buckets (inclusive on the lower bound, exclusive on
# the upper, except the open-ended last bucket). Tuned to give roughly
# balanced sample counts on MIMIC-III drug-rec.
DEFAULT_BUCKETS = [(1, 2), (2, 3), (3, 5), (5, 8), (8, 999)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Path to a trained drug_rec run dir.")
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args()


def bucket_label(lo: int, hi: int) -> str:
    if hi >= 999:
        return f"{lo}+"
    if hi - lo == 1:
        return str(lo)
    return f"{lo}-{hi-1}"


def safe_pr_auc_samples(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Sample-averaged AP, skipping samples with zero positives.

    sklearn's average_precision_score warns and returns 1.0 when a sample has
    no positives — that's misleading at the bucket level, so we drop those.
    """
    from sklearn.metrics import average_precision_score
    n_pos = y_true.sum(axis=1)
    keep = n_pos > 0
    if keep.sum() == 0:
        return float("nan")
    aps = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in np.where(keep)[0]:
            aps.append(average_precision_score(y_true[i], y_prob[i]))
    return float(np.mean(aps))


def safe_roc_auc_samples(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    n_pos = y_true.sum(axis=1)
    keep = n_pos > 0
    if keep.sum() == 0:
        return float("nan")
    aucs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in np.where(keep)[0]:
            try:
                aucs.append(roc_auc_score(y_true[i], y_prob[i]))
            except ValueError:
                pass
    return float(np.mean(aucs)) if aucs else float("nan")


def jaccard_samples(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    inter = (y_true * y_pred).sum(axis=1)
    union = ((y_true + y_pred) > 0).sum(axis=1)
    keep = union > 0
    if keep.sum() == 0:
        return float("nan")
    return float((inter[keep] / union[keep]).mean())


def main():
    args = parse_args()
    run_dir = Path(args.run)

    print(f"[cold_start] loading {run_dir}")
    run_args, samples, splits, model, loaders = load_run(run_dir)
    if run_args["task"] != "drug_rec":
        raise SystemExit(f"cold_start.py is for drug_rec only (got {run_args['task']}).")

    test_loader = loaders[2]

    print("[cold_start] running inference on test set")
    n_visits_all, y_true_all, y_prob_all = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            conditions = batch["conditions"].long()
            n_visits = (conditions != 0).any(dim=2).sum(dim=1).cpu().numpy()
            out = model(**batch)
            n_visits_all.append(n_visits)
            y_true_all.append(out["y_true"].cpu().numpy())
            y_prob_all.append(out["y_prob"].cpu().numpy())

    n_visits = np.concatenate(n_visits_all)
    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    print(f"[cold_start] N={len(n_visits)} test samples, "
          f"visit count range [{n_visits.min()}, {n_visits.max()}]")

    rows = []
    for lo, hi in DEFAULT_BUCKETS:
        mask = (n_visits >= lo) & (n_visits < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bucket": bucket_label(lo, hi), "n_samples": 0,
                         "pr_auc_samples": float("nan"),
                         "roc_auc_samples": float("nan"),
                         "jaccard_samples": float("nan")})
            continue
        rows.append({
            "bucket": bucket_label(lo, hi),
            "n_samples": n,
            "pr_auc_samples":  safe_pr_auc_samples(y_true[mask],  y_prob[mask]),
            "roc_auc_samples": safe_roc_auc_samples(y_true[mask], y_prob[mask]),
            "jaccard_samples": jaccard_samples(y_true[mask],      y_prob[mask]),
        })

    print(f"\n  {'bucket':>6} {'n':>6} {'pr_auc':>8} {'roc_auc':>8} {'jaccard':>8}")
    for r in rows:
        print(f"  {r['bucket']:>6} {r['n_samples']:>6} "
              f"{r['pr_auc_samples']:>8.4f} {r['roc_auc_samples']:>8.4f} "
              f"{r['jaccard_samples']:>8.4f}")

    out_csv = run_dir / "cold_start.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[cold_start] wrote {out_csv}")

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            buckets = [r["bucket"] for r in rows]
            pr  = [r["pr_auc_samples"]  for r in rows]
            roc = [r["roc_auc_samples"] for r in rows]
            jac = [r["jaccard_samples"] for r in rows]

            fig, ax1 = plt.subplots(figsize=(7, 4))
            x = np.arange(len(buckets))
            w = 0.27
            ax1.bar(x - w, pr,  width=w, label="pr_auc_samples")
            ax1.bar(x,     roc, width=w, label="roc_auc_samples")
            ax1.bar(x + w, jac, width=w, label="jaccard_samples")
            ax1.set_xticks(x)
            ax1.set_xticklabels(buckets)
            ax1.set_xlabel("# visits in patient history (incl. current)")
            ax1.set_ylabel("score")
            ax1.set_ylim(0, 1)
            ax1.set_title(f"Drug rec — degradation by visit count ({run_dir.name})")
            ax1.legend(loc="lower right")

            ax2 = ax1.twinx()
            ax2.plot(x, [r["n_samples"] for r in rows], "ko--", alpha=0.6, label="n_samples")
            ax2.set_ylabel("n_samples")
            ax2.legend(loc="upper right")

            fig.tight_layout()
            out_png = run_dir / "cold_start.png"
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[cold_start] wrote {out_png}")
        except ImportError:
            print("[cold_start] matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()
