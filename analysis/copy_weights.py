"""B1 — Copy-stream weight inspection (drug recommendation).

Dumps the per-drug `w_recent` and `w_any` parameters learned by HCATDrugRec.
These are *trained-in* interpretability signals: a high w_recent on drug X
means "if X was prescribed at the most recent visit, predict X again",
i.e. chronic / refill behaviour. A low or negative w_recent means the model
learned NOT to copy this drug from history (typical for one-off acute meds).

Usage:
    python -m analysis.copy_weights --run results/drug_rec/drug_rec_ABC_seed42_e20

Outputs:
    <run>/copy_weights.csv   per-drug w_recent + w_any, sorted by w_recent
    Plus a printed top-20 / bottom-20 summary.
"""
import argparse
import csv
from pathlib import Path

import torch

from analysis._common import build_samples, build_model, load_run_args, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Path to a trained drug_rec ABC run dir.")
    p.add_argument("--top_k", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run)
    run_args = load_run_args(run_dir)

    if run_args["task"] != "drug_rec" or run_args["variant"] != "ABC":
        raise SystemExit(
            f"copy_weights.py only applies to drug_rec ABC runs (got "
            f"task={run_args['task']}, variant={run_args['variant']})."
        )

    print(f"[copy_weights] loading {run_dir}")
    samples = build_samples(run_args)
    model = build_model(samples, run_args)
    load_checkpoint(model, run_dir)

    label_vocab = samples.output_processors["drugs"].label_vocab
    # label_vocab is {code -> id}; invert so we can name rows
    id_to_code = {v: k for k, v in label_vocab.items()}

    w_recent = model.w_recent.detach().cpu().numpy()
    w_any    = model.w_any.detach().cpu().numpy()
    L = w_recent.shape[0]

    rows = []
    for i in range(L):
        rows.append({
            "label_id": i,
            "drug_code": id_to_code.get(i, "<unk>"),
            "w_recent": float(w_recent[i]),
            "w_any":    float(w_any[i]),
        })
    rows.sort(key=lambda r: r["w_recent"], reverse=True)

    out_path = run_dir / "copy_weights.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label_id", "drug_code", "w_recent", "w_any"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[copy_weights] wrote {out_path} ({L} rows)")

    print(f"\nTop-{args.top_k} drugs by w_recent (most likely to be CARRIED FORWARD from history):")
    print(f"  {'drug_code':<10} {'w_recent':>10} {'w_any':>10}")
    for r in rows[: args.top_k]:
        print(f"  {r['drug_code']:<10} {r['w_recent']:>10.4f} {r['w_any']:>10.4f}")

    print(f"\nBottom-{args.top_k} drugs by w_recent (model learned NOT to copy from history):")
    print(f"  {'drug_code':<10} {'w_recent':>10} {'w_any':>10}")
    for r in rows[-args.top_k:][::-1]:
        print(f"  {r['drug_code']:<10} {r['w_recent']:>10.4f} {r['w_any']:>10.4f}")

    n_pos = sum(1 for r in rows if r["w_recent"] > 0)
    n_neg = sum(1 for r in rows if r["w_recent"] < 0)
    print(f"\nSummary: {n_pos}/{L} drugs with w_recent>0, {n_neg}/{L} with w_recent<0")


if __name__ == "__main__":
    main()
