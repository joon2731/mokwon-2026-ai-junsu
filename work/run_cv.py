# -*- coding: utf-8 -*-
"""Phase 2: run all folds sequentially by invoking train.py, so a crash in one
fold doesn't lose the others. Usage:
    python work/run_cv.py --model xlm-roberta-base --tag xlmr_v2 --max_len 320 --epochs 3 --folds 0,1,2,3,4
"""
import argparse
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--tag", default="xlmr_v2")
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--weighting", default="sqrt")
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",")]
    for f in folds:
        cmd = [sys.executable, os.path.join(HERE, "train.py"),
               "--model", args.model, "--tag", args.tag, "--fold", str(f),
               "--max_len", str(args.max_len), "--epochs", str(args.epochs),
               "--bs", str(args.bs), "--grad_accum", str(args.grad_accum),
               "--lr", str(args.lr), "--precision", args.precision,
               "--weighting", args.weighting]
        print(f"\n===== FOLD {f} =====\n{' '.join(cmd)}\n", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"FOLD {f} FAILED rc={r.returncode}", flush=True)
            sys.exit(r.returncode)
    print("\nALL FOLDS DONE", flush=True)


if __name__ == "__main__":
    main()
