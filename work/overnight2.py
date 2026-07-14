# -*- coding: utf-8 -*-
"""Overnight driver #2 — autonomous, zero-intervention.

  1. fold-0 base recipe with 4 EPOCHS   (epoch-headroom check, ~70 min)
  2. xlm-roberta-large fold-0 with the stability protocol; up to 3 attempts:
       (seed 42, lr 8e-6, R-Drop)  ->  (seed 43, lr 5e-6, R-Drop)
       -> (seed 42, lr 8e-6, NO R-Drop, bigger batch)   [OOM/instability fallback]
     an attempt "succeeds" if fold-0 macro-F1 > 0.55 (large is known to
     collapse to near-random on bad runs — retry instead of trusting it).
  3. summary.

Blocks system sleep; per-step 1 retry for transient failures.
"""
import ctypes
import os
import subprocess
import sys
import time

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
PY = sys.executable

ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def run(name, args, retries=1):
    rc = 1
    for attempt in range(retries + 1):
        print(f"\n===== START {name} (attempt {attempt + 1}/{retries + 1}) =====", flush=True)
        t0 = time.time()
        r = subprocess.run([PY] + args, cwd=ROOT)
        rc = r.returncode
        print(f"===== END {name} rc={rc} ({(time.time()-t0)/60:.1f} min) =====", flush=True)
        if rc == 0:
            return 0
        if attempt < retries:
            print(f"RETRY {name}: rc={rc}; waiting 60s", flush=True)
            time.sleep(60)
    return rc


def macro_of(tag, fold=0):
    import numpy as np
    from sklearn.metrics import f1_score
    p = os.path.join(ART, "oof", f"{tag}_fold{fold}.npz")
    if not os.path.exists(p):
        return -1.0
    d = np.load(p, allow_pickle=True)
    return float(f1_score(d["y"], d["logits"].argmax(1), average="macro"))


def main():
    results = {}

    # ---- step 1: 4-epoch base ----
    rc = run("fold0 base 4ep", [
        "work/train.py", "--model", "xlm-roberta-base", "--fold", "0",
        "--tag", "xlmr_v2_rdrop_lr4_e4", "--max_len", "512", "--epochs", "4",
        "--bs", "8", "--grad_accum", "4", "--precision", "bf16",
        "--rdrop", "1.0", "--lr", "4e-5"])
    results["base_4ep"] = macro_of("xlmr_v2_rdrop_lr4_e4") if rc == 0 else -1.0
    print(f"\nBASE-4EP RESULT: {results['base_4ep']:.4f} (3ep reference 0.7231)", flush=True)

    # ---- step 2: xlm-roberta-large attempts ----
    attempts = [
        ("xlmr_large_s42", ["--seed", "42", "--lr", "8e-6", "--rdrop", "1.0",
                            "--bs", "4", "--grad_accum", "8"]),
        ("xlmr_large_s43", ["--seed", "43", "--lr", "5e-6", "--rdrop", "1.0",
                            "--bs", "4", "--grad_accum", "8"]),
        ("xlmr_large_nord", ["--seed", "42", "--lr", "8e-6", "--rdrop", "0",
                             "--bs", "8", "--grad_accum", "4"]),
    ]
    large_tag, large_macro = None, -1.0
    for tag, extra in attempts:
        rc = run(f"large {tag}", [
            "work/train.py", "--model", "xlm-roberta-large", "--fold", "0",
            "--tag", tag, "--max_len", "512", "--epochs", "3",
            "--precision", "bf16", "--warmup", "0.10",
            "--grad_ckpt", "--optim", "adamw_bnb_8bit"] + extra, retries=0)
        m = macro_of(tag) if rc == 0 else -1.0
        print(f"LARGE ATTEMPT {tag}: macro={m:.4f}", flush=True)
        if m > 0.55:
            large_tag, large_macro = tag, m
            print(f"LARGE SUCCESS -> {tag} {m:.4f}", flush=True)
            break
        print(f"LARGE ATTEMPT {tag} judged FAILED (collapse or crash); next config", flush=True)
    results["large"] = large_macro

    print("\nOVERNIGHT2 DONE", flush=True)
    print(f"summary: base4ep={results['base_4ep']:.4f} (ref 0.7231) | "
          f"large={large_macro:.4f} ({large_tag})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
