# -*- coding: utf-8 -*-
"""Overnight autonomous driver — zero human/Claude intervention needed.

Sequence:
  1. regenerate train_prepared.parquet with featurize V2
  2. fold-0 plain V2 baseline          (~40 min)
  3. fold-0 V2 + R-Drop(alpha=1.0)     (~80 min; OOM/crash -> falls back)
  4. winner config on folds 1-4        (consistent 5-fold OOF for the winner)
  5. print per-fold macros + mean

Safety: blocks Windows system sleep while running (display may still sleep);
any fold failure stops cleanly with a FATAL line (no retry storms).
"""
import ctypes
import os
import subprocess
import sys
import time

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
PY = sys.executable

# prevent system sleep while this process lives (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

BASE = ["work/train.py", "--model", "xlm-roberta-base", "--max_len", "512",
        "--epochs", "3", "--bs", "8", "--grad_accum", "4", "--precision", "bf16"]


def run(name, args, retries=1):
    """Run a step with up to `retries` fresh-process retries.

    One retry covers transient failures (driver hiccup, fragmentation OOM);
    deterministic failures (code bug, hard OOM) just fail once more cheaply —
    no infinite retry storms.
    """
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
            print(f"RETRY {name}: rc={rc}; waiting 60s for fresh GPU state", flush=True)
            time.sleep(60)
    return rc


def macro_of(tag, fold=0):
    import numpy as np
    from sklearn.metrics import f1_score
    d = np.load(os.path.join(ART, "oof", f"{tag}_fold{fold}.npz"), allow_pickle=True)
    return float(f1_score(d["y"], d["logits"].argmax(1), average="macro"))


def main():
    if run("prepare V2 data", ["work/prepare_data.py"]):
        print("FATAL: data prep failed", flush=True)
        return 1

    if run("fold0 plain V2", BASE + ["--fold", "0", "--tag", "xlmr_v2"]):
        print("FATAL: fold0 plain failed", flush=True)
        return 1
    m_plain = macro_of("xlmr_v2")

    rc_rd = run("fold0 R-Drop V2", BASE + ["--fold", "0", "--tag", "xlmr_v2_rdrop", "--rdrop", "1.0"])
    m_rdrop = macro_of("xlmr_v2_rdrop") if rc_rd == 0 else -1.0
    print(f"\nFOLD0 RESULT: plain={m_plain:.4f}  rdrop={m_rdrop:.4f}", flush=True)

    # adopt R-Drop only on a clear win (single-fold noise ~ +-0.004)
    if rc_rd == 0 and m_rdrop >= m_plain + 0.002:
        tag, extra = "xlmr_v2_rdrop", ["--rdrop", "1.0"]
    else:
        tag, extra = "xlmr_v2", []
    print(f"WINNER -> tag={tag}", flush=True)

    # skip-and-continue: one dead fold must not kill the night
    failed = []
    for f in (1, 2, 3, 4):
        if run(f"fold{f} {tag}", BASE + ["--fold", str(f), "--tag", tag] + extra):
            print(f"SKIP fold{f}: failed twice, continuing with remaining folds", flush=True)
            failed.append(f)

    import numpy as np
    done = [f for f in range(5) if f not in failed]
    ms = [macro_of(tag, f) for f in done]
    print("\nOVERNIGHT DONE tag=" + tag, flush=True)
    if failed:
        print("FAILED FOLDS:", failed, "(retrain in the morning)", flush=True)
    print(f"fold macros ({len(done)}/5 folds):", [round(m, 4) for m in ms],
          "MEAN", round(float(np.mean(ms)), 4), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
