# -*- coding: utf-8 -*-
"""Driver #4: Qwen2.5-Coder-0.5B folds 1-4 (스모크와 동일 레시피, 같은 태그로 5-fold 완성)."""
import ctypes
import subprocess
import sys
import time

ROOT = r"C:\Users\joon2\Desktop\da2"
PY = sys.executable

ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def run(name, args, retries=1):
    rc = 1
    for attempt in range(retries + 1):
        print(f"\n===== START {name} (attempt {attempt + 1}/{retries + 1}) =====", flush=True)
        t0 = time.time()
        rc = subprocess.run([PY] + args, cwd=ROOT).returncode
        print(f"===== END {name} rc={rc} ({(time.time()-t0)/60:.1f} min) =====", flush=True)
        if rc == 0:
            return 0
        if attempt < retries:
            print(f"RETRY {name}: rc={rc}; waiting 60s", flush=True)
            time.sleep(60)
    return rc


for f in (1, 2, 3, 4):
    run(f"qwen fold{f}", [
        "work/train.py", "--model", "pretrained\\Qwen2.5-Coder-0.5B", "--fold", str(f),
        "--tag", "qwen05_smoke", "--max_len", "512", "--epochs", "3",
        "--bs", "8", "--grad_accum", "4", "--precision", "bf16",
        "--lr", "2e-5", "--warmup", "0.1", "--optim", "adamw_bnb_8bit",
        "--grad_ckpt", "--weighting", "sqrt"])

print("\nOVERNIGHT4 DONE (qwen folds 1-4)", flush=True)
