# -*- coding: utf-8 -*-
"""Driver: Qwen3-0.6B-Base folds 1-4 (fold0와 동일 레시피로 5-fold OOF 완성).

fold0 확정 레시피(training_args.bin 대조): max_len512 · 3ep · bs8 · ga4 · bf16 ·
lr2e-5 · warmup0.1 · wd0.01 · adamw_bnb_8bit · grad_ckpt · sqrt · seed42 · cosine.

크래시 대비: 해당 fold의 OOF가 이미 있으면 스킵(재실행 시 이어감). 컴퓨터가 또
꺼지면 이 스크립트를 다시 실행만 하면 남은 fold부터 진행됨.
"""
import ctypes
import os
import subprocess
import sys
import time

ROOT = r"C:\Users\joon2\Desktop\da2"
PY = sys.executable
OOF = os.path.join(ROOT, "artifacts", "oof")

# 절전/화면보호기로 인한 중단 방지 (Windows)
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception:
    pass


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
    oof_path = os.path.join(OOF, f"qwen3_smoke_fold{f}.npz")
    if os.path.exists(oof_path):
        print(f"[skip] fold{f}: OOF 이미 존재 ({oof_path})", flush=True)
        continue
    run(f"qwen3 fold{f}", [
        "work/train.py", "--model", "pretrained\\Qwen3-0.6B-Base", "--fold", str(f),
        "--tag", "qwen3_smoke", "--max_len", "512", "--epochs", "3",
        "--bs", "8", "--grad_accum", "4", "--precision", "bf16",
        "--lr", "2e-5", "--warmup", "0.1", "--wd", "0.01", "--optim", "adamw_bnb_8bit",
        "--grad_ckpt", "--weighting", "sqrt", "--seed", "42"])

print("\nOVERNIGHT_QWEN3 DONE (qwen3 folds 1-4)", flush=True)
