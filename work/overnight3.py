# -*- coding: utf-8 -*-
"""Overnight driver #3 — chained AFTER overnight2.

Waits for overnight2's PID to exit (whatever the reason), cools 90s for VRAM
release, then trains folds 1-4 of the confirmed best recipe
(xlm-roberta-base · 512 · R-Drop 1.0 · LR 4e-5 · 4 epochs) under the SAME tag
as tonight's fold-0 run, completing the 5-fold set by morning.

Usage: python work\overnight3.py <overnight2_pid>
"""
import ctypes
import os
import subprocess
import sys
import time

ROOT = r"C:\Users\joon2\Desktop\da2"
PY = sys.executable
WAIT_PID = int(sys.argv[1]) if len(sys.argv) > 1 else 0

ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

STILL_ACTIVE = 259


def alive(pid):
    if pid <= 0:
        return False
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return bool(ok) and code.value == STILL_ACTIVE


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


def main():
    print(f"waiting for overnight2 (pid {WAIT_PID}) to exit ...", flush=True)
    while alive(WAIT_PID):
        time.sleep(60)
    print("overnight2 gone; cooling 90s for VRAM release", flush=True)
    time.sleep(90)

    for f in (1, 2, 3, 4):
        run(f"fold{f} base 4ep", [
            "work/train.py", "--model", "xlm-roberta-base", "--fold", str(f),
            "--tag", "xlmr_v2_rdrop_lr4_e4", "--max_len", "512", "--epochs", "4",
            "--bs", "8", "--grad_accum", "4", "--precision", "bf16",
            "--rdrop", "1.0", "--lr", "4e-5"])

    print("\nOVERNIGHT3 DONE (folds 1-4, tag=xlmr_v2_rdrop_lr4_e4)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
