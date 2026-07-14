# -*- coding: utf-8 -*-
"""Morning dashboard: overnight logs + all OOF scores + GPU + next steps.

Usage: python work\\status.py
"""
import glob
import os
import re
import subprocess

import numpy as np
from sklearn.metrics import f1_score

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")

KEY = re.compile(r"=====|RESULT|summary:|MACRO-F1|OVERNIGHT|RETRY|Traceback|waiting for|cooling")


def show_log(name):
    p = os.path.join(ART, name)
    print(f"\n== {name} ==")
    if not os.path.exists(p):
        print("  (없음)")
        return
    lines = [l.rstrip() for l in open(p, encoding="utf-8", errors="replace")]
    hits = [l for l in lines if KEY.search(l)]
    for l in hits[-30:]:
        print(" ", l)
    if lines and not KEY.search(lines[-1]):
        print("  [마지막 줄]", lines[-1][:160])


def oof_table():
    print("\n== OOF macro-F1 (artifacts/oof) ==")
    rows = {}
    for p in sorted(glob.glob(os.path.join(ART, "oof", "*.npz"))):
        base = os.path.basename(p)[:-4]
        m = re.match(r"(.+)_fold(\d)$", base)
        if not m:
            continue
        tag, fold = m.group(1), int(m.group(2))
        d = np.load(p, allow_pickle=True)
        sc = f1_score(d["y"], d["logits"].argmax(1), average="macro")
        rows.setdefault(tag, {})[fold] = sc
    for tag, fs in sorted(rows.items()):
        cells = " ".join(f"f{k}={v:.4f}" for k, v in sorted(fs.items()))
        mean = np.mean(list(fs.values()))
        full = "OK(5/5)" if len(fs) == 5 else f"{len(fs)}/5"
        print(f"  {tag:30s} {cells}")
        print(f"  {'':30s} -> mean={mean:.4f}  [{full}]")


def gpu():
    print("\n== GPU ==")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        print(" ", out)
    except Exception as e:
        print("  nvidia-smi 실패:", e)


if __name__ == "__main__":
    show_log("overnight2.log")
    show_log("overnight2.err")
    show_log("overnight3.log")
    oof_table()
    gpu()
    print("\n다음 단계 → MORNING.md 참고 (Qwen 스모크 / blend_oof.py / package_multi.py)")
