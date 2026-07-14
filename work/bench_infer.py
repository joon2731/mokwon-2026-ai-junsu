# -*- coding: utf-8 -*-
"""Measure fp16 inference speed of a trained model dir; estimate T4 submit time.

Usage: python work\\bench_infer.py --model artifacts\\models\\qwen05_smoke_fold0_best
       [--n 3000 --max_len 512 --bs 128]

T4 conversion factor 2.75 comes from the measured xlm-r-base pair
(660 samples/s on 4070Ti vs ~240 on the eval T4).
"""
import argparse
import io
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import build_text

ROOT = r"C:\Users\joon2\Desktop\da2"
T4_FACTOR = 2.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    mdir = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)

    rows = [json.loads(l) for l in io.open(
        os.path.join(ROOT, "data", "train.jsonl"), encoding="utf-8")]
    random.seed(0)
    texts = [build_text(r) for r in random.sample(rows, args.n)]

    # 배포 코드(script.predict_probs)를 그대로 호출해 측정 — 측정=배포 보장.
    # 1차 호출(200개)은 워밍업 겸 로드 확인, 2차(전량)를 계측 (로드 1~3s 포함, <5% 왜곡).
    from script import predict_probs
    torch.cuda.synchronize()
    t0 = time.time()
    predict_probs(mdir, texts[:200], args.max_len, batch_size=args.bs)
    torch.cuda.synchronize()
    warm = time.time() - t0

    torch.cuda.synchronize()
    t0 = time.time()
    predict_probs(mdir, texts, args.max_len, batch_size=args.bs)
    torch.cuda.synchronize()
    dt = time.time() - t0

    sps = len(texts) / dt
    t4 = sps / T4_FACTOR
    print(f"local  : {sps:7.1f} samples/s (로드 포함, warm-call {warm:.0f}s)")
    print(f"T4 est : {t4:7.1f} samples/s")
    for n in (30000, 40000):
        tot = n / t4 / 60 + 1.5  # +1.5분: 서버 로드/기록 여유
        print(f"  {n} samples -> {n/t4/60:5.1f}분 + 여유 1.5분 = {tot:5.1f}분 (한도 10)")


if __name__ == "__main__":
    main()
