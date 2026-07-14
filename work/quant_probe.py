# -*- coding: utf-8 -*-
"""1.5B 제약 통과 가능성 프로브: int8/int4 양자화의 용량·T4추론시간 추정.

학습 전에 '이 모델을 제출할 수 있는가'부터 답한다. base 모델(미파인튜닝)로
속도·메모리만 재도 결론은 동일 (분류 헤드는 무시 가능).

Usage: python work\\quant_probe.py
"""
import io
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import build_text

ROOT = r"C:\Users\joon2\Desktop\da2"
MODEL = os.path.join(ROOT, "pretrained", "Qwen2.5-Coder-1.5B")
T4_FACTOR = 2.75
N = 2000
MAXLEN = 512
BS = 64

rows = [json.loads(l) for l in io.open(os.path.join(ROOT, "data", "train.jsonl"), encoding="utf-8")]
random.seed(0)
texts = [build_text(r) for r in random.sample(rows, N)]

from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
lens = [len(x) for x in tok(texts, truncation=True, max_length=MAXLEN)["input_ids"]]
order = sorted(range(N), key=lambda i: -lens[i])


def bench(tag, quant):
    kw = dict(num_labels=14, torch_dtype=torch.float16)
    if quant == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "int4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, **kw)
    if quant == "fp16":
        model = model.to("cuda")
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    load = time.time() - t0
    vram = torch.cuda.memory_allocated() / 1e9

    # warmup + timed
    with torch.no_grad():
        for _ in range(2):
            enc = tok([texts[i] for i in order[:BS]], truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt").to("cuda")
            model(**enc)
        torch.cuda.synchronize()
        t0 = time.time()
        s = 0
        while s < N:
            idx = order[s:s + BS]
            s += len(idx)
            enc = tok([texts[i] for i in idx], truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt").to("cuda")
            model(**enc)
        torch.cuda.synchronize()
        dt = time.time() - t0

    sps = N / dt
    t4 = sps / T4_FACTOR
    est30k = 30000 / t4 / 60
    # 디스크 용량 추정 (safetensors 저장 크기)
    print(f"[{tag:5s}] VRAM={vram:.2f}GB load={load:.0f}s | "
          f"local {sps:.0f}/s → T4 {t4:.0f}/s → 30k {est30k:.1f}분 "
          f"{'✓' if est30k < 8 else '✗ 초과위험'}")
    del model
    torch.cuda.empty_cache()
    return est30k


print(f"모델: {MODEL}\n샘플 {N} · max_len {MAXLEN} · bs {BS}\n")
for tag, q in [("fp16", "fp16"), ("int8", "int8"), ("int4", "int4")]:
    try:
        bench(tag, q)
    except Exception as e:
        print(f"[{tag}] 실패: {type(e).__name__}: {str(e)[:80]}")
print("\n판정: 30k가 T4 8분 이내(로드+여유 감안 10분) + zip 1GB 통과해야 제출 가능")
print("용량: int8 ~1.5GB·int4 ~0.85GB (+ vocab 프루닝시 추가 절감). fp16 3.1GB=불가")
