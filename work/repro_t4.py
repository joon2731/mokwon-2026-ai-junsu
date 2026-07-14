# -*- coding: utf-8 -*-
"""T4 조건 재현: flash/mem-efficient SDPA 끄고(math 경로) fp16으로 fold0 val 채점.

가설: T4(플래시 미지원)의 fp16 math-attention에서 오버플로우 → 예측 붕괴.
비교: ①fp16+math(=T4 재현) ②fp16+flash(=로컬 기본) ③fp32가중치+fp16 autocast(수정안)
각각 fold0 val 3000행 macro + NaN 로짓 수 측정.
"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
MDIR = os.path.join(ART, "models", "qwen05_smoke_fold0_best")
N = 3000

df = pd.read_parquet(os.path.join(ART, "train_prepared.parquet"))
va = df[df.fold == 0].reset_index(drop=True).iloc[:N]
texts = va.text.tolist()
y = va.y.values

tok = AutoTokenizer.from_pretrained(MDIR)
lens = [len(x) for x in tok(texts, truncation=True, max_length=512)["input_ids"]]
order = sorted(range(len(texts)), key=lambda i: -lens[i])


def run(model, autocast_fp16=False):
    probs = np.zeros((len(texts), 14))
    nan_rows = 0
    with torch.no_grad():
        s = 0
        while s < len(texts):
            bl = lens[order[s]]
            bs = max(8, min(128, (128 * 512) // max(bl, 1)))
            idx = order[s:s + bs]
            s += len(idx)
            enc = tok([texts[i] for i in idx], truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to("cuda")
            if autocast_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = model(**enc).logits
            else:
                lg = model(**enc).logits
            lg = lg.float().cpu().numpy()
            nan_rows += int(np.isnan(lg).any(1).sum() + np.isinf(lg).any(1).sum())
            probs[idx] = np.nan_to_num(lg, nan=0.0, posinf=0.0, neginf=0.0)
    return probs, nan_rows


def report(name, probs, nan_rows, dt):
    m = f1_score(y, probs.argmax(1), average="macro")
    print(f"{name:34s} macro={m:.4f}  NaN/inf행={nan_rows}  ({N/dt:.0f}/s)", flush=True)


from torch.nn.attention import SDPBackend, sdpa_kernel

# ① T4 재현: fp16 가중치 + math 어텐션만 허용
model = AutoModelForSequenceClassification.from_pretrained(MDIR).to("cuda").eval().half()
t0 = time.time()
with sdpa_kernel(SDPBackend.MATH):
    p, nr = run(model)
report("1) fp16 + MATH attn (T4 재현)", p, nr, time.time() - t0)

# ② 로컬 기본: fp16 + flash 허용
t0 = time.time()
p, nr = run(model)
report("2) fp16 + FLASH attn (로컬 기본)", p, nr, time.time() - t0)
del model
torch.cuda.empty_cache()

# ③ 수정안: fp32 가중치 + fp16 autocast (+ math 강제 = T4 최악 조건)
model = AutoModelForSequenceClassification.from_pretrained(MDIR).to("cuda").eval().float()
t0 = time.time()
with sdpa_kernel(SDPBackend.MATH):
    p, nr = run(model, autocast_fp16=True)
report("3) fp32 + fp16 autocast + MATH", p, nr, time.time() - t0)
