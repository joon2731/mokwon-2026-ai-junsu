# -*- coding: utf-8 -*-
"""서버 재현: transformers 4.46.3 환경에서 Qwen fold0 모델을 fp16으로 채점.

- fold0 val 3000행 macro 측정
- 학습 OOF(5.13 로짓)와 행 단위 argmax 일치율 비교 → 어디서 갈라지는지 특정
- 토큰화 일치 여부도 확인 (같은 텍스트 → 같은 input_ids?)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
MDIR = os.path.join(ART, "models", "qwen05_smoke_fold0_best")
N = 3000

import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
print("transformers:", transformers.__version__, flush=True)

df = pd.read_parquet(os.path.join(ART, "train_prepared.parquet"))
va = df[df.fold == 0].reset_index(drop=True).iloc[:N]
texts = va.text.tolist()
y = va.y.values
ids = va.id.values.astype(str)

tok = AutoTokenizer.from_pretrained(MDIR)
print("padding_side:", tok.padding_side, "| pad:", tok.pad_token, flush=True)

model = AutoModelForSequenceClassification.from_pretrained(MDIR).to("cuda").eval().half()
print("model pad_token_id:", model.config.pad_token_id, flush=True)

lens = [len(x) for x in tok(texts, truncation=True, max_length=512)["input_ids"]]
order = sorted(range(N), key=lambda i: -lens[i])
probs = np.zeros((N, 14))
with torch.no_grad():
    s = 0
    while s < N:
        bl = lens[order[s]]
        bs = max(8, min(128, (128 * 512) // max(bl, 1)))
        idx = order[s:s + bs]
        s += len(idx)
        enc = tok([texts[i] for i in idx], truncation=True, max_length=512,
                  padding=True, return_tensors="pt").to("cuda")
        lg = model(**enc).logits.float().cpu().numpy()
        probs[idx] = lg

macro = f1_score(y, probs.argmax(1), average="macro")
print(f"\n[4.46.3] fold0-val {N}행 macro = {macro:.4f}  (5.13 학습시 0.7571 / 로컬재현 0.7645)", flush=True)

# 학습 OOF(5.13)와 행 단위 비교
d = np.load(os.path.join(ART, "oof", "qwen05_smoke_fold0.npz"), allow_pickle=True)
oof_map = dict(zip(np.asarray(d["ids"]).astype(str), d["logits"].argmax(1)))
pr446 = probs.argmax(1)
same = np.mean([oof_map[i] == p for i, p in zip(ids, pr446)])
print(f"5.13 OOF와 argmax 일치율: {same:.1%}", flush=True)

# 토큰화 비교 (더 긴 텍스트 위주 32개)
long_idx = sorted(range(N), key=lambda i: -lens[i])[:32]
np.save(os.path.join(ART, "diag_446_ids.npy"),
        np.array(tok([texts[i] for i in long_idx], truncation=True, max_length=512)["input_ids"], dtype=object),
        allow_pickle=True)
print("토큰화 샘플 저장 -> diag_446_ids.npy (5.13측과 비교용)", flush=True)
