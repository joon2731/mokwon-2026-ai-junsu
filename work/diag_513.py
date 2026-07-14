# -*- coding: utf-8 -*-
"""5.13 측 진단: ① 같은 텍스트의 토큰화 비교 ② 같은 input_ids의 로짓 비교."""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

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

tok = AutoTokenizer.from_pretrained(MDIR)
lens = [len(x) for x in tok(texts, truncation=True, max_length=512)["input_ids"]]
long_idx = sorted(range(N), key=lambda i: -lens[i])[:32]

# ① 토큰화 비교
ids_446 = np.load(os.path.join(ART, "diag_446_ids.npy"), allow_pickle=True)
ids_513 = tok([texts[i] for i in long_idx], truncation=True, max_length=512)["input_ids"]
mismatch = 0
for a, b in zip(ids_446, ids_513):
    if list(a) != list(b):
        mismatch += 1
print(f"토큰화 불일치: {mismatch}/32  (0이면 토크나이저 무죄)", flush=True)

# ② 같은 input_ids 강제 투입 → 로짓 비교용 저장
model = AutoModelForSequenceClassification.from_pretrained(MDIR).to("cuda").eval().half()
batch = ids_513[:8]
maxlen = max(len(x) for x in batch)
pad_id = tok.pad_token_id
input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
for i, seq in enumerate(batch):
    input_ids[i, :len(seq)] = torch.tensor(seq)
    attn[i, :len(seq)] = 1
with torch.no_grad():
    lg = model(input_ids=input_ids.to("cuda"), attention_mask=attn.to("cuda")).logits.float().cpu().numpy()
np.save(os.path.join(ART, "diag_fixed_inputs.npy"), input_ids.numpy())
np.save(os.path.join(ART, "diag_fixed_attn.npy"), attn.numpy())
np.save(os.path.join(ART, "diag_logits_513.npy"), lg)
print("고정 입력 로짓 저장 (5.13):", lg.argmax(1).tolist(), flush=True)
