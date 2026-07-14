# -*- coding: utf-8 -*-
"""Measure real inference throughput of the submission model, to estimate the
hidden test-set size from the observed ~3 min server runtime."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from featurize import build_text

MDIR = r"C:\Users\joon2\Desktop\da2\submit\model\m0"
DATA = r"C:\Users\joon2\Desktop\da2\open\data"
MAXLEN, BS, N = 512, 128, 4000

# build N texts by tiling the 5 sample test records
recs = []
with open(os.path.join(DATA, "test.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            recs.append(json.loads(line))
texts = [build_text(recs[i % len(recs)]) for i in range(N)]

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR)
model = AutoModelForSequenceClassification.from_pretrained(MDIR).cuda().half().eval()
torch.cuda.synchronize()
load_t = time.time() - t0
print(f"model load: {load_t:.1f}s")

# warmup
with torch.no_grad():
    enc = tok(texts[:BS], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
    model(**enc)
torch.cuda.synchronize()

t0 = time.time()
with torch.no_grad():
    for i in range(0, N, BS):
        b = texts[i:i+BS]
        enc = tok(b, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        model(**enc).logits.float().cpu()
torch.cuda.synchronize()
inf_t = time.time() - t0
print(f"inference: {N} samples in {inf_t:.1f}s  ->  {N/inf_t:.1f} samples/sec (RTX 4070 Ti, fp16, seq{MAXLEN}, bs{BS})")
print(f"per 10000 samples: {10000/(N/inf_t):.1f}s on 4070 Ti")
