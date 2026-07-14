# -*- coding: utf-8 -*-
"""Does XLM-R + mDeBERTa ensemble beat XLM-R alone on fold-0 OOF?"""
import os
import numpy as np
from sklearn.metrics import f1_score

ART = r"C:\Users\joon2\Desktop\da2\artifacts"

def load(tag):
    d = np.load(os.path.join(ART, "oof", f"{tag}_fold0.npz"), allow_pickle=True)
    logits = d["logits"].astype(np.float64)
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    return dict(zip(d["ids"], p)), dict(zip(d["ids"], d["y"]))

pa, ya = load("xlmr_len512")
pb, yb = load("mdeberta_fp32")
ids = [i for i in pa if i in pb]
A = np.array([pa[i] for i in ids])
B = np.array([pb[i] for i in ids])
y = np.array([ya[i] for i in ids])
print(f"aligned {len(ids)} samples")

mac = lambda P: f1_score(y, P.argmax(1), average="macro")
print(f"XLM-R-512 alone   : {mac(A):.4f}")
print(f"mDeBERTa alone    : {mac(B):.4f}")
print("--- weighted ensemble (w*XLMR + (1-w)*mDeBERTa) ---")
best = (0, 0)
for w in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
    m = mac(w * A + (1 - w) * B)
    flag = "  <== beats XLM-R" if m > mac(A) else ""
    print(f"  w={w:.2f}: {m:.4f}{flag}")
    if m > best[1]:
        best = (w, m)
print(f"\nbest ensemble: w={best[0]:.2f} -> {best[1]:.4f}  (XLM-R alone {mac(A):.4f}, delta {best[1]-mac(A):+.4f})")
