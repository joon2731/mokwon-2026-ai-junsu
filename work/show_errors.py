# -*- coding: utf-8 -*-
"""Pull misclassified, genuinely-ambiguous examples (small top-2 margin) from
the fold-0 512 OOF, focused on the nav cluster confusions."""
import json
import os
import numpy as np
import pandas as pd

DATA = r"C:\Users\joon2\Desktop\da2\open\data"
ART = r"C:\Users\joon2\Desktop\da2\artifacts"
OUT = os.path.join(ART, "error_examples.txt")

classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
nav = {"read_file", "grep_search", "glob_pattern", "list_directory"}

recs = {}
with open(os.path.join(DATA, "train.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            recs[r["id"]] = r

d = np.load(os.path.join(ART, "oof", "xlmr_len512_fold0.npz"), allow_pickle=True)
ids, logits, y = d["ids"], d["logits"].astype(np.float64), d["y"]
e = np.exp(logits - logits.max(1, keepdims=True))
probs = e / e.sum(1, keepdims=True)
pred = probs.argmax(1)

srt = np.argsort(probs, 1)
top1 = srt[:, -1]; top2 = srt[:, -2]
margin = probs[np.arange(len(y)), top1] - probs[np.arange(len(y)), top2]

# misclassified, true is a nav action, and the model was torn (small margin)
mask = (pred != y) & np.array([classes[t] in nav for t in y])
cand = [i for i in np.argsort(margin) if mask[i]]

def last_ctx(r):
    h = r.get("history") or []
    out = []
    for t in h[-2:]:
        if t.get("role") == "user":
            out.append("u: " + (t.get("content", "") or "")[:70])
        else:
            a = t.get("args", {}) or {}
            av = a.get("path") or a.get("pattern") or a.get("query") or a.get("target") or ""
            out.append(f"a: {t.get('name')}({av})")
    return "  |  ".join(out)

with open(OUT, "w", encoding="utf-8") as o:
    for n, i in enumerate(cand[:12], 1):
        r = recs[ids[i]]
        m = r["session_meta"]; ws = m["workspace"]
        o.write(f"[{n}] lang={m.get('language_pref')}  turn={m.get('turn_index')}  open={len(ws.get('open_files') or [])}\n")
        o.write(f"    PROMPT: {r.get('current_prompt','')}\n")
        o.write(f"    직전문맥: {last_ctx(r)}\n")
        o.write(f"    정답: {classes[y[i]]} (p={probs[i, y[i]]:.2f})   |   모델: {classes[top1[i]]} (p={probs[i, top1[i]]:.2f})\n\n")
print("saved", OUT)
