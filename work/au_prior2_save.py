# -*- coding: utf-8 -*-
"""au-prior v2 최종 bias 저장: 턴버킷(0-1/2+)별 bias, tau는 fold cross-fit 다수결."""
import collections
import io
import json
import os

import numpy as np
from sklearn.metrics import f1_score

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
TAG = "xlmr_v2_rdrop_lr4_e4"
classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
n_cls = len(classes)

turn_of = {}
for line in io.open(os.path.join(ROOT, "data", "train.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    turn_of[r["id"]] = r["session_meta"].get("turn_index", 99)

ids, lg, y, fold = [], [], [], []
for f in range(5):
    d = np.load(os.path.join(ART, "oof", f"{TAG}_fold{f}.npz"), allow_pickle=True)
    ids.append(np.asarray(d["ids"]).astype(str))
    lg.append(d["logits"].astype(np.float64))
    y.append(d["y"])
    fold.append(np.full(len(d["y"]), f))
ids = np.concatenate(ids); lg = np.concatenate(lg); y = np.concatenate(y); fold = np.concatenate(fold)
au = np.array([i.startswith("sess_au_") for i in ids])
low = np.array([turn_of[i] <= 1 for i in ids])


def macro(yy, ll):
    return f1_score(yy, ll.argmax(1), average="macro")


def probs(ll):
    e = np.exp(ll - ll.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


TAUS = (0.25, 0.5, 0.75, 1.0)
out = {"classes": classes}
for name, g in (("low", au & low), ("high", au & ~low)):
    votes = collections.Counter()
    for f in range(5):
        tr = (fold != f) & g
        p = np.bincount(y[tr], minlength=n_cls).astype(np.float64); p /= p.sum()
        pm = probs(lg[tr]).mean(0)
        b = np.log(p + 1e-9) - np.log(pm + 1e-9)
        best_t, best = 0.0, macro(y[tr], lg[tr])
        for t in TAUS:
            sc = macro(y[tr], lg[tr] + t * b[None, :])
            if sc > best + 1e-6:
                best_t, best = t, sc
        votes[best_t] += 1
    tau = votes.most_common(1)[0][0]
    p = np.bincount(y[g], minlength=n_cls).astype(np.float64); p /= p.sum()
    pm = probs(lg[g]).mean(0)
    bias = (np.log(p + 1e-9) - np.log(pm + 1e-9)) * tau
    out[f"au_bias_{name}"] = bias.round(6).tolist()
    out[f"tau_{name}"] = tau
    print(f"{name}: n={int(g.sum())} tau={tau} (votes {dict(votes)})")

op = os.path.join(ART, "au_bias.json")
json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("saved v2 ->", op)
