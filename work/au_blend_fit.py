# -*- coding: utf-8 -*-
"""블렌드(Qwen+XLM-R 가중 prob 평균) 위에 au 턴버킷 bias를 재적합.

script.py의 추론 순서와 동일하게: probs 가중평균 → log → au bias 가산 → argmax.
cross-fit으로 정직 이득 측정 후 최종 bias를 artifacts/au_bias.json(v2 포맷)으로 저장.
"""
import io
import json
import os

import numpy as np
from sklearn.metrics import f1_score

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
TAGS = ["qwen05_smoke", "xlmr_v2_rdrop_lr4_e4"]
W = [0.627, 0.373]  # blend_oof cross-fit 가중치
classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
n_cls = len(classes)

turn_of = {}
for line in io.open(os.path.join(ROOT, "data", "train.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    turn_of[r["id"]] = r["session_meta"].get("turn_index", 99)


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def load_tag(tag):
    out = {}
    for f in range(5):
        d = np.load(os.path.join(ART, "oof", f"{tag}_fold{f}.npz"), allow_pickle=True)
        ids = np.asarray(d["ids"]).astype(str)
        order = np.argsort(ids)
        out[f] = (ids[order], softmax(d["logits"].astype(np.float64)[order]), d["y"][order])
    return out


A, B = load_tag(TAGS[0]), load_tag(TAGS[1])
ids, probs, y, fold = [], [], [], []
for f in range(5):
    ia, pa, ya = A[f]
    ib, pb, yb = B[f]
    assert np.array_equal(ia, ib), f"fold{f} id 불일치"
    ids.append(ia)
    probs.append(W[0] * pa + W[1] * pb)
    y.append(ya)
    fold.append(np.full(len(ya), f))
ids = np.concatenate(ids)
probs = np.concatenate(probs)
y = np.concatenate(y)
fold = np.concatenate(fold)
scores = np.log(probs + 1e-9)  # script.py와 동일

au = np.array([i.startswith("sess_au_") for i in ids])
low = np.array([turn_of[i] <= 1 for i in ids])


def macro(yy, ss):
    return f1_score(yy, ss.argmax(1), average="macro")


base = macro(y, scores)
print(f"블렌드 base OOF        : {base:.4f}")
print(f"  sim={macro(y[~au], scores[~au]):.4f}  au={macro(y[au], scores[au]):.4f}")

TAUS = (0.25, 0.5, 0.75, 1.0)
adj = scores.copy()
for f in range(5):
    for g in (au & low, au & ~low):
        tr = (fold != f) & g
        va = (fold == f) & g
        if tr.sum() < 200 or va.sum() == 0:
            continue
        p = np.bincount(y[tr], minlength=n_cls).astype(np.float64)
        p /= p.sum()
        pm = probs[tr].mean(0)
        b = np.log(p + 1e-9) - np.log(pm + 1e-9)
        best_t, best = 0.0, macro(y[tr], scores[tr])
        for t in TAUS:
            sc = macro(y[tr], scores[tr] + t * b[None, :])
            if sc > best + 1e-6:
                best_t, best = t, sc
        adj[va] = scores[va] + best_t * b[None, :]

print(f"+ au 턴버킷 (cross-fit): {macro(y, adj):.4f}  ({macro(y, adj)-base:+.4f})")
print(f"  au {macro(y[au], scores[au]):.4f} -> {macro(y[au], adj[au]):.4f}")

# 최종 bias 저장 (전체 au로, tau=다수결)
import collections
out = {"classes": classes}
for name, g in (("low", au & low), ("high", au & ~low)):
    votes = collections.Counter()
    for f in range(5):
        tr = (fold != f) & g
        p = np.bincount(y[tr], minlength=n_cls).astype(np.float64); p /= p.sum()
        pm = probs[tr].mean(0)
        b = np.log(p + 1e-9) - np.log(pm + 1e-9)
        best_t, best = 0.0, macro(y[tr], scores[tr])
        for t in TAUS:
            sc = macro(y[tr], scores[tr] + t * b[None, :])
            if sc > best + 1e-6:
                best_t, best = t, sc
        votes[best_t] += 1
    tau = votes.most_common(1)[0][0]
    p = np.bincount(y[g], minlength=n_cls).astype(np.float64); p /= p.sum()
    pm = probs[g].mean(0)
    bias = (np.log(p + 1e-9) - np.log(pm + 1e-9)) * tau
    out[f"au_bias_{name}"] = bias.round(6).tolist()
    out[f"tau_{name}"] = tau
    print(f"{name}: tau={tau} (votes {dict(votes)})")

op = os.path.join(ART, "au_bias.json")
json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("saved ->", op)
