# -*- coding: utf-8 -*-
"""au-prior v2: 턴 버킷(0-1 / 2+) 조건부 bias — cross-fit 검증.

v1(전역 au bias, ALL +0.0023) 대비 추가 이득이 있는지. 게이트: v1 대비 +0.001 이상.
"""
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
lowturn = np.array([turn_of[i] <= 1 for i in ids])


def macro(yy, ll):
    return f1_score(yy, ll.argmax(1), average="macro")


def probs(ll):
    e = np.exp(ll - ll.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def crossfit(groups, taus=(0.25, 0.5, 0.75, 1.0)):
    """groups: list of boolean masks (au 하위그룹). 각 그룹별 bias+tau를 cross-fit."""
    adj = lg.copy()
    for f in range(5):
        for g in groups:
            tr = (fold != f) & g
            va = (fold == f) & g
            if tr.sum() < 100 or va.sum() == 0:
                continue
            p = np.bincount(y[tr], minlength=n_cls).astype(np.float64)
            p /= p.sum()
            pm = probs(lg[tr]).mean(0)
            b = np.log(p + 1e-9) - np.log(pm + 1e-9)
            best_t, best = 0.0, macro(y[tr], lg[tr])
            for t in taus:
                sc = macro(y[tr], lg[tr] + t * b[None, :])
                if sc > best + 1e-6:
                    best_t, best = t, sc
            adj[va] = lg[va] + best_t * b[None, :]
    return adj


base = macro(y, lg)
v1 = crossfit([au])
v2 = crossfit([au & lowturn, au & ~lowturn])
print(f"base                ALL={base:.4f}  au={macro(y[au], lg[au]):.4f}  au-low={macro(y[au&lowturn], lg[au&lowturn]):.4f}")
print(f"v1 (au 전역)        ALL={macro(y, v1):.4f}  au={macro(y[au], v1[au]):.4f}  au-low={macro(y[au&lowturn], v1[au&lowturn]):.4f}")
print(f"v2 (au x 턴버킷)    ALL={macro(y, v2):.4f}  au={macro(y[au], v2[au]):.4f}  au-low={macro(y[au&lowturn], v2[au&lowturn]):.4f}")
print(f"au-low n={int((au&lowturn).sum())}, au-high n={int((au&~lowturn).sum())}")
print("게이트: v2 ALL >= v1 ALL + 0.001 이면 v2 채택")
