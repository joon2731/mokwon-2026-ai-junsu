# -*- coding: utf-8 -*-
"""sim 행 대상 메타 서브그룹 bias cross-fit (au v2 위에 얹었을 때 추가 이득 측정).

후보 그룹: dirty=False / elapsed=0 / ci=none (모델이 약한 곳들).
게이트: au-v2 기준 ALL OOF(0.7395) 대비 +0.001 이상.
"""
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

meta = {}
for line in io.open(os.path.join(ROOT, "data", "train.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    m = r.get("session_meta", {}) or {}
    ws = m.get("workspace", {}) or {}
    meta[r["id"]] = (bool(ws.get("git_dirty", False)),
                     int(m.get("elapsed_session_sec", 0)) // 600,
                     str(ws.get("last_ci_status")),
                     int(m.get("turn_index", 99)))

ids, lg, y, fold = [], [], [], []
for f in range(5):
    d = np.load(os.path.join(ART, "oof", f"{TAG}_fold{f}.npz"), allow_pickle=True)
    ids.append(np.asarray(d["ids"]).astype(str))
    lg.append(d["logits"].astype(np.float64))
    y.append(d["y"])
    fold.append(np.full(len(d["y"]), f))
ids = np.concatenate(ids); lg = np.concatenate(lg); y = np.concatenate(y); fold = np.concatenate(fold)
au = np.array([i.startswith("sess_au_") for i in ids])
dirty = np.array([meta[i][0] for i in ids])
el0 = np.array([meta[i][1] == 0 for i in ids])
ci_none = np.array([meta[i][2] == "none" for i in ids])
low = np.array([meta[i][3] <= 1 for i in ids])


def macro(yy, ll):
    return f1_score(yy, ll.argmax(1), average="macro")


def probs(ll):
    e = np.exp(ll - ll.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def crossfit(base_lg, groups, taus=(0.25, 0.5, 0.75, 1.0)):
    adj = base_lg.copy()
    for f in range(5):
        for g in groups:
            tr = (fold != f) & g
            va = (fold == f) & g
            if tr.sum() < 200 or va.sum() == 0:
                continue
            p = np.bincount(y[tr], minlength=n_cls).astype(np.float64); p /= p.sum()
            pm = probs(base_lg[tr]).mean(0)
            b = np.log(p + 1e-9) - np.log(pm + 1e-9)
            best_t, best = 0.0, macro(y[tr], base_lg[tr])
            for t in taus:
                sc = macro(y[tr], base_lg[tr] + t * b[None, :])
                if sc > best + 1e-6:
                    best_t, best = t, sc
            adj[va] = base_lg[va] + best_t * b[None, :]
    return adj


# 기준: au v2 (au x 턴버킷) 적용본
au_v2 = crossfit(lg, [au & low, au & ~low])
print(f"기준 au-v2            ALL={macro(y, au_v2):.4f}")

sim = ~au
cands = {
    "sim x dirty(F/T)": [sim & ~dirty, sim & dirty],
    "sim x elapsed0(T/F)": [sim & el0, sim & ~el0],
    "sim x ci_none(T/F)": [sim & ci_none, sim & ~ci_none],
    "sim x lowturn(T/F)": [sim & low, sim & ~low],
}
for name, groups in cands.items():
    adj = crossfit(au_v2, groups)
    print(f"+ {name:22s} ALL={macro(y, adj):.4f}  ({macro(y, adj)-macro(y, au_v2):+.4f})")
