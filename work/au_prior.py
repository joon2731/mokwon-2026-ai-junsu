# -*- coding: utf-8 -*-
"""au-prior 보정 설계 + OOF 검증 (cross-fit).

au 행에만 additive logit bias b_c = tau * log(P_au(c) / P_model_marginal(c)) 적용.
- tau는 fold-wise cross-fit 그리드 (다른 4 fold의 au 라벨분포로 bias 계산 + tau 선택
  -> 해당 fold au에 적용) => 완전 정직 추정.
- 산출: au/ALL macro 변화, au flip rate, 권장 tau, bias json 저장.
"""
import json
import os

import numpy as np
from sklearn.metrics import f1_score

ART = r"C:\Users\joon2\Desktop\da2\artifacts"
TAG = "xlmr_v2_rdrop_lr4_e4"
classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
n_cls = len(classes)

ids, lg, y, fold = [], [], [], []
for f in range(5):
    d = np.load(os.path.join(ART, "oof", f"{TAG}_fold{f}.npz"), allow_pickle=True)
    ids.append(np.asarray(d["ids"]).astype(str))
    lg.append(d["logits"].astype(np.float64))
    y.append(d["y"])
    fold.append(np.full(len(d["y"]), f))
ids = np.concatenate(ids)
lg = np.concatenate(lg)
y = np.concatenate(y)
fold = np.concatenate(fold)
au = np.array([i.startswith("sess_au_") for i in ids])


def macro(yy, ll):
    return f1_score(yy, ll.argmax(1), average="macro")


def probs(ll):
    e = np.exp(ll - ll.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


base_all = macro(y, lg)
base_au = macro(y[au], lg[au])
base_sim = macro(y[~au], lg[~au])
print(f"base: ALL={base_all:.4f}  sim={base_sim:.4f}  au={base_au:.4f}  (au n={au.sum()})")

TAUS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
# cross-fit: bias & tau learned on 4 folds' au rows, applied to the held-out fold's au rows
adj = lg.copy()
picked = []
for f in range(5):
    tr = (fold != f) & au
    va = (fold == f) & au
    # au label prior from other folds
    p_au = np.bincount(y[tr], minlength=n_cls).astype(np.float64)
    p_au = p_au / p_au.sum()
    # model's implied marginal on those same rows (what the model currently believes)
    p_mod = probs(lg[tr]).mean(0)
    bias = np.log(p_au + 1e-9) - np.log(p_mod + 1e-9)
    # pick tau on the TRAIN side (other folds' au rows)
    best_tau, best_sc = 0.0, macro(y[tr], lg[tr])
    for t in TAUS:
        sc = macro(y[tr], lg[tr] + t * bias[None, :])
        if sc > best_sc + 1e-6:
            best_tau, best_sc = t, sc
    adj[va] = lg[va] + best_tau * bias[None, :]
    picked.append(best_tau)

adj_au = macro(y[au], adj[au])
adj_all = macro(y, adj)
flips = float((adj[au].argmax(1) != lg[au].argmax(1)).mean())
print(f"cross-fit au-prior: au {base_au:.4f} -> {adj_au:.4f}  ({adj_au-base_au:+.4f})")
print(f"                    ALL {base_all:.4f} -> {adj_all:.4f}  ({adj_all-base_all:+.4f})")
print(f"au flip rate: {flips:.1%}  | fold별 tau: {picked}")

# 제출용 bias (전체 au로 계산, tau=fold 중앙값)
tau_star = float(np.median(picked))
p_au = np.bincount(y[au], minlength=n_cls).astype(np.float64)
p_au /= p_au.sum()
p_mod = probs(lg[au]).mean(0)
bias = (np.log(p_au + 1e-9) - np.log(p_mod + 1e-9)) * tau_star
out = {"tau": tau_star, "au_bias": bias.round(6).tolist(), "classes": classes}
op = os.path.join(ART, "au_bias.json")
json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"saved tau={tau_star} -> {op}")
print("게이트: cross-fit au 이득 >= +0.01 && ALL 손실 없음이면 채택")
