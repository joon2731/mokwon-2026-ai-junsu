# -*- coding: utf-8 -*-
"""4차 데이터 감사: au 유저구조 / 메타필드 방치신호 / 초단문 / 세션중복 누수."""
import collections
import hashlib
import io
import json
import os
import re
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
OUT = io.open(os.path.join(ART, "analysis_0707d.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


def section(fn):
    def w():
        P("\n" + "=" * 70)
        P("== " + fn.__name__)
        P("=" * 70)
        try:
            fn()
        except Exception:
            P("!! FAILED\n" + traceback.format_exc())
    return w


df = pd.read_parquet(os.path.join(ART, "train_prepared.parquet")).sort_values("id").reset_index(drop=True)
classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
n_cls = len(classes)
raw = {}
for line in io.open(os.path.join(ROOT, "data", "train.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    raw[r["id"]] = r
ids = np.array(df.id.tolist())
y = df.y.values
fold = df.fold.values
au = np.array([i.startswith("sess_au_") for i in ids])

# OOF (5-fold, 현 최고 태그)
o_ids, o_lg, o_y = [], [], []
for f in range(5):
    d = np.load(os.path.join(ART, "oof", f"xlmr_v2_rdrop_lr4_e4_fold{f}.npz"), allow_pickle=True)
    o_ids.append(np.asarray(d["ids"]).astype(str)); o_lg.append(d["logits"]); o_y.append(d["y"])
o_ids = np.concatenate(o_ids)
o_pr = np.concatenate(o_lg).argmax(1)
o_y = np.concatenate(o_y)
oof_pred = dict(zip(o_ids, o_pr))
pred_arr = np.array([oof_pred[i] for i in ids])


def crossfit_majority(keys, mask=None):
    keys = np.asarray(keys)
    m = np.ones(len(keys), bool) if mask is None else mask
    preds = np.full(len(keys), -1)
    for f in range(5):
        tr = (fold != f) & m
        va = (fold == f) & m
        table = {}
        for k, yy in zip(keys[tr], y[tr]):
            table.setdefault(k, collections.Counter())[yy] += 1
        gmaj = collections.Counter(y[tr]).most_common(1)[0][0]
        idx = np.where(va)[0]
        preds[idx] = [table[k].most_common(1)[0][0] if k in table else gmaj for k in keys[idx]]
    mm = m & (preds >= 0)
    return f1_score(y[mm], preds[mm], average="macro"), accuracy_score(y[mm], preds[mm])


@section
def A_au_users():
    pat = re.compile(r"sess_au_(\d+)_(\d+)-step_(\d+)")
    users = {}
    for i in ids[au]:
        m = pat.match(i)
        users.setdefault(m.group(1), set()).add(m.group(2))
    P(f"au 유저(XXXXXX) 수: {len(users)}  | 유저당 세션수 분포: "
      f"{collections.Counter(len(v) for v in users.values()).most_common(8)}")
    uarr = np.array([pat.match(i).group(1) if a else "" for i, a in zip(ids, au)])
    # 유저별 라벨 분포가 유저마다 다른가: 유저 prior cross-fit vs 글로벌 au prior
    mac_u, acc_u = crossfit_majority(uarr, au)
    mac_g, acc_g = crossfit_majority(np.where(au, "AU", "SIM"), au)
    P(f"au rows에서 cross-fit majority — 유저별: macro={mac_u:.4f} acc={acc_u:.4f}")
    P(f"                              글로벌au: macro={mac_g:.4f} acc={acc_g:.4f}")
    P("(유저별 >> 글로벌이면 per-user prior 가치 있음. 단 히든테스트에 같은 유저가 있어야 전이)")
    # 유저 겹침 구조: 유저가 여러 fold에 나뉘나 (per-user prior가 test에 전이될 모델)
    ufold = collections.defaultdict(set)
    for i, f_, a in zip(ids, fold, au):
        if a:
            ufold[pat.match(i).group(1)].add(int(f_))
    multi = sum(1 for v in ufold.values() if len(v) >= 2)
    P(f"여러 fold에 걸친 유저: {multi}/{len(ufold)} (세션이 유저 단위로 안 묶여 있으면 높음)")


@section
def B_meta_signals():
    def bucket(i):
        m = raw[i].get("session_meta", {}) or {}
        ws = m.get("workspace", {}) or {}
        return {
            "tier": str(m.get("user_tier")),
            "ci": str(ws.get("last_ci_status")),
            "dirty": str(ws.get("git_dirty")),
            "budget": str(int(m.get("budget_tokens_remaining", 0)) // 50000),
            "elapsed": str(int(m.get("elapsed_session_sec", 0)) // 600),
        }
    feats = [bucket(i) for i in ids]
    gl = np.bincount(y, minlength=n_cls) / len(y)
    for key in ("tier", "ci", "dirty", "budget", "elapsed"):
        vals = np.array([f[key] for f in feats])
        P(f"\n  [{key}]")
        for v, cnt in collections.Counter(vals).most_common(6):
            m = vals == v
            dist = np.bincount(y[m], minlength=n_cls) / m.sum()
            l1 = float(np.abs(dist - gl).sum())
            acc_model = accuracy_score(y[m], pred_arr[m])
            top = classes[dist.argmax()]
            P(f"    {v:8s} n={m.sum():6d} L1={l1:.3f} top={top:14s} 모델acc={acc_model:.3f}")
        mac, acc = crossfit_majority(vals)
        P(f"    cross-fit majority: macro={mac:.4f} acc={acc:.4f} (라벨 단독예측력)")


@section
def C_short_prompts():
    plen = np.array([len(raw[i].get("current_prompt") or "") for i in ids])
    last1 = np.array([([t.get("name") for t in (raw[i].get("history") or []) if t.get("role") != "user"] or ["<none>"])[-1] for i in ids])
    for cap in (10, 20):
        m = plen <= cap
        if m.sum() < 50:
            P(f"  len<={cap}: n={m.sum()} (너무 적음)")
            continue
        rep = float(np.mean([classes[yy] == l1 for yy, l1 in zip(y[m], last1[m])]))
        acc_model = accuracy_score(y[m], pred_arr[m])
        P(f"  len<={cap}: n={m.sum():5d}  라벨==직전행동 {rep:.1%}  모델acc={acc_model:.3f}")
    # 대표 초단문 예시
    ex = collections.Counter(raw[i].get("current_prompt") for i in ids[plen <= 10]).most_common(8)
    P(f"  초단문 예시: {ex}")


@section
def D_session_dup_leak():
    def sig(sess_rows):
        acts = []
        for i in sess_rows:
            r = raw[i]
            acts.append((r.get("current_prompt") or "")[:80])
        return hashlib.md5("||".join(sorted(acts)).encode("utf-8")).hexdigest()
    by_sess = collections.defaultdict(list)
    for i in ids:
        by_sess[i.rsplit("-", 1)[0]].append(i)
    sig_map = collections.defaultdict(list)
    for s, rows in by_sess.items():
        sig_map[sig(rows)].append(s)
    dups = {k: v for k, v in sig_map.items() if len(v) >= 2}
    P(f"  세션 시그니처 중복 그룹: {len(dups)} (전체 세션 {len(by_sess)})")
    # fold 간 걸친 중복 = CV 누수
    sess_fold = {}
    for i, f in zip(ids, fold):
        sess_fold[i.rsplit("-", 1)[0]] = int(f)
    leak = sum(1 for v in dups.values() if len({sess_fold[s] for s in v}) >= 2)
    P(f"  fold를 넘나드는 중복 세션 그룹: {leak} → 0이면 CV 청정")


A_au_users()
B_meta_signals()
C_short_prompts()
D_session_dup_leak()
P("\nDONE -> artifacts/analysis_0707d.txt")
OUT.close()
