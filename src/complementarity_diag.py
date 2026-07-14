# -*- coding: utf-8 -*-
"""오차 상보성 진단: Qwen3/XLM-R/mmBERT OOF가 어느 축에서 상보적인가.

계획서 지목 "미해결 최유력 지점" — oracle 0.8028을 만드는 상보성의 소재를
클래스/혼동그룹/au 세그먼트별로 분해. 태그류 4회 기각을 감안, 진단만 수행.
GPU 학습 중 CPU 경량 분석 (OOF npz 로드 + numpy).
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

OOF = r"C:\Users\joon2\Desktop\da2\artifacts\oof"
ART = r"C:\Users\joon2\Desktop\da2\artifacts"
classes = json.load(open(f"{ART}/classes.json", encoding="utf-8"))
GROUP = {
    "glob_pattern": "nav", "grep_search": "nav", "list_directory": "nav", "read_file": "nav",
    "lint_or_typecheck": "verify", "run_bash": "verify", "run_tests": "verify",
    "ask_user": "dialogue", "plan_task": "dialogue", "respond_only": "dialogue",
    "apply_patch": "modify", "edit_file": "modify", "write_file": "modify",
    "web_search": "web",
}


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def load5(tag):
    d = {}
    for f in range(5):
        z = np.load(f"{OOF}/{tag}_fold{f}.npz", allow_pickle=True)
        for i, lg, yy in zip(z["ids"], z["logits"], z["y"]):
            d[i] = (lg, int(yy))
    return d


print("loading OOF (qwen3_smoke / xlmr_v2_rdrop / mmbert_v2)...", flush=True)
Q, X, M = load5("qwen3_smoke"), load5("xlmr_v2_rdrop"), load5("mmbert_v2")
ids = [i for i in Q if i in X and i in M]
print(f"common ids: {len(ids)}", flush=True)

y = np.array([Q[i][1] for i in ids])
qp = softmax(np.stack([Q[i][0] for i in ids]))
xp = softmax(np.stack([X[i][0] for i in ids]))
mp = softmax(np.stack([M[i][0] for i in ids]))

q_pred, x_pred, m_pred = qp.argmax(1), xp.argmax(1), mp.argmax(1)
blend = (0.40 * qp + 0.27 * xp + 0.33 * mp).argmax(1)

macro = lambda p: f1_score(y, p, average="macro")
acc = lambda p: (p == y).mean()
print(f"\n== 단일/블렌드 macro-F1 ==")
print(f"  Qwen3   macro={macro(q_pred):.4f} acc={acc(q_pred):.4f}")
print(f"  XLM-R   macro={macro(x_pred):.4f} acc={acc(x_pred):.4f}")
print(f"  mmBERT  macro={macro(m_pred):.4f} acc={acc(m_pred):.4f}")
print(f"  blend(.40/.27/.33) macro={macro(blend):.4f} acc={acc(blend):.4f}")

# oracle: any of the three correct
any_correct = (q_pred == y) | (x_pred == y) | (m_pred == y)
print(f"\n== oracle (셋 중 하나라도 정답) ==")
print(f"  either-correct 비율: {any_correct.mean():.4f}")

# 상보성: Qwen(주엔진) 오답인데 다른 모델이 맞는 샘플
q_wrong = q_pred != y
rescued = q_wrong & ((x_pred == y) | (m_pred == y))
print(f"  Qwen 오답: {q_wrong.sum()} ({q_wrong.mean():.4f})")
print(f"  그중 X/M이 구제 가능: {rescued.sum()} ({rescued.sum()/max(q_wrong.sum(),1):.4f} of Qwen-wrong)")
print(f"    - XLM-R 구제: {(q_wrong & (x_pred==y)).sum()}  mmBERT 구제: {(q_wrong & (m_pred==y)).sum()}")

# 축별 상보성: 구제 가능 샘플이 어느 정답 클래스/그룹/세그먼트에 몰리나
grp = np.array([GROUP[classes[t]] for t in y])
is_au = np.array([str(i).startswith("sess_au_") for i in ids])

print(f"\n== 정답 클래스별: Qwen recall / blend recall / 구제여지(rescued 비율) ==")
rows = []
for c in range(14):
    m = y == c
    n = m.sum()
    if n == 0:
        continue
    q_rec = (q_pred[m] == c).mean()
    b_rec = (blend[m] == c).mean()
    resc = rescued[m].mean()
    rows.append((classes[c], n, q_rec, b_rec, b_rec - q_rec, resc))
rows.sort(key=lambda r: -r[5])
print(f"  {'class':18s} {'n':>5s} {'Qwen':>6s} {'blend':>6s} {'Δ':>6s} {'rescue':>7s}")
for name, n, qr, br, d, rs in rows:
    print(f"  {name:18s} {n:5d} {qr:6.3f} {br:6.3f} {d:+6.3f} {rs:7.3f}")

print(f"\n== 혼동그룹별 ==")
for g in ["nav", "verify", "dialogue", "modify", "web"]:
    m = grp == g
    n = m.sum()
    if n == 0:
        continue
    print(f"  {g:9s} n={n:5d}  Qwen_acc={acc(q_pred[m] == y[m]) if False else (q_pred[m]==y[m]).mean():.3f}  "
          f"blend_acc={(blend[m]==y[m]).mean():.3f}  rescue={rescued[m].mean():.3f}")

print(f"\n== au 세그먼트별 ==")
for seg, m in [("au", is_au), ("sim", ~is_au)]:
    n = m.sum()
    print(f"  {seg:4s} n={n:5d}  Qwen_acc={(q_pred[m]==y[m]).mean():.3f}  "
          f"blend_acc={(blend[m]==y[m]).mean():.3f}  rescue={rescued[m].mean():.3f}")

# group-confined vs cross-group 오답: Qwen 오답이 같은 그룹 내 혼동인가 그룹 밖인가
q_pred_grp = np.array([GROUP[classes[t]] for t in q_pred])
within = q_wrong & (q_pred_grp == grp)
cross = q_wrong & (q_pred_grp != grp)
step_no = np.array([int(str(i).rsplit("-step_", 1)[1]) if "-step_" in str(i) else 0 for i in ids])
nav_m = grp == "nav"
print(f"\n== nav 오답의 step(턴 위상) 분포 — 정보한계 vs 경계모호 판별 ==")
print(f"  {'step':8s} {'n':>6s} {'Qwen_acc':>9s} {'blend_acc':>10s} {'rescue':>7s}")
for lo, hi, lbl in [(1, 1, "step01"), (2, 2, "step02"), (3, 5, "step3-5"), (6, 99, "step6+")]:
    sm = nav_m & (step_no >= lo) & (step_no <= hi)
    n = sm.sum()
    if n:
        print(f"  {lbl:8s} {n:6d} {(q_pred[sm]==y[sm]).mean():9.3f} "
              f"{(blend[sm]==y[sm]).mean():10.3f} {rescued[sm].mean():7.3f}")

print(f"\n== Qwen 오답의 그룹내/그룹밖 분해 (group CE 타당성) ==")
print(f"  그룹내 혼동(같은 그룹 오답): {within.sum()} ({within.sum()/max(q_wrong.sum(),1):.3f} of wrong)")
print(f"  그룹밖 혼동(다른 그룹 오답): {cross.sum()} ({cross.sum()/max(q_wrong.sum(),1):.3f} of wrong)")
print(f"  → group CE는 그룹내 혼동만 겨냥. 그룹내 비율이 낮으면 상한이 낮음")
