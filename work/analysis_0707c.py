# -*- coding: utf-8 -*-
"""Follow-up 2: (1) sess_au vs sess_sim 분포·OOF 성능, (2) sim 내부 세션번호 신호,
(3) train-side overlay 실효성 (스텁 세션으로 메커니즘 확인)."""
import collections
import io
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
OUT = io.open(os.path.join(ART, "analysis_0707c.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")


df = pd.read_parquet(os.path.join(ART, "train_prepared.parquet"))
classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
n_cls = len(classes)
raw = {}
for line in io.open(os.path.join(ROOT, "data", "train.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    raw[r["id"]] = r
df = df.sort_values("id").reset_index(drop=True)
ids = np.array(df.id.tolist())
y = df.y.values

# ---- (1) au vs sim ----
is_au = np.array([i.startswith("sess_au_") for i in ids])
P(f"sess_au rows: {is_au.sum()} ({is_au.mean():.1%})  sim rows: {(~is_au).sum()}")
au_sess = {i.rsplit("-", 1)[0] for i in ids[is_au]}
sim_sess = {i.rsplit("-", 1)[0] for i in ids[~is_au]}
P(f"au sessions: {len(au_sess)}  sim sessions: {len(sim_sess)}")
P(f"au rows/session: {is_au.sum()/len(au_sess):.2f}  sim: {(~is_au).sum()/len(sim_sess):.2f}")

gl_sim = np.bincount(y[~is_au], minlength=n_cls) / (~is_au).sum()
gl_au = np.bincount(y[is_au], minlength=n_cls) / is_au.sum()
P(f"label dist L1(au, sim) = {np.abs(gl_au - gl_sim).sum():.4f}  (0.02 미만=동일 분포)")
P("  class          sim    au")
for i, c in enumerate(classes):
    mark = " <-- diff" if abs(gl_au[i] - gl_sim[i]) > 0.02 else ""
    P(f"  {c:18s} {gl_sim[i]:.3f}  {gl_au[i]:.3f}{mark}")

# language / turns
lang = collections.Counter(raw[i]["session_meta"].get("language_pref", "?") for i in ids[is_au])
P(f"au language_pref: {dict(lang)}")
ht_au = np.mean([len(raw[i].get("history") or []) // 2 for i in ids[is_au]])
ht_sim = np.mean([len(raw[i].get("history") or []) // 2 for i in ids[~is_au]])
P(f"avg history turns: au={ht_au:.2f} sim={ht_sim:.2f}")

# OOF fold0 performance by source
d = np.load(os.path.join(ART, "oof", "xlmr_v2_rdrop_lr4_e4_fold0.npz"), allow_pickle=True)
oids = np.asarray(d["ids"]).astype(str)
pr, yy = d["logits"].argmax(1), d["y"]
au_m = np.array([i.startswith("sess_au_") for i in oids])
for name, m in (("sim", ~au_m), ("au", au_m)):
    if m.sum():
        P(f"OOF fold0 {name}: n={m.sum():5d} macro={f1_score(yy[m], pr[m], average='macro'):.4f} "
          f"acc={accuracy_score(yy[m], pr[m]):.4f}")

# ---- (2) session-number deciles within SIM only ----
pat = re.compile(r"sess_sim_\d{8}_(\d+)-step_(\d+)")
snum = np.array([int(pat.match(i).group(1)) for i in ids[~is_au]])
ysim = y[~is_au]
dec = np.digitize(snum, np.percentile(snum, np.arange(10, 100, 10)))
gl = np.bincount(ysim, minlength=n_cls) / len(ysim)
worst = max(float(np.abs(np.bincount(ysim[dec == q], minlength=n_cls) / (dec == q).sum() - gl).sum())
            for q in range(10))
P(f"\n[sim only] label-dist L1 dev across session-number deciles: max={worst:.4f} (<0.02=신호없음)")

# ---- (3) train-side overlay mechanics on the local test stub ----
P("\n== train-side overlay (스텁 5행으로 메커니즘 확인) ==")
by_sess = {}
for i in ids:
    s, st = i.rsplit("-", 1)
    by_sess.setdefault(s, {})[int(st.split("_")[1])] = i
te = [json.loads(l) for l in io.open(os.path.join(ROOT, "data", "test.jsonl"), encoding="utf-8")]
hit = 0
for t in te:
    s, st = t["id"].rsplit("-", 1)
    K = int(st.split("_")[1])
    steps = sorted(by_sess.get(s, {}))
    win = [J for J in steps if K + 1 <= J <= K + 6]
    P(f"  {t['id']}: train에 같은 세션 steps={steps}")
    if win:
        J = min(win)
        jrow = raw[by_sess[s][J]]
        acts = [x for x in jrow.get("history", []) if x.get("role") != "user"]
        # step J's history covers steps J-len..J-1; step K = index -(J-K)
        offset = J - K
        if len(acts) >= offset:
            act = acts[-offset].get("name")
            P(f"    -> step_{J} history에서 step_{K} 행동 복원: **{act}**")
            hit += 1
    else:
        P("    -> 복원 불가 (K+1..K+6 스텝 없음)")
P(f"\n스텁 {len(te)}행 중 train-side 복원 가능 = {hit}")
P("주의: 스텁이 train에서 샘플된 예시일 가능성 있음. 히든 테스트 적용률은 제출 프로브로만 확정 가능.")
OUT.close()
print("DONE -> artifacts/analysis_0707c.txt", flush=True)
