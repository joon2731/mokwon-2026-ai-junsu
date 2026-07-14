# -*- coding: utf-8 -*-
"""From-scratch signal audit (report-only, CPU-only, no training).

Writes artifacts/analysis_0707.txt. Sections are independent; a failure in one
does not kill the rest.
"""
import collections
import io
import json
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
OUT = io.open(os.path.join(ART, "analysis_0707.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


def section(fn):
    def wrap(*a, **k):
        P("\n" + "=" * 72)
        P(f"== {fn.__name__}")
        P("=" * 72)
        try:
            fn(*a, **k)
        except Exception:
            P("!! SECTION FAILED")
            P(traceback.format_exc())
    return wrap


# ---------- shared data ----------
df = pd.read_parquet(os.path.join(ART, "train_prepared.parquet"))
classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
n_cls = len(classes)
raw = {}
for line in io.open(os.path.join(ROOT, "data", "train.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    raw[r["id"]] = r
df = df.sort_values("id").reset_index(drop=True)
ids = df.id.tolist()
y = df.y.values
fold = df.fold.values
P(f"rows={len(df)}  classes={n_cls}")


def last_actions(r, n=2):
    acts = [t for t in (r.get("history") or []) if t.get("role") != "user"]
    names = [t.get("name", "?") for t in acts[-n:]]
    return (["<none>"] * (n - len(names)) + names)


LAST1 = np.array([last_actions(raw[i], 1)[0] for i in ids])
LAST2 = np.array(["|".join(last_actions(raw[i], 2)) for i in ids])


def crossfit_majority(keys, fallback_global=True):
    """Honest memorization score: per-key majority learned on other folds."""
    keys = np.asarray(keys)
    preds = np.full(len(keys), -1)
    for f in range(5):
        tr, va = fold != f, fold == f
        table = {}
        for k, yy in zip(keys[tr], y[tr]):
            table.setdefault(k, collections.Counter())[yy] += 1
        gmaj = collections.Counter(y[tr]).most_common(1)[0][0]
        kv = keys[va]
        preds[va] = [table[k].most_common(1)[0][0] if k in table else gmaj for k in kv]
    cov = float(np.mean([1.0]))  # coverage computed separately below
    return preds


def key_coverage(keys):
    keys = np.asarray(keys)
    cov = np.zeros(len(keys), bool)
    for f in range(5):
        tr, va = fold != f, fold == f
        seen = set(keys[tr])
        cov[va] = [k in seen for k in keys[va]]
    return cov.mean()


def insample_majority(keys):
    keys = np.asarray(keys)
    table = {}
    for k, yy in zip(keys, y):
        table.setdefault(k, collections.Counter())[yy] += 1
    return np.array([table[k].most_common(1)[0][0] for k in keys])


def report_pred(name, preds, extra=""):
    P(f"  {name:44s} macro={f1_score(y, preds, average='macro'):.4f} "
      f"acc={accuracy_score(y, preds):.4f} {extra}")


# ---------- A. action-history Markov ----------
@section
def A_markov():
    P("(cross-fit majority per context; fallback=global majority)")
    report_pred("last1 action", crossfit_majority(LAST1),
                f"cov={key_coverage(LAST1):.3f}")
    report_pred("last2 actions", crossfit_majority(LAST2),
                f"cov={key_coverage(LAST2):.3f}")
    # transition matrix: top rows
    P("\n  P(y | last1) top transitions (n>=2000):")
    for ctx, cnt in collections.Counter(LAST1).most_common(16):
        sub = y[LAST1 == ctx]
        top = collections.Counter(sub).most_common(3)
        s = "  ".join(f"{classes[c]}:{n/len(sub):.2f}" for c, n in top)
        P(f"    last={ctx:18s} n={len(sub):5d}  {s}")


# ---------- A2. last-result surface features ----------
@section
def A2_result_features():
    def feat(r):
        acts = [t for t in (r.get("history") or []) if t.get("role") != "user"]
        if not acts:
            return "<none>", "<none>"
        t = acts[-1]
        rs = str(t.get("result_summary", "") or "")
        st = "other"
        m = re.match(r"^(\d+) matches", rs)
        if rs.startswith("PASS"):
            st = "PASS"
        elif rs.startswith("FAIL"):
            st = "FAIL"
        elif rs.startswith("ok"):
            st = "ok"
        elif rs.startswith("error") or "error" in rs[:30].lower():
            st = "error"
        if m:
            st = "0match" if int(m.group(1)) == 0 else "match+"
        return t.get("name", "?"), st

    pairs = [feat(raw[i]) for i in ids]
    KEY = np.array([f"{a}|{s}" for a, s in pairs])
    report_pred("last1+result-status", crossfit_majority(KEY),
                f"cov={key_coverage(KEY):.3f}")
    P("\n  interesting conditionals (same action, different result):")
    for act in ("grep_search", "run_tests", "run_bash", "glob_pattern", "lint_or_typecheck"):
        stats = {}
        for (a, s), yy in zip(pairs, y):
            if a == act:
                stats.setdefault(s, collections.Counter())[yy] += 1
        for s, cnt in sorted(stats.items(), key=lambda kv: -sum(kv[1].values())):
            tot = sum(cnt.values())
            if tot < 300:
                continue
            top = "  ".join(f"{classes[c]}:{n/tot:.2f}" for c, n in cnt.most_common(3))
            P(f"    last={act:18s} res={s:7s} n={tot:5d}  {top}")


# ---------- B. prompt templates ----------
@section
def B_templates():
    def skel(s):
        s = (s or "").lower()
        s = re.sub(r"[0-9]+", "#", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    SK = np.array([skel(raw[i].get("current_prompt", "")) for i in ids])
    uniq = len(set(SK))
    P(f"  prompt skeletons: {uniq} unique / 70000  (dup rate {1-uniq/70000:.1%})")
    report_pred("skeleton (cross-fit)", crossfit_majority(SK),
                f"cov={key_coverage(SK):.3f}")
    report_pred("skeleton (IN-SAMPLE ceiling proxy)", insample_majority(SK))
    K2 = np.array([f"{a}||{b}" for a, b in zip(SK, LAST1)])
    report_pred("skeleton+last1 (cross-fit)", crossfit_majority(K2),
                f"cov={key_coverage(K2):.3f}")
    report_pred("skeleton+last1 (IN-SAMPLE)", insample_majority(K2))

    # purity of big groups = direct noise measurement
    table = {}
    for k, yy in zip(SK, y):
        table.setdefault(k, collections.Counter())[yy] += 1
    big = [(k, c) for k, c in table.items() if sum(c.values()) >= 20]
    pur = [c.most_common(1)[0][1] / sum(c.values()) for _, c in big]
    P(f"\n  groups(n>=20): {len(big)}  covering {sum(sum(c.values()) for _, c in big)} rows")
    P(f"  purity: mean={np.mean(pur):.3f}  p10={np.percentile(pur,10):.3f} "
      f"p50={np.percentile(pur,50):.3f} p90={np.percentile(pur,90):.3f}")
    P("  (purity ~1.0 = 규칙 결정적, ~0.5 = 라벨 확률적 → 천장)")
    amb = sorted(big, key=lambda kc: kc[1].most_common(1)[0][1] / sum(kc[1].values()))[:5]
    for k, c in amb:
        tot = sum(c.values())
        top = "  ".join(f"{classes[cc]}:{n/tot:.2f}" for cc, n in c.most_common(3))
        P(f"    ambiguous n={tot:4d}: '{k[:70]}' -> {top}")


# ---------- C. id structure ----------
@section
def C_ids():
    pat = re.compile(r"sess_sim_(\d{8})_(\d+)-step_(\d+)")
    dates, snum, steps = [], [], []
    for i in ids:
        m = pat.match(i)
        dates.append(m.group(1)); snum.append(int(m.group(2))); steps.append(int(m.group(3)))
    P(f"  dates: {sorted(set(dates))}")
    snum = np.array(snum)
    P(f"  session number: min={snum.min()} max={snum.max()} unique={len(set(snum.tolist()))}")
    # label distribution vs session-number decile
    dec = np.digitize(snum, np.percentile(snum, np.arange(10, 100, 10)))
    gl = np.bincount(y, minlength=n_cls) / len(y)
    worst = 0.0
    for d in range(10):
        sub = y[dec == d]
        dist = np.bincount(sub, minlength=n_cls) / len(sub)
        worst = max(worst, float(np.abs(dist - gl).sum()))
    P(f"  label-dist L1 deviation across deciles: max={worst:.4f} "
      f"(<0.02 = 세션번호에 신호 없음)")
    # test stub overlap
    tr_sess = {i.rsplit("-", 1)[0] for i in ids}
    te = [json.loads(l) for l in io.open(os.path.join(ROOT, "data", "test.jsonl"), encoding="utf-8")]
    hits = [t["id"] for t in te if t["id"].rsplit("-", 1)[0] in tr_sess]
    P(f"  local test stub: {len(te)} rows, session-overlap with train = {len(hits)}")
    if hits:
        P(f"    !! OVERLAP: {hits}  -> train에서 같은 세션 후속 step의 history로 정답 조회 가능성. 즉시 확인 필요")


# ---------- D. TF-IDF kNN (fold-0 honest) ----------
@section
def D_knn():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    tr, va = fold != 0, fold == 0
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                          max_features=150000, min_df=2, dtype=np.float32)
    Xtr = vec.fit_transform(df.text[tr])
    Xva = vec.transform(df.text[va])
    P(f"  tfidf: train {Xtr.shape}, val {Xva.shape}")
    nn = NearestNeighbors(n_neighbors=25, metric="cosine", algorithm="brute", n_jobs=2)
    nn.fit(Xtr)
    ytr = y[tr]
    preds = np.zeros(int(va.sum()), dtype=int)
    B = 2000
    for s in range(0, int(va.sum()), B):
        dist, idx = nn.kneighbors(Xva[s:s + B])
        w = 1.0 - dist
        for j in range(idx.shape[0]):
            votes = np.zeros(n_cls)
            for k_, i_ in enumerate(idx[j]):
                votes[ytr[i_]] += w[j, k_]
            preds[s + j] = votes.argmax()
    yv = y[va]
    P(f"  kNN(k=25, char 3-4) fold-0: macro={f1_score(yv, preds, average='macro'):.4f} "
      f"acc={accuracy_score(yv, preds):.4f}   (모델 0.7278 대비)")
    # exact duplicates
    seen = {}
    for t, yy in zip(df.text[tr], ytr):
        seen.setdefault(t, collections.Counter())[yy] += 1
    hit = agree = 0
    for t, yy in zip(df.text[va], yv):
        if t in seen:
            hit += 1
            agree += int(seen[t].most_common(1)[0][0] == yy)
    P(f"  exact-text dup: {hit}/{int(va.sum())} val rows ({hit/va.sum():.1%}), "
      f"label-agree {agree}/{hit if hit else 1} = {agree/max(hit,1):.1%}")


# ---------- E/F. current-model OOF dissection ----------
@section
def EF_oof():
    p = os.path.join(ART, "oof", "xlmr_v2_rdrop_lr4_e4_fold0.npz")
    d = np.load(p, allow_pickle=True)
    oids = np.asarray(d["ids"]).astype(str)
    lg, yy = d["logits"], d["y"]
    pr = lg.argmax(1)
    P(f"  fold0 OOF n={len(yy)}  macro={f1_score(yy, pr, average='macro'):.4f}")
    meta = {i: raw[i] for i in oids}
    lang = np.array([meta[i].get("session_meta", {}).get("language_pref", "?") for i in oids])
    turn = np.array([meta[i].get("session_meta", {}).get("turn_index", -1) for i in oids])
    l1 = np.array([last_actions(meta[i], 1)[0] for i in oids])
    P("\n  by language_pref:")
    for g in sorted(set(lang.tolist())):
        m = lang == g
        P(f"    {g:6s} n={m.sum():5d} macro={f1_score(yy[m], pr[m], average='macro'):.4f} "
          f"acc={accuracy_score(yy[m], pr[m]):.4f}")
    P("\n  by turn bucket:")
    for lo, hi in ((0, 1), (2, 3), (4, 5), (6, 99)):
        m = (turn >= lo) & (turn <= hi)
        if m.sum():
            P(f"    turn {lo}-{hi}: n={m.sum():5d} macro={f1_score(yy[m], pr[m], average='macro'):.4f} "
              f"acc={accuracy_score(yy[m], pr[m]):.4f}")
    P("\n  top confusions (true -> pred, count):")
    cm = collections.Counter((int(t), int(q)) for t, q in zip(yy, pr) if t != q)
    for (t, q), c in cm.most_common(15):
        P(f"    {classes[t]:18s} -> {classes[q]:18s} {c:4d}")
    P("\n  per-class F1:")
    pc = f1_score(yy, pr, average=None, labels=list(range(n_cls)))
    for i, c in enumerate(classes):
        P(f"    {c:18s} {pc[i]:.3f}")


A_markov()
A2_result_features()
B_templates()
C_ids()
D_knn()
EF_oof()
P("\nDONE -> artifacts/analysis_0707.txt")
OUT.close()
