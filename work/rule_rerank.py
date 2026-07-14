# -*- coding: utf-8 -*-
"""Margin-gated cue rerank for the verify cluster, honestly evaluated on OOF.

Rule: if model's top-1 is a verify-cluster class, the prompt contains EXACTLY
ONE verify-class cue, that cue class differs from top-1, and the probability
margin (p_top1 - p_cue) < theta  ->  flip prediction to the cue class.
theta is tuned with a session-disjoint split-half (tune on A, eval on B, and
vice versa) so the reported gain is honest.
"""
import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import f1_score

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
sys.path.insert(0, os.path.join(ROOT, "work"))
from tune_bias import load_oof, _session_of

CUES = {
    "run_tests": re.compile(r"pytest|jest|unit ?test|테스트|스펙|spec\b|coverage|커버리지|test suite|npm test|go test|integration", re.I),
    "lint_or_typecheck": re.compile(r"lint|eslint|tsc|mypy|ruff|flake8|prettier|type ?check|타입|정적 ?분석|컴파일 에러|형식", re.I),
    "run_bash": re.compile(r"docker|빌드|서버|띄워|npm run|make\b|uvicorn|커맨드|명령|스크립트|deploy|migrate|셸|shell|실행", re.I),
}
VERIFY = list(CUES.keys())


def main():
    classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
    cls2i = {c: i for i, c in enumerate(classes)}
    vset = {cls2i[c] for c in VERIFY}

    ids, probs, y = load_oof("xlmr_v2_rdrop")

    # prompts for OOF ids
    prompts = {}
    with open(os.path.join(ROOT, "data", "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prompts[r["id"]] = r.get("current_prompt", "") or ""

    # precompute: cue class index (or -1 if 0 or >=2 hits)
    cue_cls = np.full(len(ids), -1, dtype=np.int64)
    for k, rid in enumerate(ids):
        hits = [c for c in VERIFY if CUES[c].search(prompts[rid])]
        if len(hits) == 1:
            cue_cls[k] = cls2i[hits[0]]

    top1 = probs.argmax(1)
    p_top1 = probs[np.arange(len(y)), top1]
    p_cue = np.where(cue_cls >= 0, probs[np.arange(len(y)), np.maximum(cue_cls, 0)], 0.0)
    margin = p_top1 - p_cue

    eligible = (cue_cls >= 0) & np.isin(top1, list(vset)) & (top1 != cue_cls)
    print(f"OOF n={len(y)} | eligible for rule: {eligible.sum()} "
          f"({eligible.sum()/len(y)*100:.1f}%)")

    def apply_rule(theta, mask=None):
        pred = top1.copy()
        m = eligible & (margin < theta)
        if mask is not None:
            m &= mask
        pred[m] = cue_cls[m]
        return pred

    base = f1_score(y, top1, average="macro")
    print(f"baseline macro = {base:.4f}\n")

    thetas = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
    print("in-sample sweep (theta -> macro, n flipped):")
    for t in thetas:
        pred = apply_rule(t)
        n_flip = int((pred != top1).sum())
        print(f"  theta={t:4.2f} -> {f1_score(y, pred, average='macro'):.4f}  (flips {n_flip})")

    # honest split-half by session
    sess = np.array([_session_of(i) for i in ids])
    uniq = np.unique(sess)
    rng = np.random.RandomState(0)
    half = set(rng.permutation(uniq)[: len(uniq) // 2].tolist())
    A = np.array([s in half for s in sess])
    gains = []
    for tr, te in ((A, ~A), (~A, A)):
        best_t, best_m = 0, -1
        for t in thetas:
            m = f1_score(y[tr], apply_rule(t)[tr], average="macro")
            if m > best_m:
                best_m, best_t = m, t
        m_base = f1_score(y[te], top1[te], average="macro")
        m_rule = f1_score(y[te], apply_rule(best_t)[te], average="macro")
        gains.append(m_rule - m_base)
        print(f"\nhalf: theta*={best_t:.2f}  held-out {m_base:.4f} -> {m_rule:.4f} "
              f"({m_rule-m_base:+.4f})")
    print(f"\nHONEST mean gain: {np.mean(gains):+.4f}")
    print("VERDICT:", "ADOPT" if np.mean(gains) > 0.002 else "reject/weak")


if __name__ == "__main__":
    main()
