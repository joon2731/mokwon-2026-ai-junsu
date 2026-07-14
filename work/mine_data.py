# -*- coding: utf-8 -*-
"""Weaponized train-data analysis:
(1) duplicate/template mining — do prompts repeat, and are labels consistent?
(2) conditional determinism — does (prompt-dup-group, last_action) pin the label?
(3) verify-cluster cue mining on OOF ERRORS — is there unused surface signal?
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = r"C:\Users\joon2\Desktop\da2"
DATA = os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "artifacts")
sys.path.insert(0, os.path.join(ROOT, "work"))
from tune_bias import load_oof

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def last_action(r):
    for t in reversed(r.get("history") or []):
        if t.get("role") == "assistant_action":
            return t.get("name")
    return "NONE"

recs, lab = {}, {}
with open(os.path.join(DATA, "train.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        recs[r["id"]] = r
labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
lab = dict(zip(labels["id"], labels["action"]))

# ---------- (1) duplicate prompt mining ----------
groups = defaultdict(list)
for rid, r in recs.items():
    groups[norm(r.get("current_prompt"))].append(rid)

sizes = np.array([len(v) for v in groups.values()])
n_dup_groups = int((sizes >= 2).sum())
n_dup_samples = int(sizes[sizes >= 2].sum())
print(f"[1] unique prompts: {len(groups)} / 70000  "
      f"(dup groups: {n_dup_groups}, samples in dups: {n_dup_samples} = {n_dup_samples/700:.1f}%)")

pure = mixed = 0
pure_samples = 0
for p, ids in groups.items():
    if len(ids) >= 2:
        ls = {lab[i] for i in ids}
        if len(ls) == 1:
            pure += 1; pure_samples += len(ids)
        else:
            mixed += 1
print(f"    dup groups with UNANIMOUS label: {pure}/{n_dup_groups} "
      f"({pure_samples} samples) | mixed-label groups: {mixed}")
big = sorted(((len(v), k) for k, v in groups.items()), reverse=True)[:5]
for n, p in big:
    dist = Counter(lab[i] for i in groups[p]).most_common(3)
    print(f"    x{n}  '{p[:60]}'  -> {dist}")

# ---------- (2) conditional determinism: (dup-prompt, last_action) ----------
combo = defaultdict(Counter)
for rid, r in recs.items():
    combo[(norm(r.get("current_prompt")), last_action(r))][lab[rid]] += 1
det = tot = 0
for k, c in combo.items():
    n = sum(c.values())
    if n >= 2:
        tot += n
        det += c.most_common(1)[0][1]
print(f"\n[2] (prompt,last_action) groups n>=2: cover {tot} samples, "
      f"majority-label agreement {det/max(tot,1)*100:.1f}%")

# ---------- (3) verify-cluster cue mining on OOF errors ----------
classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
ids, probs, y = load_oof("xlmr_v2_rdrop")
pred = probs.argmax(1)
cls2i = {c: i for i, c in enumerate(classes)}
VERIFY = ["run_tests", "lint_or_typecheck", "run_bash"]
vidx = [cls2i[c] for c in VERIFY]

CUES = {
    "run_tests": re.compile(r"pytest|jest|unit ?test|테스트|스펙|spec\b|coverage|커버리지|test suite|npm test|go test|돌려|integration", re.I),
    "lint_or_typecheck": re.compile(r"lint|eslint|tsc|mypy|ruff|flake8|prettier|type ?check|타입|정적 ?분석|컴파일 에러|형식", re.I),
    "run_bash": re.compile(r"docker|build|빌드|서버|실행|띄워|npm run|make\b|uvicorn|커맨드|명령|스크립트|deploy|migrate|셸|shell", re.I),
}

err = [(i, classes[y[i]], classes[pred[i]]) for i in range(len(y))
       if y[i] != pred[i] and y[i] in vidx and pred[i] in vidx]
print(f"\n[3] verify-cluster internal errors in OOF: {len(err)}")
cue_hit = 0
rescueable = 0
for i, true_c, pred_c in err:
    prompt = recs[ids[i]].get("current_prompt", "")
    has_true_cue = bool(CUES[true_c].search(prompt))
    has_pred_cue = bool(CUES[pred_c].search(prompt))
    if has_true_cue:
        cue_hit += 1
        if not has_pred_cue:
            rescueable += 1
print(f"    errors where TRUE-class cue exists in prompt: {cue_hit} ({cue_hit/max(len(err),1)*100:.0f}%)")
print(f"    ... and PRED-class cue absent (clean rescue): {rescueable} ({rescueable/max(len(err),1)*100:.0f}%)")
# what would perfect rescue be worth?
from sklearn.metrics import f1_score
pred2 = pred.copy()
n_flip = 0
for i, true_c, pred_c in err:
    prompt = recs[ids[i]].get("current_prompt", "")
    if CUES[true_c].search(prompt) and not CUES[pred_c].search(prompt):
        pred2[i] = y[i]; n_flip += 1
m0 = f1_score(y, pred, average="macro")
m1 = f1_score(y, pred2, average="macro")
print(f"    upper-bound if all {n_flip} clean-rescues fixed: {m0:.4f} -> {m1:.4f} (+{m1-m0:.4f})")
print("    (upper bound only — a real rule/model must also not break correct preds)")
