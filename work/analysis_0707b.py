# -*- coding: utf-8 -*-
"""Follow-up: (1) label purity by dup-group size — THE noise measurement,
(2) id regex failure cause, (3) purity on full serialized text."""
import collections
import io
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
OUT = io.open(os.path.join(ART, "analysis_0707b.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


df = pd.read_parquet(os.path.join(ART, "train_prepared.parquet"))
classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
raw = {}
for line in io.open(os.path.join(ROOT, "data", "train.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    raw[r["id"]] = r
df = df.sort_values("id").reset_index(drop=True)
ids = df.id.tolist()
y = df.y.values

# ---- (2) which ids break the sess_sim regex ----
pat = re.compile(r"sess_sim_(\d{8})_(\d+)-step_(\d+)")
bad = [i for i in ids if not pat.match(i)]
P(f"ids not matching 'sess_sim_YYYYMMDD_NUM-step_NN': {len(bad)}")
for b in bad[:10]:
    P("   ", b)
# retry with a general pattern
pat2 = re.compile(r"^(.*)_(\d+)-step_(\d+)$")
ok2 = sum(1 for i in ids if pat2.match(i))
P(f"general '<prefix>_NUM-step_NN' matches: {ok2}/70000")
pre = collections.Counter(pat2.match(i).group(1) for i in ids if pat2.match(i))
P("prefixes:", dict(pre.most_common(6)))

# session number vs label distribution (using general pattern)
snum = np.array([int(pat2.match(i).group(2)) for i in ids])
n_cls = len(classes)
gl = np.bincount(y, minlength=n_cls) / len(y)
dec = np.digitize(snum, np.percentile(snum, np.arange(10, 100, 10)))
worst = max(float(np.abs(np.bincount(y[dec == d], minlength=n_cls) / (dec == d).sum() - gl).sum())
            for d in range(10))
P(f"label-dist L1 dev across session-number deciles: max={worst:.4f} (<0.02 = 신호 없음)")

# test stub session overlap
tr_sess = {i.rsplit("-", 1)[0] for i in ids}
te = [json.loads(l) for l in io.open(os.path.join(ROOT, "data", "test.jsonl"), encoding="utf-8")]
hits = [t["id"] for t in te if t["id"].rsplit("-", 1)[0] in tr_sess]
P(f"local test stub: {len(te)} rows, train과 세션 겹침 = {len(hits)}")
for h in hits:
    P("    overlap:", h)

# ---- (1) purity by dup-group size: skeleton ----
def skel(s):
    s = (s or "").lower()
    s = re.sub(r"[0-9]+", "#", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

SK = [skel(raw[i].get("current_prompt", "")) for i in ids]

def purity_curve(keys, name):
    table = {}
    for k, yy in zip(keys, y):
        table.setdefault(k, collections.Counter())[yy] += 1
    P(f"\n== purity by group size: {name} ==")
    bins = [(2, 2), (3, 4), (5, 9), (10, 19), (20, 10 ** 9)]
    tot_rows = pure_rows = 0
    for lo, hi in bins:
        g = [c for c in table.values() if lo <= sum(c.values()) <= hi]
        if not g:
            P(f"  n={lo}-{hi}: (없음)")
            continue
        rows = sum(sum(c.values()) for c in g)
        wpur = sum(c.most_common(1)[0][1] for c in g) / rows
        perfect = sum(1 for c in g if len(c) == 1) / len(g)
        P(f"  n={lo:2d}-{hi if hi < 10**9 else '+'}: groups={len(g):5d} rows={rows:6d} "
          f"가중순도={wpur:.4f}  완전순수그룹비율={perfect:.3f}")
        tot_rows += rows
        pure_rows += sum(c.most_common(1)[0][1] for c in g)
    if tot_rows:
        P(f"  전체 dup rows={tot_rows}  가중순도={pure_rows/tot_rows:.4f}")
    singles = sum(1 for c in table.values() if sum(c.values()) == 1)
    P(f"  singleton groups: {singles} ({singles/len(table):.1%} of groups)")

purity_curve(SK, "prompt skeleton")
purity_curve(df.text.tolist(), "full serialized text (V2)")

# label composition of impure skeleton dup groups: which classes collide?
table = {}
for k, yy in zip(SK, y):
    table.setdefault(k, collections.Counter())[yy] += 1
coll = collections.Counter()
for c in table.values():
    if sum(c.values()) >= 2 and len(c) >= 2:
        top2 = [cc for cc, _ in c.most_common(2)]
        coll[(classes[top2[0]], classes[top2[1]])] += 1
P("\n== 불순 dup 그룹의 클래스 충돌 top12 ==")
for (a, b), n in coll.most_common(12):
    P(f"  {a:18s} vs {b:18s} {n}")
OUT.close()
P("\nDONE -> artifacts/analysis_0707b.txt")
