"""result_summary 상태 신호의 헤드룸 정량화 (GPU 게이트 전 CPU 판정).

직전 액션의 result_summary에서 상태(OK/ERR/FAIL/ZERO/EMPTY)·개수를 추출하고,
상태별로 (a) Qwen OOF 정확도, (b) 상태→정답 분포 집중도를 측정.
"틀림이 몰려있고 + 상태가 정답을 가르는" 구간이 있어야 재직렬화 가치가 있다.
"""
import json
import re

import numpy as np
import pandas as pd

ART = r"C:\Users\joon2\Desktop\da2\artifacts"
classes = json.load(open(ART + r"\classes.json", encoding="utf-8"))
C = len(classes)

samples = [json.loads(l) for l in open(r"C:\Users\joon2\Desktop\da2\data\train.jsonl", encoding="utf-8") if l.strip()]
by_id = {s["id"]: s for s in samples}

ids, logits, y = [], [], []
for f in range(5):
    z = np.load(ART + rf"\oof\qwen3_smoke_fold{f}.npz", allow_pickle=True)
    ids.extend(z["ids"].tolist()); logits.append(z["logits"]); y.append(z["y"])
ids = np.array(ids); pred = np.vstack(logits).argmax(1); y = np.concatenate(y)


def rs_state(s):
    """직전 액션의 (상태, 개수버킷) 추출. 히스토리 없으면 ('NONE', '')."""
    h = by_id[s].get("history", [])
    last = None
    for e in reversed(h):
        if e.get("role") == "assistant_action":
            last = e
            break
    if last is None:
        return "NOHIST", ""
    rs = (last.get("result_summary") or "").strip()
    name = last.get("name", "")
    if rs.startswith("ERROR") or rs.startswith("FAIL") or "command failed" in rs:
        st = "ERR"
    elif rs.startswith("0 matches") or rs.startswith("no relevant results") or "0 entries" in rs:
        st = "ZERO"
    elif rs.startswith("ok") or rs.startswith("PASS") or rs.startswith("exit=0"):
        st = "OK"
    else:
        st = "OTHER"
    m = re.match(r"(\d+)\s+(matches|files matched|entries|results|tests)", rs)
    if m:
        n = int(m.group(1))
        cnt = "0" if n == 0 else ("1" if n == 1 else ("2_5" if n <= 5 else "6P"))
    else:
        cnt = ""
    return f"{name}:{st}", cnt


states = [rs_state(i) for i in ids]
st_arr = np.array([s[0] for s in states])

print(f"{'last_action:state':28s} {'n':>6s} {'Qwen acc':>8s} {'top label(share)':>26s}")
rows = []
for st in pd.unique(st_arr):
    m = st_arr == st
    if m.sum() < 300:
        continue
    acc = (pred[m] == y[m]).mean()
    top = pd.Series([classes[i] for i in y[m]]).value_counts()
    rows.append((st, m.sum(), acc, top.index[0], top.iloc[0] / m.sum()))
for st, n, acc, tl, ts in sorted(rows, key=lambda r: r[2]):
    print(f"{st:28s} {n:6d} {acc:8.3f} {tl:>18s} ({ts:.0%})")

# 종합: ERR/ZERO 상태의 전체 규모와 정확도
for tag in ["ERR", "ZERO"]:
    m = np.array([tag in s for s in st_arr])
    print(f"\n전체 {tag}: n={m.sum()} ({m.mean():.1%}), Qwen acc={(pred[m]==y[m]).mean():.3f} (전체 {((pred==y).mean()):.3f})")
