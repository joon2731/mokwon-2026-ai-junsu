"""turn-1(히스토리 없음) 구간에서 메타-전용 보정기의 헤드룸 측정.

이 구간(9,000행, 12.9%)은 Qwen acc 0.576으로 최악. 입력이 프롬프트+메타뿐이라
"모델이 텍스트로 다 읽는다" 논리가 얼마나 성립하는지 메타-전용 GBDT로 검증:
  (a) 메타만으로 얼마나 맞나 (b) Qwen 확률과 블렌드하면 turn-1 macro가 오르나 (fold-정직)
이득이 있으면: turn-1 한정 소형 보정기(sklearn, 추론비용 ~0)로 제출 탑재 가능.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ART = r"C:\Users\joon2\Desktop\dacon\artifacts"
classes = json.load(open(ART + r"\classes.json", encoding="utf-8"))
C = len(classes)

samples = [json.loads(l) for l in open(r"C:\Users\joon2\Desktop\da2\data\train.jsonl", encoding="utf-8") if l.strip()]
by_id = {s["id"]: s for s in samples}

ids, logits, y = [], [], []
folds = []
for f in range(5):
    z = np.load(ART + rf"\oof\qwen3_smoke_fold{f}.npz", allow_pickle=True)
    ids.extend(z["ids"].tolist()); logits.append(z["logits"]); y.append(z["y"])
    folds.extend([f] * len(z["y"]))
ids = np.array(ids); L = np.vstack(logits); y = np.concatenate(y); folds = np.array(folds)
probs_q = np.exp(L - L.max(1, keepdims=True)); probs_q /= probs_q.sum(1, keepdims=True)

# turn-1만
t1 = np.array([by_id[i]["session_meta"]["turn_index"] == 1 for i in ids])
print(f"turn-1 rows: {t1.sum()}  Qwen acc {(probs_q[t1].argmax(1)==y[t1]).mean():.4f}")


def macro(pred, true):
    cm = np.bincount(true * C + pred, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    return np.divide(2 * tp, d, out=np.zeros_like(tp), where=d > 0).mean()


# 메타 피처 구성 (turn-1 행만)
LANG_KEYS = ["py", "ts", "tsx", "js", "java", "go", "rs", "md", "yaml", "json", "css", "html", "dockerfile", "sh"]
rows = []
idx_t1 = np.where(t1)[0]
for k in idx_t1:
    s = by_id[ids[k]]
    m = s["session_meta"]; ws = m.get("workspace", {})
    lm = ws.get("language_mix") or {}
    of = ws.get("open_files") or []
    prompt = s.get("current_prompt") or ""
    r = {
        "tier": {"free": 0, "pro": 1, "enterprise": 2}.get(m.get("user_tier"), -1),
        "lang": {"ko": 0, "en": 1, "mixed": 2}.get(m.get("language_pref"), -1),
        "ci": {"none": 0, "passed": 1, "failed": 2}.get(ws.get("last_ci_status"), -1),
        "dirty": int(bool(ws.get("git_dirty"))),
        "loc": np.log1p(ws.get("loc") or 0),
        "budget": np.log1p(m.get("budget_tokens_remaining") or 0),
        "elapsed": np.log1p(m.get("elapsed_session_sec") or 0),
        "n_open": len(of),
        "prompt_len": len(prompt),
        "has_path": int("/" in prompt or "\\" in prompt or ".py" in prompt or ".ts" in prompt),
        "has_q": int("?" in prompt),
        "au": int(ids[k].startswith("sess_au")),
    }
    for lk in LANG_KEYS:
        r[f"lm_{lk}"] = lm.get(lk, 0.0)
    rows.append(r)
X = pd.DataFrame(rows).values
y1 = y[idx_t1]; f1 = folds[idx_t1]; pq1 = probs_q[idx_t1]

# fold-정직 메타 GBDT OOF
meta_oof = np.zeros((len(idx_t1), C))
for f in range(5):
    tr, va = f1 != f, f1 == f
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                         early_stopping=False, random_state=42)
    clf.fit(X[tr], y1[tr])
    p = clf.predict_proba(X[va])
    cols = clf.classes_
    full = np.zeros((va.sum(), C)); full[:, cols] = p
    meta_oof[va] = full

acc_meta = (meta_oof.argmax(1) == y1).mean()
print(f"메타-전용 GBDT: acc {acc_meta:.4f}, macro {macro(meta_oof.argmax(1), y1):.4f}")
print(f"Qwen (turn-1): acc {(pq1.argmax(1)==y1).mean():.4f}, macro {macro(pq1.argmax(1), y1):.4f}")

# 블렌드 스윕 (turn-1 한정)
base_m = macro(pq1.argmax(1), y1)
best = (0, base_m)
for w in np.linspace(0, 0.5, 21):
    bl = (1 - w) * pq1 + w * meta_oof
    s = macro(bl.argmax(1), y1)
    if s > best[1]:
        best = (w, s)
print(f"\nturn-1 블렌드 best w_meta={best[0]:.3f}: macro {base_m:.4f} -> {best[1]:.4f} ({best[1]-base_m:+.4f})")
overall = macro(probs_q.argmax(1), y)
# 전체 반영치: turn-1 행만 교체했을 때 전체 macro
pred_all = probs_q.argmax(1).copy()
if best[0] > 0:
    bl = (1 - best[0]) * pq1 + best[0] * meta_oof
    pred_all[idx_t1] = bl.argmax(1)
print(f"전체 macro: {overall:.4f} -> {macro(pred_all, y):.4f} ({macro(pred_all, y)-overall:+.4f})")
