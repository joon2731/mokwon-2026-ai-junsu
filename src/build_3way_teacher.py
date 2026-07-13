"""3-way OOF 교사 로짓 생성 — Qwen3(0.40) + XLM-R(0.27) + mmBERT(0.33).

blend_search.py에서 찾은 최적 가중 (OOF 0.7754). 전 행 OOF라 암기 오염 없음.
실행: python src\\build_3way_teacher.py
산출: da2/artifacts/teacher_logits_3way.npz
"""
import json

import numpy as np

ART = r"C:\Users\joon2\Desktop\dacon\artifacts"
OUT = r"C:\Users\joon2\Desktop\da2\artifacts\teacher_logits_3way.npz"
W = {"qwen3_smoke": 0.40, "xlmr_v2_rdrop_lr4_e4": 0.27, "mmbert_v2": 0.33}

classes = json.load(open(ART + r"\classes.json", encoding="utf-8"))
C = len(classes)


def load(tag):
    ids, logits, y = [], [], []
    for f in range(5):
        z = np.load(ART + rf"\oof\{tag}_fold{f}.npz", allow_pickle=True)
        ids.extend(z["ids"].tolist()); logits.append(z["logits"]); y.append(z["y"])
    return np.array(ids), np.vstack(logits), np.concatenate(y)


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def macro(pred, y):
    cm = np.bincount(y * C + pred, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    return np.divide(2 * tp, d, out=np.zeros_like(tp), where=d > 0).mean()


ref_ids = None
blend = None
y_ref = None
for tag, w in W.items():
    ids_t, lt, yt = load(tag)
    if ref_ids is None:
        ref_ids, y_ref = ids_t, yt
        p = softmax(lt)
    else:
        o = {i: k for k, i in enumerate(ids_t)}
        idx = np.array([o[i] for i in ref_ids])
        assert (yt[idx] == y_ref).all()
        p = softmax(lt[idx])
    blend = w * p if blend is None else blend + w * p

print(f"3-way teacher OOF macro = {macro(blend.argmax(1), y_ref):.4f} (기대 0.7754)")
np.savez(OUT, ids=ref_ids, logits=np.log(np.clip(blend, 1e-9, None)).astype(np.float32))
print("saved", OUT)
