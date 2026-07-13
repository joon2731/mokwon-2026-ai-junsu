"""OOF 다중교사 로짓 생성 — Qwen3 5-fold + XLM-R(e4) 5-fold OOF를 확률 블렌드.

교사 = 0.6·softmax(Qwen3_OOF) + 0.4·softmax(XLMR_OOF)  (검증된 가중, 홀드아웃 0.7735)
저장 형식은 log(blend_prob) — KD 손실에서 softmax(logits/T)로 쓰이므로 로짓과 등가.
OOF라서 모든 행이 '그 행을 학습하지 않은 모델'의 예측 → 암기 전이 없음 (딥리서치 ① 권장 경로).

실행: python src\\build_blend_teacher.py
산출: da2/artifacts/teacher_logits.npz (train_prepared 순서와 무관하게 ids 포함)
"""
import json

import numpy as np

ART = r"C:\Users\joon2\Desktop\dacon\artifacts"
OUT = r"C:\Users\joon2\Desktop\da2\artifacts\teacher_logits.npz"
W_QWEN = 0.6

classes = json.load(open(ART + r"\classes.json", encoding="utf-8"))
C = len(classes)


def load_oof(tag):
    ids, logits, y = [], [], []
    for f in range(5):
        z = np.load(ART + rf"\oof\{tag}_fold{f}.npz", allow_pickle=True)
        ids.extend(z["ids"].tolist())
        logits.append(z["logits"])
        y.append(z["y"])
    return np.array(ids), np.vstack(logits), np.concatenate(y)


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def macro_f1(pred, true):
    cm = np.bincount(true * C + pred, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    return np.divide(2 * tp, d, out=np.zeros_like(tp), where=d > 0).mean()


ids_q, log_q, y_q = load_oof("qwen3_smoke")
ids_x, log_x, y_x = load_oof("xlmr_v2_rdrop_lr4_e4")
order = {i: k for k, i in enumerate(ids_x)}
idx = np.array([order[i] for i in ids_q])
assert (y_x[idx] == y_q).all()

blend = W_QWEN * softmax(log_q) + (1 - W_QWEN) * softmax(log_x[idx])
print(f"Qwen3 OOF macro: {macro_f1(log_q.argmax(1), y_q):.4f}")
print(f"blend OOF macro: {macro_f1(blend.argmax(1), y_q):.4f}  (기대 ~0.7735)")

teacher_logits = np.log(np.clip(blend, 1e-9, None)).astype(np.float32)
np.savez(OUT, ids=ids_q, logits=teacher_logits)
print("saved", OUT)
