"""OOF 블렌드 가중 탐색 — Qwen3 / XLM-R / mmBERT 조합의 macro-F1 최적 가중.

mmBERT 5-fold 완료 직후 실행:
  python src\\blend_search.py

목적: mmBERT가 Qwen과의 블렌드에서 기존 XLM-R 블렌드(0.7735)를 넘는지,
넘으면 어느 2-way/3-way 가중이 최적인지. 넘으면 서버 추론시간 실측 → 패키징 판단.
전 OOF는 세션 GroupKFold라 정직(누수 없음).
"""
import json
import numpy as np

ART = r"C:\Users\joon2\Desktop\dacon\artifacts"
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


ids_q, lq, yq = load("qwen3_smoke")
P = {"qwen": softmax(lq)}
y = yq

for tag, name in [("xlmr_v2_rdrop_lr4_e4", "xlmr"), ("mmbert_v2", "mmbert")]:
    try:
        ids_t, lt, yt = load(tag)
        o = {i: k for k, i in enumerate(ids_t)}
        idx = np.array([o[i] for i in ids_q])
        assert (yt[idx] == yq).all()
        P[name] = softmax(lt[idx])
        print(f"{name:8s} OOF macro = {macro(P[name].argmax(1), y):.4f}")
    except FileNotFoundError:
        print(f"{name}: OOF 없음 (스킵)")

print(f"qwen     OOF macro = {macro(P['qwen'].argmax(1), y):.4f}")
print()

# 2-way: qwen + 각 파트너
for partner in ["xlmr", "mmbert"]:
    if partner not in P:
        continue
    best = (0, 0)
    for w in np.linspace(0, 1, 41):
        s = macro((w * P["qwen"] + (1 - w) * P[partner]).argmax(1), y)
        if s > best[1]:
            best = (w, s)
    print(f"qwen×{partner:6s} best w={best[0]:.2f} → {best[1]:.4f}  (vs qwen 단독 {macro(P['qwen'].argmax(1), y):.4f})")

# 3-way (mmbert 있을 때)
if "mmbert" in P and "xlmr" in P:
    best = (None, 0)
    for wq in np.linspace(0, 1, 21):
        for wx in np.linspace(0, 1 - wq, 21):
            wm = 1 - wq - wx
            if wm < 0:
                continue
            s = macro((wq * P["qwen"] + wx * P["xlmr"] + wm * P["mmbert"]).argmax(1), y)
            if s > best[1]:
                best = ((round(wq, 2), round(wx, 2), round(wm, 2)), s)
    print(f"3-way best (q,x,m)={best[0]} → {best[1]:.4f}")

print("\n기준: 기존 qwen x xlmr 블렌드 0.7735. 이걸 넘는 조합만 서버시간 실측 가치 있음.")
print("주의: 추론 10분 예산 - qwen(9:31)+인코더 1개가 한계. 3-way는 시간 초과 확실.")
