"""테스트 분포 시프트 보정 검증 — OOF로 만든 의사-테스트(세션당 1스텝, au 15%)에서
EM prior 적응(Saerens et al. 2002)이 macro-F1을 올리는지 측정.

실행: python src\\test_shift_adapt.py  (리포지토리 루트, CPU 전용)
근거: 테스트는 세션당 1스텝(턴 분포 시프트) + au ~15%(train 7.2%) — docs/01_data.md
"""
import json

import numpy as np
import pandas as pd

ART = r"C:\Users\joon2\Desktop\da2\artifacts"
N_DRAWS = 20
AU_SHARE = 0.15

classes = json.load(open(ART + r"\classes.json", encoding="utf-8"))
C = len(classes)
df = pd.read_parquet(ART + r"\train_prepared.parquet")

ids, logits, y = [], [], []
for f in range(5):
    z = np.load(ART + rf"\oof\qwen3_smoke_fold{f}.npz", allow_pickle=True)
    ids.extend(z["ids"].tolist())
    logits.append(z["logits"])
    y.append(z["y"])
ids = np.array(ids)
y = np.concatenate(y)
probs = np.exp(logits := np.vstack(logits) - np.vstack(logits).max(1, keepdims=True))
probs /= probs.sum(1, keepdims=True)

sess = pd.Series(ids).str.replace(r"-step_\d+$", "", regex=True).values
is_au_row = pd.Series(ids).str.startswith("sess_au").values
train_prior = np.bincount(y, minlength=C) / len(y)


def macro_f1(pred, true):
    cm = np.bincount(true * C + pred, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    return np.divide(2 * tp, d, out=np.zeros_like(tp), where=d > 0).mean()


def em_prior(p, iters=50):
    """모델 확률 p(train-prior 기준)에서 테스트 prior를 EM으로 추정."""
    pt = train_prior.copy()
    for _ in range(iters):
        w = p * (pt / train_prior)
        w /= w.sum(1, keepdims=True)
        pt_new = w.mean(0)
        if np.abs(pt_new - pt).max() < 1e-7:
            break
        pt = pt_new
    return pt


def adapt(p, pt, lam=1.0):
    """posterior 재보정: p * (pt/train_prior)^lam 후 정규화."""
    ratio = (pt / train_prior) ** lam
    q = p * ratio
    return q / q.sum(1, keepdims=True)


# 의사-테스트 드로우: 세션당 1스텝 무작위 + au 세션 오버샘플(15%)
rng = np.random.default_rng(42)
sess_ids = pd.unique(sess)
sess_is_au = {s: s.startswith("sess_au") for s in sess_ids}
au_sess = np.array([s for s in sess_ids if sess_is_au[s]])
sim_sess = np.array([s for s in sess_ids if not sess_is_au[s]])
row_by_sess = pd.Series(range(len(ids)), index=sess).groupby(level=0).apply(list)

results = {k: [] for k in ["raw", "em_g", "em_g_half", "em_seg"]}
est_priors = []
for d in range(N_DRAWS):
    # au 15%가 되도록 au 세션 복원추출 확장
    n_au = int(AU_SHARE / (1 - AU_SHARE) * len(sim_sess))
    pick_au = rng.choice(au_sess, size=n_au, replace=True)
    chosen = np.concatenate([sim_sess, pick_au])
    rows = np.array([rng.choice(row_by_sess[s]) for s in chosen])
    p, t = probs[rows], y[rows]
    au_m = is_au_row[rows]

    results["raw"].append(macro_f1(p.argmax(1), t))
    pt = em_prior(p)
    est_priors.append(pt)
    results["em_g"].append(macro_f1(adapt(p, pt).argmax(1), t))
    results["em_g_half"].append(macro_f1(adapt(p, pt, 0.5).argmax(1), t))
    # 세그먼트별 EM (au / sim 분리)
    q = p.copy()
    for m in (au_m, ~au_m):
        if m.sum() > 100:
            q[m] = adapt(p[m], em_prior(p[m]))
    results["em_seg"].append(macro_f1(q.argmax(1), t))

print(f"의사-테스트 (세션당 1스텝, au {AU_SHARE:.0%}, {N_DRAWS} draws):")
for k, v in results.items():
    v = np.array(v)
    print(f"  {k:10s} {v.mean():.5f} +- {v.std():.5f}   (raw 대비 {v.mean()-np.mean(results['raw']):+.5f})")

# 추정 prior vs train prior 비교 (시프트가 실제로 감지되는가)
pt_mean = np.mean(est_priors, axis=0)
print("\n클래스별 train prior -> EM 추정 테스트 prior (상위 변화 6개):")
delta = pt_mean - train_prior
for i in np.argsort(-np.abs(delta))[:6]:
    print(f"  {classes[i]:18s} {train_prior[i]:.4f} -> {pt_mean[i]:.4f} ({delta[i]:+.4f})")
