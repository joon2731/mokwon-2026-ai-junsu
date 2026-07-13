"""시프트-매칭 의사-테스트에서 F1 직접 최적화 bias의 '정직한' 전이 검증.

프로토콜 (세션 누수 차단):
  세션을 반으로 분할 → A(튜닝용)·B(평가용) 각각 "세션당 1스텝 + au 15%" 의사-테스트 구성
  A에서 coordinate-ascent bias(macro-F1 직접 최대화) 튜닝 → B에서 raw vs 보정 비교
  bias 변형: 전역 / au-조건부 / turn-조건부(<=2, 3+)
실행: python src\\test_shift_bias.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\joon2\Desktop\da2\src")
from evaluate import fast_macro_f1, tune_biases

ART = r"C:\Users\joon2\Desktop\dacon\artifacts"
N_DRAWS = 10
AU_SHARE = 0.15

classes = json.load(open(ART + r"\classes.json", encoding="utf-8"))
C = len(classes)

ids, logits, y = [], [], []
for f in range(5):
    z = np.load(ART + rf"\oof\qwen3_smoke_fold{f}.npz", allow_pickle=True)
    ids.extend(z["ids"].tolist())
    logits.append(z["logits"])
    y.append(z["y"])
ids = np.array(ids)
y = np.concatenate(y)
L = np.vstack(logits)
probs = np.exp(L - L.max(1, keepdims=True))
probs /= probs.sum(1, keepdims=True)

df = pd.read_parquet(ART + r"\train_prepared.parquet").set_index("id")
turn = df.loc[ids, "text"].str.extract(r"turn=(\d+)")[0].astype(int).values
sess = pd.Series(ids).str.replace(r"-step_\d+$", "", regex=True).values
is_au = pd.Series(ids).str.startswith("sess_au").values

rng = np.random.default_rng(7)
sess_ids = pd.unique(sess)
rng.shuffle(sess_ids)
half = len(sess_ids) // 2
sess_A, sess_B = set(sess_ids[:half]), set(sess_ids[half:])
row_by_sess = pd.Series(range(len(ids)), index=sess).groupby(level=0).apply(list)


def draw(sess_pool, seed):
    r = np.random.default_rng(seed)
    pool = [s for s in sess_pool]
    au_p = [s for s in pool if s.startswith("sess_au")]
    sim_p = [s for s in pool if not s.startswith("sess_au")]
    n_au = int(AU_SHARE / (1 - AU_SHARE) * len(sim_p))
    chosen = sim_p + list(r.choice(au_p, size=n_au, replace=True))
    return np.array([r.choice(row_by_sess[s]) for s in chosen])


def apply_bias(logp, rows, bias_g=None, bias_au=None, bias_turn=None):
    q = logp[rows].copy()
    if bias_g is not None:
        q += bias_g
    if bias_au is not None:
        q[is_au[rows]] += bias_au
    if bias_turn is not None:
        lo = turn[rows] <= 2
        q[lo] += bias_turn
    return q


logp = np.log(np.clip(probs, 1e-9, None))

# A에서 튜닝 (draw 3개 합쳐 안정화)
rows_A = np.concatenate([draw(sess_A, 100 + i) for i in range(3)])
base_A = fast_macro_f1(logp[rows_A].argmax(1), y[rows_A], C)
bias_g, tuned_A = tune_biases(probs[rows_A], y[rows_A])
print(f"[A 튜닝셋] raw {base_A:.5f} -> global bias 튜닝 후 {tuned_A:.5f} (in-sample)")

# au-조건부: 전역 bias 위에 au 행 추가 bias
au_rows_A = rows_A[is_au[rows_A]]
bias_au, _ = tune_biases(
    np.exp(logp[au_rows_A] + bias_g) / np.exp(logp[au_rows_A] + bias_g).sum(1, keepdims=True),
    y[au_rows_A])
# turn-조건부: 전역 위에 저턴 행 추가 bias
lo_rows_A = rows_A[turn[rows_A] <= 2]
bias_tn, _ = tune_biases(
    np.exp(logp[lo_rows_A] + bias_g) / np.exp(logp[lo_rows_A] + bias_g).sum(1, keepdims=True),
    y[lo_rows_A])

# B에서 정직 평가 (fresh draws)
res = {k: [] for k in ["raw", "global", "global+au", "global+turn", "g+au+turn"]}
for i in range(N_DRAWS):
    rows = draw(sess_B, 200 + i)
    t = y[rows]
    res["raw"].append(fast_macro_f1(logp[rows].argmax(1), t, C))
    res["global"].append(fast_macro_f1(apply_bias(logp, rows, bias_g).argmax(1), t, C))
    res["global+au"].append(fast_macro_f1(apply_bias(logp, rows, bias_g, bias_au).argmax(1), t, C))
    res["global+turn"].append(fast_macro_f1(apply_bias(logp, rows, bias_g, None, bias_tn).argmax(1), t, C))
    res["g+au+turn"].append(fast_macro_f1(apply_bias(logp, rows, bias_g, bias_au, bias_tn).argmax(1), t, C))

print(f"\n[B 홀드아웃] {N_DRAWS} draws (세션 분리, 1스텝/세션, au 15%):")
raw_m = np.mean(res["raw"])
for k, v in res.items():
    v = np.array(v)
    print(f"  {k:12s} {v.mean():.5f} +- {v.std():.5f}  (raw 대비 {v.mean()-raw_m:+.5f})")
print("\nglobal bias 벡터:", np.round(bias_g, 2).tolist())
