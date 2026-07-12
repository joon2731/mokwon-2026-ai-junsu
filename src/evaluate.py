"""Macro-F1 평가 + OOF 확률 기반 클래스별 로짓 바이어스 튜닝 (coordinate ascent).

핵심 함수:
  fast_macro_f1(pred_idx, true_idx, C)      — bincount 기반 고속 macro-F1
  tune_biases(probs, y_idx)                 — log P + b 의 b를 클래스별로 탐색
  cv_tuned_macro_f1(probs, y_idx, folds)    — fold-out으로 튜닝 이득을 '정직하게' 추정
      (같은 데이터에서 튜닝+평가하면 낙관 편향. fold f 평가 시 나머지 fold로만 튜닝)
"""
import numpy as np


def fast_macro_f1(pred_idx, true_idx, n_classes):
    cm = np.bincount(
        true_idx * n_classes + pred_idx, minlength=n_classes * n_classes
    ).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean()


def per_class_f1(pred_idx, true_idx, classes):
    n = len(classes)
    cm = np.bincount(true_idx * n + pred_idx, minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return {c: round(float(v), 4) for c, v in zip(classes, f1)}


def tune_tau(probs, y_idx, taus=None):
    """Logit adjustment 초기화: bias = -tau * log(prior), tau를 1-D 탐색.

    Fisher-consistent(Menon et al. ICLR'21)한 사전확률 보정. 1-파라미터라
    coordinate ascent보다 과적합 위험이 낮다 → tune_biases의 초기값으로 사용.
    """
    if taus is None:
        taus = np.linspace(0.0, 2.0, 41)
    C = probs.shape[1]
    prior = np.bincount(y_idx, minlength=C) / len(y_idx)
    log_prior = np.log(np.clip(prior, 1e-9, None))
    logp = np.log(np.clip(probs, 1e-9, None))
    best_tau, best = 0.0, -1.0
    for t in taus:
        s = fast_macro_f1(np.argmax(logp - t * log_prior, axis=1), y_idx, C)
        if s > best:
            best_tau, best = float(t), s
    return best_tau, -best_tau * log_prior, best


def tune_biases(probs, y_idx, n_rounds=4, grid=None, init_bias=None):
    """argmax(log p + bias) 의 클래스별 bias를 coordinate ascent로 최적화.

    init_bias 미지정 시 tune_tau의 logit adjustment로 초기화한 뒤 정밀화한다.
    반환: (bias 벡터, 튜닝 후 macro-F1). 튜닝은 반드시 OOF 확률로 할 것.
    """
    if grid is None:
        grid = np.linspace(-1.5, 1.5, 31)
    logp = np.log(np.clip(probs, 1e-9, None))
    C = probs.shape[1]
    if init_bias is None:
        _, init_bias, _ = tune_tau(probs, y_idx)
    bias = np.asarray(init_bias, dtype=np.float64).copy()
    best = fast_macro_f1(np.argmax(logp + bias, axis=1), y_idx, C)
    for _ in range(n_rounds):
        improved = False
        for c in range(C):
            cur = bias[c]
            for g in cur + grid:  # 현재값 기준 상대 탐색 (init이 그리드 밖이어도 유지됨)
                if g == cur:
                    continue
                bias[c] = g
                s = fast_macro_f1(np.argmax(logp + bias, axis=1), y_idx, C)
                if s > best + 1e-6:
                    best = s
                    cur = g
                    improved = True
            bias[c] = cur
        if not improved:
            break
    return bias, best


def cv_tuned_macro_f1(probs, y_idx, folds, n_rounds=4):
    """fold-out 튜닝: fold f를 평가할 때 나머지 fold OOF로만 bias를 튜닝.

    반환: fold별 (튜닝 전, 튜닝 후) macro-F1 리스트.
    """
    C = probs.shape[1]
    out = []
    for f in sorted(set(folds.tolist())):
        tr = folds != f
        va = folds == f
        bias, _ = tune_biases(probs[tr], y_idx[tr], n_rounds=n_rounds)
        logp_va = np.log(np.clip(probs[va], 1e-9, None))
        before = fast_macro_f1(np.argmax(logp_va, axis=1), y_idx[va], C)
        after = fast_macro_f1(np.argmax(logp_va + bias, axis=1), y_idx[va], C)
        out.append((float(before), float(after)))
    return out
