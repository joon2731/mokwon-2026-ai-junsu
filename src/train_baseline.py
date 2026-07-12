"""TF-IDF + LogisticRegression 베이스라인 CV.

실행 (리포지토리 루트에서):
  python src\\train_baseline.py --mode prompt --exp E000
  python src\\train_baseline.py --mode full   --exp E001 --class-weight balanced

산출물: artifacts/{exp}/oof_probs.npy, classes.json, report.json
OOF 확률은 train.jsonl 순서 그대로 저장한다 (threshold 튜닝·앙상블 재료).
"""
import argparse
import json
import time

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data import ARTIFACTS_DIR, load_folds, load_train
from evaluate import cv_tuned_macro_f1, fast_macro_f1, per_class_f1, tune_biases
from serialize import serialize


def build_matrix(texts_tr, texts_va):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                           sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                           max_features=250_000, sublinear_tf=True)
    Xw_tr = word.fit_transform(texts_tr)
    Xc_tr = char.fit_transform(texts_tr)
    Xw_va = word.transform(texts_va)
    Xc_va = char.transform(texts_va)
    X_tr = sparse.hstack([Xw_tr, Xc_tr], format="csr")
    X_va = sparse.hstack([Xw_va, Xc_va], format="csr")
    return X_tr, X_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prompt", "full"], required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    t0 = time.time()
    samples = load_train()
    fold_of = load_folds()
    ids = [s["id"] for s in samples]
    folds = np.array([fold_of[i] for i in ids])
    classes = sorted({s["action"] for s in samples})
    cls_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_idx[s["action"]] for s in samples])
    print(f"[{args.exp}] mode={args.mode} n={len(ids)} classes={len(classes)}")

    texts = [serialize(s, args.mode) for s in samples]
    oof = np.zeros((len(ids), len(classes)), dtype=np.float64)
    per_fold = []
    cw = None if args.class_weight == "none" else "balanced"

    for f in sorted(set(folds.tolist())):
        tf = time.time()
        tr, va = folds != f, folds == f
        X_tr, X_va = build_matrix(
            [t for t, m in zip(texts, tr) if m],
            [t for t, m in zip(texts, va) if m],
        )
        clf = LogisticRegression(C=args.C, max_iter=2000, class_weight=cw)
        clf.fit(X_tr, y[tr])
        # clf.classes_ 는 y[tr]에 등장한 클래스의 정렬 — 전 클래스 존재 가정 검증
        assert list(clf.classes_) == list(range(len(classes))), clf.classes_
        oof[va] = clf.predict_proba(X_va)
        score = fast_macro_f1(np.argmax(oof[va], axis=1), y[va], len(classes))
        per_fold.append(float(score))
        print(f"  fold {f}: macro_f1={score:.5f} "
              f"({X_tr.shape[1]} feats, {time.time() - tf:.0f}s)")

    mean, std = float(np.mean(per_fold)), float(np.std(per_fold))
    print(f"[{args.exp}] CV macro_f1 = {mean:.5f} +- {std:.5f}")

    # threshold 튜닝: fold-out(정직) + 전체 OOF(테스트 적용용 bias)
    tuned_pairs = cv_tuned_macro_f1(oof, y, folds)
    tuned_scores = [a for _, a in tuned_pairs]
    print(f"[{args.exp}] fold-out tuned macro_f1 = "
          f"{np.mean(tuned_scores):.5f} +- {np.std(tuned_scores):.5f}")
    bias_full, oof_tuned_insample = tune_biases(oof, y)
    print(f"[{args.exp}] (in-sample tuned on all OOF = {oof_tuned_insample:.5f}, "
          f"낙관 상한 참고용)")

    out_dir = ARTIFACTS_DIR / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof_probs.npy", oof.astype(np.float32))
    (out_dir / "classes.json").write_text(json.dumps(classes), encoding="utf-8")
    report = {
        "exp": args.exp,
        "method": "tfidf(word12+char24) + logreg",
        "mode": args.mode,
        "class_weight": args.class_weight,
        "C": args.C,
        "per_fold": [round(s, 5) for s in per_fold],
        "cv_mean": round(mean, 5),
        "cv_std": round(std, 5),
        "foldout_tuned": [round(s, 5) for s in tuned_scores],
        "foldout_tuned_mean": round(float(np.mean(tuned_scores)), 5),
        "bias_full_oof": [round(float(b), 3) for b in bias_full],
        "per_class_f1": per_class_f1(np.argmax(oof, axis=1), y, classes),
        "runtime_sec": round(time.time() - t0),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_dir}\\report.json ({report['runtime_sec']}s)")


if __name__ == "__main__":
    main()
