# -*- coding: utf-8 -*-
"""Phase 3: tune per-class additive logit bias on OOF to maximize macro-F1.

Loads oof/{tag}_fold*.npz (from train.py), builds full out-of-fold softmax
probs, then coordinate-ascends a 14-dim additive bias on log-probs so that
argmax(logprob + bias) maximizes macro-F1. macro-F1 is piecewise-constant in
each coordinate, so a per-coordinate grid sweep, repeated, converges.
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import f1_score

ART = r"C:\Users\joon2\Desktop\da2\artifacts"


def load_oof(tag):
    files = sorted(glob.glob(os.path.join(ART, "oof", f"{tag}_fold*.npz")))
    assert files, f"no oof files for tag={tag}"
    ids, probs, ys = [], [], []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        logits = d["logits"].astype(np.float64)
        e = np.exp(logits - logits.max(1, keepdims=True))
        probs.append(e / e.sum(1, keepdims=True))
        ys.append(d["y"])
        ids.append(d["ids"])
    return (np.concatenate(ids), np.concatenate(probs, 0), np.concatenate(ys))


def macro(logp, bias, y):
    return f1_score(y, (logp + bias).argmax(1), average="macro")


def coordinate_ascent(logp, y, n_cls, rounds=8, grid=None):
    """Coarse grid (step 0.125, bounds +-0.75) + tie-break toward 0.

    macro-F1 is piecewise-constant, so grid sweeps produce wide plateaus;
    picking a plateau's extreme edge (old argmax behavior) systematically
    inflates bias magnitudes and overfits the tuning fold. Among tied maxima
    we now pick the value closest to 0.
    """
    bias = np.zeros(n_cls)
    if grid is None:
        grid = np.linspace(-0.75, 0.75, 13)
    best = macro(logp, bias, y)
    for r in range(rounds):
        improved = False
        for c in range(n_cls):
            base = bias.copy()
            scores = []
            for g in grid:
                base[c] = g
                scores.append(macro(logp, base, y))
            scores = np.asarray(scores)
            ties = np.where(scores >= scores.max() - 1e-9)[0]
            j = int(ties[np.argmin(np.abs(grid[ties]))])
            if scores[j] > best + 1e-6:
                best = scores[j]
                bias[c] = grid[j]
                improved = True
        if not improved:
            break
    return bias, best


def _session_of(i):
    return str(i).rsplit("-step_", 1)[0]


def crossfit_gain(ids, logp, y, n_cls, seed=0):
    """Honest gain estimate: session-disjoint split-half — tune the bias on one
    half, evaluate on the other, both directions. In-sample tuned scores are
    inflated (14 free params); adopt a bias ONLY if this mean gain is clearly
    positive."""
    sess = np.array([_session_of(i) for i in ids])
    uniq = np.unique(sess)
    rng = np.random.RandomState(seed)
    half = set(rng.permutation(uniq)[: len(uniq) // 2].tolist())
    a = np.array([s in half for s in sess])
    gains = []
    for tr, te in ((a, ~a), (~a, a)):
        b, _ = coordinate_ascent(logp[tr], y[tr], n_cls)
        g = macro(logp[te], b, y[te]) - macro(logp[te], np.zeros(n_cls), y[te])
        gains.append(g)
    return gains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="xlmr_v1")
    args = ap.parse_args()

    classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
    n_cls = len(classes)
    ids, probs, y = load_oof(args.tag)
    logp = np.log(probs + 1e-9)
    print(f"OOF: {len(y)} samples, {n_cls} classes, tag={args.tag}")

    base_pred = logp.argmax(1)
    base_macro = f1_score(y, base_pred, average="macro")
    base_acc = (base_pred == y).mean()
    print(f"\n[argmax] macro-F1 = {base_macro:.4f}   acc = {base_acc:.4f}")

    bias, tuned = coordinate_ascent(logp, y, n_cls)
    tuned_pred = (logp + bias).argmax(1)
    tuned_acc = (tuned_pred == y).mean()
    print(f"[bias  ] macro-F1 = {tuned:.4f}   acc = {tuned_acc:.4f}   "
          f"(+{tuned-base_macro:.4f})  <- in-sample, INFLATED")

    gains = crossfit_gain(ids, logp, y, n_cls)
    print(f"[HONEST] session-disjoint split-half cross-fit gain: "
          f"{gains[0]:+.4f} / {gains[1]:+.4f}   mean {np.mean(gains):+.4f}")
    if np.mean(gains) <= 0:
        print("  ** bias does NOT transfer -> do not ship it (USE_BIAS=False) **")

    print("\nper-class F1 (argmax -> bias):")
    f0 = f1_score(y, base_pred, average=None, labels=list(range(n_cls)))
    f1v = f1_score(y, tuned_pred, average=None, labels=list(range(n_cls)))
    for i, c in enumerate(classes):
        print(f"  {c:18s} {f0[i]:.3f} -> {f1v[i]:.3f}   bias={bias[i]:+.2f}")

    out = os.path.join(ART, f"bias_{args.tag}.json")
    json.dump({"classes": classes, "class_bias": bias.tolist(),
               "oof_macro_argmax": base_macro, "oof_macro_bias": tuned},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
