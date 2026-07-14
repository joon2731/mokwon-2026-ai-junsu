# -*- coding: utf-8 -*-
"""Assemble 5-fold OOF per tag, evaluate, and (optionally) blend multiple tags.

All tags must be trained on the SAME fold split (artifacts/train_prepared.parquet),
which makes row-wise blending across tags an honest OOF evaluation.

Usage:
  python work\\blend_oof.py --tags xlmr_v2_rdrop_lr4_e4
  python work\\blend_oof.py --tags xlmr_v2_rdrop_lr4_e4,qwen05_smoke --step 0.05

Outputs per-fold + concat macro-F1 per tag; for 2+ tags searches blend weights
(coordinate ascent on the simplex) and reports a fold-split honest check
(fit weights on folds 0-2, eval on 3-4, and vice versa).
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import f1_score

ART = r"C:\Users\joon2\Desktop\da2\artifacts"


def load_tag(tag):
    """-> {fold: (ids, logits, y)} for folds that exist."""
    out = {}
    for f in range(5):
        p = os.path.join(ART, "oof", f"{tag}_fold{f}.npz")
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            out[f] = (np.asarray(d["ids"]).astype(str), d["logits"], d["y"])
    return out


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def macro(y, probs):
    return f1_score(y, probs.argmax(1), average="macro")


def align(tag_folds, tags):
    """Align rows by id on folds common to all tags.

    -> y (N,), fold_id (N,), probs {tag: (N, C)}
    """
    common = sorted(set.intersection(*(set(tag_folds[t].keys()) for t in tags)))
    if not common:
        raise SystemExit("no common folds across tags")
    ys, folds, probs = [], [], {t: [] for t in tags}
    for f in common:
        ref_ids, _, ref_y = tag_folds[tags[0]][f]
        order0 = np.argsort(ref_ids)
        ys.append(ref_y[order0])
        folds.append(np.full(len(ref_ids), f))
        for t in tags:
            ids, lg, y = tag_folds[t][f]
            order = np.argsort(ids)
            if not np.array_equal(ids[order], ref_ids[order0]):
                raise SystemExit(f"id mismatch: tag={t} fold={f} (different split?)")
            probs[t].append(softmax(lg[order].astype(np.float64)))
    return (np.concatenate(ys), np.concatenate(folds),
            {t: np.concatenate(v) for t, v in probs.items()}, common)


def search_weights(y, probs_list, step):
    """Coordinate ascent on the weight simplex; returns (weights, macro)."""
    k = len(probs_list)
    w = np.ones(k) / k
    grid = np.arange(0.0, 1.0 + 1e-9, step)

    def blended(wv):
        m = sum(wi * p for wi, p in zip(wv, probs_list))
        return macro(y, m)

    best = blended(w)
    for _ in range(3):  # passes
        improved = False
        for i in range(k):
            for g in grid:
                cand = w.copy()
                cand[i] = g
                s = cand.sum()
                if s == 0:
                    continue
                cand = cand / s
                sc = blended(cand)
                if sc > best + 1e-6:
                    best, w, improved = sc, cand, True
        if not improved:
            break
    return w, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated OOF tags")
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    tag_folds = {t: load_tag(t) for t in tags}
    for t in tags:
        if not tag_folds[t]:
            raise SystemExit(f"no OOF files found for tag '{t}'")

    # ---- per-tag report ----
    print("== per-tag OOF ==")
    for t in tags:
        parts = []
        for f, (ids, lg, y) in sorted(tag_folds[t].items()):
            parts.append(f"f{f}={macro(y, softmax(lg.astype(np.float64))):.4f}")
        ally = np.concatenate([v[2] for _, v in sorted(tag_folds[t].items())])
        allp = np.concatenate([softmax(v[1].astype(np.float64)) for _, v in sorted(tag_folds[t].items())])
        print(f"  {t:28s} {' '.join(parts)}  | concat={macro(ally, allp):.4f} "
              f"({len(tag_folds[t])}/5 folds)")

    if len(tags) < 2:
        return

    # ---- blend ----
    y, fold_id, probs, common = align(tag_folds, tags)
    plist = [probs[t] for t in tags]
    print(f"\n== blend on common folds {common} (N={len(y)}) ==")
    for t in tags:
        print(f"  single {t:26s} {macro(y, probs[t]):.4f}")
    w_eq = np.ones(len(tags)) / len(tags)
    print(f"  equal-weight blend           {macro(y, sum(wi*p for wi, p in zip(w_eq, plist))):.4f}")
    w, sc = search_weights(y, plist, args.step)
    print(f"  best in-sample blend         {sc:.4f}  w={np.round(w, 3).tolist()}")

    # ---- honest split check (fit on A folds, eval on B) ----
    half = len(common) // 2 or 1
    a_folds, b_folds = set(common[:half + len(common) % 2]), set(common[half + len(common) % 2:])
    if a_folds and b_folds:
        ma = np.isin(fold_id, list(a_folds))
        mb = ~ma
        wA, _ = search_weights(y[ma], [p[ma] for p in plist], args.step)
        wB, _ = search_weights(y[mb], [p[mb] for p in plist], args.step)
        honest = 0.5 * (macro(y[mb], sum(w_*p[mb] for w_, p in zip(wA, plist)))
                        + macro(y[ma], sum(w_*p[ma] for w_, p in zip(wB, plist))))
        base_single = max(macro(y, p) for p in plist)
        eq = macro(y, sum(wi*p for wi, p in zip(w_eq, plist)))
        print(f"  cross-fit honest blend       {honest:.4f}  "
              f"(vs best single {base_single:.4f}, vs equal {eq:.4f})")
        print("  -> 채택 기준: honest > best single + 0.002. 아니면 equal-weight 또는 단일 유지")

    out = {"tags": tags, "weights": np.round(w, 4).tolist(), "in_sample": round(sc, 4)}
    op = os.path.join(ART, f"blend_{'_'.join(tags)}.json")
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"saved -> {op}")


if __name__ == "__main__":
    main()
