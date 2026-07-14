# -*- coding: utf-8 -*-
"""Post-hoc scan on the full 5-fold OOF (70k): scalar-tau logit adjustment vs
14-param per-class bias, each gated by a session-disjoint cross-fit estimate."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_bias import load_oof, macro, coordinate_ascent, crossfit_gain, _session_of

ART = r"C:\Users\joon2\Desktop\da2\artifacts"
TAG = "xlmr_v2_rdrop"


def main():
    classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
    n = len(classes)
    ids, probs, y = load_oof(TAG)
    logp = np.log(probs + 1e-9)
    base = macro(logp, np.zeros(n), y)
    print(f"full OOF: {len(y)} samples | plain argmax macro-F1 = {base:.4f}")

    pi = np.bincount(y, minlength=n) / len(y)
    logpi = np.log(pi)

    # --- scalar-tau logit adjustment (1 dof, robust) ---
    taus = np.arange(0.0, 1.51, 0.05)
    scores = [macro(logp, -t * logpi, y) for t in taus]
    j = int(np.argmax(scores))
    print(f"\n[scalar-tau LA] best tau={taus[j]:.2f} -> {scores[j]:.4f} "
          f"({scores[j]-base:+.4f}) in-sample")
    sess = np.array([_session_of(i) for i in ids])
    uniq = np.unique(sess)
    rng = np.random.RandomState(0)
    half = set(rng.permutation(uniq)[: len(uniq) // 2].tolist())
    a = np.array([s in half for s in sess])
    tau_gains = []
    for tr, te in ((a, ~a), (~a, a)):
        js = int(np.argmax([macro(logp[tr], -t * logpi, y[tr]) for t in taus]))
        g = macro(logp[te], -taus[js] * logpi, y[te]) - macro(logp[te], np.zeros(n), y[te])
        tau_gains.append(g)
        print(f"  half: tau*={taus[js]:.2f}  honest gain {g:+.4f}")
    print(f"  mean honest gain {np.mean(tau_gains):+.4f}")

    # --- 14-param per-class bias (regularized coordinate ascent) ---
    bias, tuned = coordinate_ascent(logp, y, n)
    print(f"\n[per-class bias] in-sample {tuned:.4f} ({tuned-base:+.4f})")
    cf = crossfit_gain(ids, logp, y, n)
    print(f"  session-disjoint cross-fit gains: {cf[0]:+.4f} / {cf[1]:+.4f} "
          f"mean {np.mean(cf):+.4f}")

    bias_v = "SHIP" if np.mean(cf) > 0.002 else ("MAYBE" if np.mean(cf) > 0 else "NO")
    tau_v = "SHIP" if np.mean(tau_gains) > 0.001 else "NO"
    print(f"\nVERDICT: per-class bias -> {bias_v} | scalar-tau LA -> {tau_v}")

    json.dump({"classes": classes, "class_bias": bias.tolist(),
               "tau_best": float(taus[j]), "oof_plain": float(base),
               "bias_crossfit_mean": float(np.mean(cf)),
               "tau_crossfit_mean": float(np.mean(tau_gains))},
              open(os.path.join(ART, f"posthoc_{TAG}.json"), "w", encoding="utf-8"),
              indent=2)
    print("saved", os.path.join(ART, f"posthoc_{TAG}.json"))


if __name__ == "__main__":
    main()
