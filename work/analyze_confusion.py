# -*- coding: utf-8 -*-
"""Confusion analysis of a saved OOF fold to see which classes get mixed up."""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

ART = r"C:\Users\joon2\Desktop\da2\artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="xlmr_v1")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
    d = np.load(os.path.join(ART, "oof", f"{args.tag}_fold{args.fold}.npz"), allow_pickle=True)
    y, logits = d["y"], d["logits"]
    pred = logits.argmax(1)

    cm = confusion_matrix(y, pred, labels=list(range(len(classes))))
    row_norm = cm / cm.sum(1, keepdims=True)
    f1s = f1_score(y, pred, average=None, labels=list(range(len(classes))))

    print(f"tag={args.tag} fold={args.fold}  macro-F1={f1_score(y,pred,average='macro'):.4f}\n")
    print("For each TRUE class: F1, then where its errors go (pred: %):")
    order = np.argsort(f1s)  # weakest first
    for i in order:
        row = row_norm[i]
        # top predicted classes for this true class
        top = np.argsort(row)[::-1][:4]
        parts = []
        for j in top:
            if row[j] > 0.02:
                mark = "*" if j == i else " "
                parts.append(f"{mark}{classes[j]}:{row[j]*100:.0f}%")
        print(f"  {classes[i]:18s} F1={f1s[i]:.3f} | " + "  ".join(parts))

    # highlight the file-navigation cluster confusion
    nav = ["read_file", "grep_search", "glob_pattern", "list_directory"]
    idx = [classes.index(c) for c in nav]
    sub = cm[np.ix_(idx, idx)]
    within = sub.sum() / cm[idx, :].sum()
    print(f"\nfile-nav cluster {nav}:")
    print(f"  fraction of cluster's true samples predicted as *some* cluster member = {within*100:.1f}%")
    print("  (high => model knows it's a nav action but picks the wrong one)")


if __name__ == "__main__":
    main()
