# -*- coding: utf-8 -*-
"""Build train_prepared_serial_v3.parquet using featurize_v3."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from featurize_v3 import build_text

DATA = r"C:\Users\joon2\Desktop\da2\open\data"
ART = r"C:\Users\joon2\Desktop\da2\artifacts"
N_FOLDS = 5
SEED = 42


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ART, "train_prepared_serial_v3.parquet"))
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    train = load_jsonl(os.path.join(DATA, "train.jsonl"))
    labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
    lab_map = dict(zip(labels["id"], labels["action"]))

    ids, sessions, texts, ys = [], [], [], []
    for r in train:
        rid = r["id"]
        ids.append(rid)
        sessions.append(rid.rsplit("-step_", 1)[0])
        texts.append(build_text(r))
        ys.append(lab_map[rid])

    classes = sorted(set(ys))
    cls2id = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls2id[y] for y in ys])

    df = pd.DataFrame({"id": ids, "session": sessions, "text": texts,
                       "label": ys, "y": y_idx})

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for f, (_, va) in enumerate(sgkf.split(df, df["y"], groups=df["session"])):
        df.loc[va, "fold"] = f
    assert (df["fold"] >= 0).all()

    clen = df["text"].str.len()
    print("=== V3 serialized text char length ===", flush=True)
    print(f"  median {clen.median():.0f}  mean {clen.mean():.0f}  "
          f"p95 {np.percentile(clen,95):.0f}  p99 {np.percentile(clen,99):.0f}  max {clen.max()}",
          flush=True)

    print("\n=== fold sizes & class coverage ===", flush=True)
    for f in range(N_FOLDS):
        sub = df[df.fold == f]
        print(f"  fold {f}: n={len(sub):6d}  sessions={sub.session.nunique():5d}  "
              f"classes={sub.label.nunique()}", flush=True)
    leak = df.groupby("session")["fold"].nunique().max()
    print(f"\nmax folds any session appears in: {leak} (must be 1)", flush=True)

    df.to_parquet(args.out)
    with open(os.path.join(ART, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ART, "sample_serialized_v3.txt"), "w", encoding="utf-8") as f:
        for i in [0, 1, 2, 10, 100]:
            f.write(f"===== id={df.id.iloc[i]}  label={df.label.iloc[i]} =====\n")
            f.write(df.text.iloc[i] + "\n\n")
    print(f"\nSaved: {args.out}  ({len(df)} rows)", flush=True)
    print(f"Saved: {ART}\\classes.json  ({len(classes)} classes)", flush=True)
    print(f"Saved: {ART}\\sample_serialized_v3.txt", flush=True)


if __name__ == "__main__":
    main()

