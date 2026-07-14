# -*- coding: utf-8 -*-
"""Phase 0: build serialized text + session-grouped stratified folds, save."""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from featurize import build_text

DATA = r"C:\Users\joon2\Desktop\da2\open\data"
ART = r"C:\Users\joon2\Desktop\da2\artifacts"
os.makedirs(ART, exist_ok=True)
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

    # session-grouped, stratified folds
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for f, (_, va) in enumerate(sgkf.split(df, df["y"], groups=df["session"])):
        df.loc[va, "fold"] = f
    assert (df["fold"] >= 0).all()

    # char length stats
    clen = df["text"].str.len()
    print("=== serialized text char length ===")
    print(f"  median {clen.median():.0f}  mean {clen.mean():.0f}  "
          f"p95 {np.percentile(clen,95):.0f}  p99 {np.percentile(clen,99):.0f}  max {clen.max()}")

    # per-fold class balance sanity
    print("\n=== fold sizes & class coverage ===")
    for f in range(N_FOLDS):
        sub = df[df.fold == f]
        print(f"  fold {f}: n={len(sub):6d}  sessions={sub.session.nunique():5d}  "
              f"classes={sub.label.nunique()}")
    # confirm no session leaks across folds
    leak = df.groupby("session")["fold"].nunique().max()
    print(f"\nmax folds any session appears in: {leak} (must be 1)")

    # try tokenizer length (optional; needs model files)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        sample_txt = df["text"].sample(2000, random_state=0).tolist()
        lens = [len(tok(t, truncation=False)["input_ids"]) for t in sample_txt]
        lens = np.array(lens)
        print("\n=== mDeBERTa token length (2000 sample) ===")
        print(f"  median {np.median(lens):.0f}  p90 {np.percentile(lens,90):.0f}  "
              f"p95 {np.percentile(lens,95):.0f}  p99 {np.percentile(lens,99):.0f}  max {lens.max()}")
        for L in (128, 160, 192, 256):
            print(f"  <= {L} tokens: {(lens<=L).mean()*100:.1f}%")
    except Exception as e:
        print(f"\n[tokenizer check skipped] {type(e).__name__}: {e}")

    df.to_parquet(os.path.join(ART, "train_prepared.parquet"))
    with open(os.path.join(ART, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {ART}\\train_prepared.parquet  ({len(df)} rows)")
    print(f"Saved: {ART}\\classes.json  ({len(classes)} classes)")

    # write a few decoded serialized samples to a UTF-8 file for inspection
    with open(os.path.join(ART, "sample_serialized.txt"), "w", encoding="utf-8") as f:
        for i in [0, 1, 2, 10, 100]:
            f.write(f"===== id={df.id.iloc[i]}  label={df.label.iloc[i]} =====\n")
            f.write(df.text.iloc[i] + "\n\n")
    print(f"Saved: {ART}\\sample_serialized.txt")


if __name__ == "__main__":
    main()
