# -*- coding: utf-8 -*-
"""Reverse-engineer what distinguishes the confusable clusters (legit EDA).

Focus: read_file / grep_search / glob_pattern / list_directory.
Question: given it's a nav action, what predicts WHICH one?
"""
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

DATA = r"C:\Users\joon2\Desktop\da2\open\data"
OUT = r"C:\Users\joon2\Desktop\da2\artifacts\cluster_analysis.txt"

CLUSTERS = {
    "nav": ["read_file", "grep_search", "glob_pattern", "list_directory"],
    "verify": ["run_tests", "lint_or_typecheck", "run_bash"],
    "comm": ["ask_user", "plan_task", "web_search"],
    "edit": ["edit_file", "apply_patch", "write_file"],
}

TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[가-힣]+|[^\sA-Za-z0-9가-힣]")


def load():
    rows = []
    with open(os.path.join(DATA, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    lab = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
    lab_map = dict(zip(lab["id"], lab["action"]))
    return rows, lab_map


def tok(s):
    return [t.lower() for t in TOKEN_RE.findall(s or "") if not re.match(r"^[^\sA-Za-z0-9가-힣]$", t)]


def last_action(r):
    for t in reversed(r.get("history", []) or []):
        if t.get("role") == "assistant_action":
            return t.get("name")
    return "NONE"


def main():
    out = open(OUT, "w", encoding="utf-8")
    def p(*a):
        print(*a); out.write(" ".join(str(x) for x in a) + "\n")

    rows, lab_map = load()
    for r in rows:
        r["_y"] = lab_map[r["id"]]

    for cname, classes in CLUSTERS.items():
        sub = [r for r in rows if r["_y"] in classes]
        p("\n" + "#" * 70)
        p(f"# CLUSTER '{cname}': {classes}  (n={len(sub)})")
        p("#" * 70)

        # ---- token -> which class (predictive words) ----
        tok_class = defaultdict(Counter)
        tok_total = Counter()
        for r in sub:
            seen = set(tok(r.get("current_prompt", "")))
            for w in seen:
                tok_class[w][r["_y"]] += 1
                tok_total[w] += 1

        p("\n[Distinctive prompt words per class]  (word: P(class|word), count)")
        for c in classes:
            cand = []
            for w, tot in tok_total.items():
                if tot >= 40 and len(w) >= 2:
                    frac = tok_class[w][c] / tot
                    cand.append((frac, tot, w))
            cand.sort(reverse=True)
            top = cand[:12]
            p(f"  -> {c}:")
            for frac, tot, w in top:
                p(f"       {w:14s} P={frac:.2f}  n={tot}")

        # ---- metadata signal ----
        p("\n[Metadata by class]  tier/lang/ci  &  turn_index mean")
        for c in classes:
            cc = [r for r in sub if r["_y"] == c]
            tier = Counter(r["session_meta"].get("user_tier") for r in cc)
            ci = Counter(r["session_meta"]["workspace"].get("last_ci_status") for r in cc)
            turn = np.mean([r["session_meta"].get("turn_index", 0) for r in cc])
            nopen = np.mean([len(r["session_meta"]["workspace"].get("open_files") or []) for r in cc])
            p(f"  {c:16s} turn~{turn:.1f} open~{nopen:.2f} | tier={dict(tier)} | ci={dict(ci)}")

        # ---- last action -> this class ----
        p("\n[Last history action -> class share]")
        la_class = defaultdict(Counter)
        for r in sub:
            la_class[last_action(r)][r["_y"]] += 1
        for la in sorted(la_class, key=lambda k: -sum(la_class[k].values()))[:6]:
            tot = sum(la_class[la].values())
            dist = ", ".join(f"{c}:{la_class[la][c]/tot*100:.0f}%" for c in classes)
            p(f"  last={la:16s} (n={tot:5d}) -> {dist}")

    out.close()
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
