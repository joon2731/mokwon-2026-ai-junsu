# -*- coding: utf-8 -*-
"""Phase 0 EDA: encoding-safe inspection of the DACON 236694 dataset."""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

io = sys.stdout
DATA = r"C:\Users\joon2\Desktop\da2\open\data"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    print("Loading train.jsonl ...", flush=True)
    train = load_jsonl(os.path.join(DATA, "train.jsonl"))
    labels = pd.read_csv(os.path.join(DATA, "train_labels.csv"))
    print(f"train rows          : {len(train)}")
    print(f"label rows          : {len(labels)}")

    # ---- id alignment ----
    train_ids = [r["id"] for r in train]
    lab_ids = labels["id"].tolist()
    print(f"unique train ids    : {len(set(train_ids))}")
    print(f"ids identical set   : {set(train_ids) == set(lab_ids)}")
    print(f"ids same order      : {train_ids == lab_ids}")

    lab_map = dict(zip(labels["id"], labels["action"]))

    # ---- session structure ----
    def sess_of(i):
        return i.rsplit("-step_", 1)[0]

    sessions = [sess_of(i) for i in train_ids]
    sess_counter = Counter(sessions)
    print(f"\nunique sessions     : {len(sess_counter)}")
    print(f"steps/session  mean : {np.mean(list(sess_counter.values())):.2f} "
          f"min {min(sess_counter.values())} max {max(sess_counter.values())}")

    # step number distribution
    step_nums = [int(i.rsplit("-step_", 1)[1]) for i in train_ids]
    print(f"step_NN  min/max    : {min(step_nums)} / {max(step_nums)}")

    # ---- class distribution ----
    print("\n=== CLASS DISTRIBUTION ===")
    cd = labels["action"].value_counts()
    for k, v in cd.items():
        print(f"  {k:18s} {v:6d}  {v/len(labels)*100:5.2f}%")
    print(f"  n_classes = {labels['action'].nunique()}")

    # ---- session_meta fields ----
    print("\n=== session_meta categorical fields ===")
    tiers = Counter(r["session_meta"].get("user_tier") for r in train)
    langs = Counter(r["session_meta"].get("language_pref") for r in train)
    ci = Counter(r["session_meta"]["workspace"].get("last_ci_status") for r in train)
    dirty = Counter(r["session_meta"]["workspace"].get("git_dirty") for r in train)
    print("user_tier      :", dict(tiers))
    print("language_pref  :", dict(langs))
    print("last_ci_status :", dict(ci))
    print("git_dirty      :", dict(dirty))

    # numeric fields
    turn_idx = np.array([r["session_meta"].get("turn_index", -1) for r in train])
    budget = np.array([r["session_meta"].get("budget_tokens_remaining", -1) for r in train])
    loc = np.array([r["session_meta"]["workspace"].get("loc", -1) for r in train])
    elapsed = np.array([r["session_meta"].get("elapsed_session_sec", -1) for r in train])
    n_open = np.array([len(r["session_meta"]["workspace"].get("open_files", []) or []) for r in train])
    print("\n=== numeric fields (min / median / max) ===")
    for name, arr in [("turn_index", turn_idx), ("budget_tokens", budget),
                      ("loc", loc), ("elapsed_sec", elapsed), ("n_open_files", n_open)]:
        print(f"  {name:14s} {arr.min():>10.0f} {np.median(arr):>10.0f} {arr.max():>10.0f}")

    # ---- history structure ----
    hist_len = np.array([len(r.get("history", []) or []) for r in train])
    print("\n=== history length (turns) ===")
    print(f"  min {hist_len.min()} median {np.median(hist_len):.0f} "
          f"mean {hist_len.mean():.2f} max {hist_len.max()} p95 {np.percentile(hist_len,95):.0f}")
    print(f"  empty history       : {(hist_len==0).sum()} "
          f"({(hist_len==0).mean()*100:.1f}%)")

    # roles present in history
    roles = Counter()
    action_names_in_hist = Counter()
    last_role = Counter()
    for r in train:
        h = r.get("history", []) or []
        for t in h:
            roles[t.get("role")] += 1
            if t.get("role") == "assistant_action":
                action_names_in_hist[t.get("name")] += 1
        if h:
            last_role[h[-1].get("role")] += 1
    print("\nhistory roles       :", dict(roles))
    print("last history role   :", dict(last_role))
    print("action names in hist:", dict(action_names_in_hist.most_common()))

    # ---- KEY SIGNAL: last assistant action -> next action ----
    print("\n=== last assistant_action in history  ->  target (top transitions) ===")
    trans = defaultdict(Counter)
    last_act_none = 0
    for r in train:
        h = r.get("history", []) or []
        last_act = None
        for t in reversed(h):
            if t.get("role") == "assistant_action":
                last_act = t.get("name")
                break
        if last_act is None:
            last_act_none += 1
        tgt = lab_map[r["id"]]
        trans[last_act][tgt] += 1
    print(f"samples w/o prior action: {last_act_none}")
    # For a few common last actions show the next-action distribution
    for la in ["run_tests", "apply_patch", "edit_file", "read_file", "grep_search", None]:
        if la in trans:
            c = trans[la]
            tot = sum(c.values())
            top = c.most_common(4)
            s = ", ".join(f"{k}:{v/tot*100:.0f}%" for k, v in top)
            print(f"  last={str(la):12s} (n={tot:5d}) -> {s}")

    # naive accuracy of "predict most common next action given last action"
    correct = 0
    for r in train:
        h = r.get("history", []) or []
        last_act = None
        for t in reversed(h):
            if t.get("role") == "assistant_action":
                last_act = t.get("name")
                break
        pred = trans[last_act].most_common(1)[0][0]
        if pred == lab_map[r["id"]]:
            correct += 1
    print(f"\nnaive last-action->argmax train acc: {correct/len(train)*100:.2f}%")

    # ---- current_prompt text length ----
    plen = np.array([len(r.get("current_prompt", "") or "") for r in train])
    print("\n=== current_prompt char length ===")
    print(f"  min {plen.min()} median {np.median(plen):.0f} mean {plen.mean():.1f} "
          f"max {plen.max()} p95 {np.percentile(plen,95):.0f} p99 {np.percentile(plen,99):.0f}")
    n_empty_prompt = (plen == 0).sum()
    print(f"  empty current_prompt: {n_empty_prompt}")

    # ---- args keys per action (to design serialization) ----
    print("\n=== assistant_action args keys by action name (sample) ===")
    arg_keys = defaultdict(Counter)
    result_examples = {}
    for r in train:
        for t in (r.get("history", []) or []):
            if t.get("role") == "assistant_action":
                nm = t.get("name")
                for k in (t.get("args", {}) or {}):
                    arg_keys[nm][k] += 1
                if nm not in result_examples:
                    result_examples[nm] = t.get("result_summary", "")
    for nm in sorted(arg_keys):
        print(f"  {nm:18s} args={dict(arg_keys[nm].most_common(5))}")
    print("\n=== result_summary examples by action ===")
    for nm in sorted(result_examples):
        print(f"  {nm:18s} :: {str(result_examples[nm])[:80]}")

    # ---- language check: show a couple ko samples decoded ----
    print("\n=== sample ko prompts (decoded) ===")
    shown = 0
    for r in train:
        if r["session_meta"].get("language_pref") == "ko":
            print(f"  [{lab_map[r['id']]}] {r.get('current_prompt','')[:90]}")
            shown += 1
            if shown >= 5:
                break

    print("\nEDA done.", flush=True)


if __name__ == "__main__":
    main()
