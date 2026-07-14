# -*- coding: utf-8 -*-
"""Cheap validation: does [transformer probs + structured features] via a GBM
beat the raw transformer argmax? Tested with internal session-grouped CV on the
fold-0 OOF (clean out-of-fold probs)."""
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

DATA = r"C:\Users\joon2\Desktop\da2\open\data"
ART = r"C:\Users\joon2\Desktop\da2\artifacts"

classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
cls2i = {c: i for i, c in enumerate(classes)}
ACTS = classes + ["NONE"]
act2i = {a: i for i, a in enumerate(ACTS)}
TIERS = {"free": 0, "pro": 1, "enterprise": 2}
LANGS = {"en": 0, "ko": 1, "mixed": 2}
CIS = {"none": 0, "passed": 1, "failed": 2}
ERR = re.compile(r"error|assertion|typeerror|nonetype|traceback|막혔|안 ?되|에러|stuck|fail", re.I)
LISTHINT = re.compile(r"목록|폴더|디렉|structure|뭐.?있|어떤.?파일|ls\b|list", re.I)
GREPHINT = re.compile(r"찾아|어디|검색|grep|references|참조|사용.?되|쓰이", re.I)
READHINT = re.compile(r"열어|봐줘|읽어|까보|보고|열어봐|내용|read|open", re.I)
GLOBHINT = re.compile(r"전부|모든|모두|all |\*\.|확장자|files?\b|하나하나", re.I)


def load_train():
    rows = []
    with open(os.path.join(DATA, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return {r["id"]: r for r in rows}


def actions_in_hist(r):
    return [t.get("name") for t in (r.get("history") or []) if t.get("role") == "assistant_action"]


def feats(r):
    m = r.get("session_meta", {}) or {}
    ws = m.get("workspace", {}) or {}
    hist_acts = actions_in_hist(r)
    last = hist_acts[-1] if hist_acts else "NONE"
    last2 = hist_acts[-2] if len(hist_acts) >= 2 else "NONE"
    prompt = r.get("current_prompt", "") or ""
    f = {
        "last_act": act2i.get(last, len(ACTS)),
        "last2_act": act2i.get(last2, len(ACTS)),
        "turn": m.get("turn_index", 0),
        "n_open": len(ws.get("open_files") or []),
        "hist_len": len(r.get("history") or []),
        "budget": np.log1p(m.get("budget_tokens_remaining", 0)),
        "elapsed": m.get("elapsed_session_sec", 0),
        "loc": np.log1p(ws.get("loc", 0)),
        "tier": TIERS.get(m.get("user_tier"), -1),
        "lang": LANGS.get(m.get("language_pref"), -1),
        "ci": CIS.get(ws.get("last_ci_status"), -1),
        "dirty": int(bool(ws.get("git_dirty"))),
        "plen": len(prompt),
        "has_err": int(bool(ERR.search(prompt))),
        "hint_list": int(bool(LISTHINT.search(prompt))),
        "hint_grep": int(bool(GREPHINT.search(prompt))),
        "hint_read": int(bool(READHINT.search(prompt))),
        "hint_glob": int(bool(GLOBHINT.search(prompt))),
    }
    # history action counts
    cnt = {a: 0 for a in classes}
    for a in hist_acts:
        if a in cnt:
            cnt[a] += 1
    for a in classes:
        f[f"cnt_{a}"] = cnt[a]
    return f


def main():
    recs = load_train()
    d = np.load(os.path.join(ART, "oof", "xlmr_len512_fold0.npz"), allow_pickle=True)
    ids, logits, y = d["ids"], d["logits"].astype(np.float64), d["y"]
    e = np.exp(logits - logits.max(1, keepdims=True))
    probs = e / e.sum(1, keepdims=True)

    # structured features + session for grouping
    F = pd.DataFrame([feats(recs[i]) for i in ids])
    sess = np.array([i.rsplit("-step_", 1)[0] for i in ids])

    Xf = F.values.astype(np.float64)
    Xp = probs
    Xboth = np.hstack([probs, Xf])

    base_macro = f1_score(y, probs.argmax(1), average="macro")
    print(f"raw transformer argmax macro-F1        : {base_macro:.4f}")

    def cv(X, name):
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=0)
        oof = np.zeros(len(y), int)
        for tr, va in sgkf.split(X, y, groups=sess):
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, max_depth=6,
                l2_regularization=1.0, random_state=0)
            clf.fit(X[tr], y[tr])
            oof[va] = clf.predict(X[va])
        macro = f1_score(y, oof, average="macro")
        print(f"{name:40s}: {macro:.4f}  ({macro-base_macro:+.4f})")
        return oof

    cv(Xf, "GBM feats only")
    cv(Xp, "GBM probs only")
    oof_both = cv(Xboth, "GBM probs + feats (STACK)")

    print("\nper-class F1  raw -> stack:")
    fr = f1_score(y, probs.argmax(1), average=None, labels=range(len(classes)))
    fs = f1_score(y, oof_both, average=None, labels=range(len(classes)))
    for i, c in enumerate(classes):
        flag = "  <==" if fs[i] - fr[i] > 0.02 else ""
        print(f"  {c:18s} {fr[i]:.3f} -> {fs[i]:.3f}{flag}")


if __name__ == "__main__":
    main()
