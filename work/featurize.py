# -*- coding: utf-8 -*-
"""Shared input serialization for training AND inference.  [SERIALIZATION V2]

script.py (inference) imports build_text from here so the model sees identical
inputs at train and test time. Keep this file dependency-free (stdlib only) so
it can be bundled in submit.zip.

V2 (2026-07-05): adds loc/budget/elapsed to [META] (verified label signal),
top-2 language_mix with shares, raised truncation caps (user 140->200,
result 70->110, args 50->80), non-dict args hardening.
NOTE: models trained on V1 text are INCOMPATIBLE — never package a V1-trained
model with this file; retrain first.
"""

# args keys rendered (in priority order) for each assistant_action
_ARG_KEYS = ("path", "target_symbol", "pattern", "scope", "cmd", "target",
             "goal", "query", "question", "n_files")


def _clip(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n]


def _fmt_action(t):
    name = t.get("name", "") or ""
    args = t.get("args") or {}
    if not isinstance(args, dict):  # hidden-test schema insurance
        args = {}
    parts = []
    for k in _ARG_KEYS:
        if k in args and args[k] not in (None, ""):
            parts.append(f"{k}={_clip(args[k], 80)}")
    argstr = " ".join(parts)
    res = _clip(t.get("result_summary", "") or "", 110)
    head = f"{name}({argstr})" if argstr else name
    return f"{head} -> {res}" if res else head


def build_text(sample, max_hist_turns=6):
    """Serialize one sample into a single string for the transformer.

    Layout (current_prompt first so it is never truncated):
        <current_prompt>
        [META] tier=.. lang=.. ci=.. dirty=.. turn=.. open=..
        [HIST] a: <most-recent action> || u: <..> || a: <..> ...   (newest first)
    """
    prompt = _clip(sample.get("current_prompt", "") or "", 400)

    m = sample.get("session_meta", {}) or {}
    ws = m.get("workspace", {}) or {}
    meta = (
        f"tier={m.get('user_tier', '?')} "
        f"lang={m.get('language_pref', '?')} "
        f"ci={ws.get('last_ci_status', '?')} "
        f"dirty={int(bool(ws.get('git_dirty', False)))} "
        f"turn={m.get('turn_index', '?')}"
    )
    # V2: bucketed always-present numerics (loc carries label signal
    # independent of turn_index; corr(turn, loc) ~= 0.02)
    try:
        meta += f" loc={int(ws.get('loc', 0)) // 1000}k"
        meta += f" budget={int(m.get('budget_tokens_remaining', 0)) // 1000}k"
        meta += f" el={int(m.get('elapsed_session_sec', 0)) // 60}m"
    except (TypeError, ValueError):
        pass
    open_files = ws.get("open_files", []) or []
    if open_files:
        meta += " open=" + ",".join(str(x) for x in open_files[:3])
    lang_mix = ws.get("language_mix", {}) or {}
    if lang_mix:
        # V2: top-2 languages with shares; deterministic tie-break
        top2 = sorted(lang_mix.items(), key=lambda kv: (-kv[1], str(kv[0])))[:2]
        meta += " lm=" + ",".join(f"{k}:{v:.1f}" for k, v in top2)

    hist = sample.get("history", []) or []
    recent = hist[-(max_hist_turns * 2):]
    lines = []
    for t in reversed(recent):  # newest first -> truncation drops oldest
        if t.get("role") == "user":
            lines.append("u: " + _clip(t.get("content", "") or "", 200))
        else:
            lines.append("a: " + _fmt_action(t))
    hist_str = " || ".join(lines)

    return f"{prompt}\n[META] {meta}\n[HIST] {hist_str}"
