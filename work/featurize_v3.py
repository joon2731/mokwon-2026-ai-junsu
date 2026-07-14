# -*- coding: utf-8 -*-
"""Shared input serialization experiment. [SERIALIZATION V3]

V3 keeps the stable V2 metadata but changes placement:
  - explicit section tags
  - last user/action pulled out of history
  - compact history
  - current prompt repeated at the tail for decoder-only classifiers

Do not package this with V2-trained models. Train and inference serialization
must match exactly.
"""

_ARG_KEYS = ("path", "target_symbol", "pattern", "scope", "cmd", "target",
             "goal", "query", "question", "n_files")


def _clip(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n]


def _turn_bucket(turn):
    try:
        t = int(turn)
    except (TypeError, ValueError):
        return "unk"
    if t <= 0:
        return "turn0"
    if t == 1:
        return "turn1"
    if t <= 3:
        return "turn2_3"
    return "turn4p"


def _fmt_action(t, arg_limit=70, res_limit=0):
    name = t.get("name", "") or ""
    args = t.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    parts = []
    for k in _ARG_KEYS:
        if k in args and args[k] not in (None, ""):
            parts.append(f"{k}={_clip(args[k], arg_limit)}")
    argstr = " ".join(parts)
    head = f"{name}({argstr})" if argstr else name
    if res_limit:
        res = _clip(t.get("result_summary", "") or "", res_limit)
        if res:
            return f"{head} -> {res}"
    return head


def _last_indices(hist):
    last_user = None
    last_action = None
    for i in range(len(hist) - 1, -1, -1):
        role = hist[i].get("role")
        if role == "user" and last_user is None:
            last_user = i
        elif role != "user" and last_action is None:
            last_action = i
        if last_user is not None and last_action is not None:
            break
    return last_user, last_action


def build_text(sample, max_hist_turns=6):
    prompt = _clip(sample.get("current_prompt", "") or "", 400)

    m = sample.get("session_meta", {}) or {}
    ws = m.get("workspace", {}) or {}
    turn = m.get("turn_index", "?")
    meta = (
        f"tb={_turn_bucket(turn)} "
        f"tier={m.get('user_tier', '?')} "
        f"lang={m.get('language_pref', '?')} "
        f"ci={ws.get('last_ci_status', '?')} "
        f"dirty={int(bool(ws.get('git_dirty', False)))} "
        f"turn={turn}"
    )
    try:
        meta += f" loc={int(ws.get('loc', 0)) // 1000}k"
        meta += f" budget={int(m.get('budget_tokens_remaining', 0)) // 1000}k"
        meta += f" el={int(m.get('elapsed_session_sec', 0)) // 60}m"
    except (TypeError, ValueError):
        pass

    open_files = ws.get("open_files", []) or []
    if open_files:
        meta += " open=" + ",".join(str(x) for x in open_files[:3])
        bases = []
        exts = []
        for p in open_files[:3]:
            sp = str(p).replace("\\", "/").rsplit("/", 1)[-1]
            bases.append(sp)
            if "." in sp:
                exts.append(sp.rsplit(".", 1)[-1])
        if bases:
            meta += " open_base=" + ",".join(_clip(x, 40) for x in bases)
        if exts:
            meta += " open_ext=" + ",".join(_clip(x, 12) for x in exts)

    lang_mix = ws.get("language_mix", {}) or {}
    if lang_mix:
        top2 = sorted(lang_mix.items(), key=lambda kv: (-kv[1], str(kv[0])))[:2]
        meta += " lm=" + ",".join(f"{k}:{v:.1f}" for k, v in top2)

    hist = sample.get("history", []) or []
    last_user_i, last_action_i = _last_indices(hist)
    if last_user_i is not None:
        last_user = _clip(hist[last_user_i].get("content", "") or "", 220)
    else:
        last_user = "<none>"
    if last_action_i is not None:
        last_action = _fmt_action(hist[last_action_i], arg_limit=80, res_limit=90)
    else:
        last_action = "<none>"

    recent_start = max(0, len(hist) - (max_hist_turns * 2))
    recent = list(enumerate(hist[recent_start:], start=recent_start))
    lines = []
    for idx, t in reversed(recent):
        if idx == last_user_i or idx == last_action_i:
            continue
        if t.get("role") == "user":
            lines.append("u: " + _clip(t.get("content", "") or "", 150))
        else:
            lines.append("a: " + _fmt_action(t, arg_limit=60, res_limit=0))
    hist_str = " || ".join(lines) if lines else "<none>"

    return (
        f"[CUR] {prompt}\n"
        f"[META] {meta}\n"
        f"[LAST_USER] {last_user}\n"
        f"[LAST_ACTION] {last_action}\n"
        f"[HIST] {hist_str}\n"
        f"[NEXT_ACTION_FROM] {prompt}"
    )

