"""샘플 → 모델 입력 문자열 직렬화.

템플릿을 바꾸면 새 mode 이름을 추가한다 (기존 mode는 수정하지 않는다 —
실험 재현성을 위해 mode 이름 = 템플릿 버전).

mode:
  prompt    : current_prompt만 (주최측 베이스라인 파리티)
  full      : [META] + [H1..H6] 히스토리 쌍(과거→최근) + [NOW] (v1, TF-IDF용)
  now_first : [NOW] + [META] + [R1..R6] 히스토리(최근→과거) — 트랜스포머용.
              우측 truncation 시 오래된 히스토리부터 잘리고 [NOW]/[META]는 보존된다.
"""
from pathlib import PurePosixPath


def _clip(s, n):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _meta_block(meta):
    ws = meta.get("workspace") or {}
    mix = ws.get("language_mix") or {}
    mix_s = ",".join(
        f"{k}:{round(v * 100)}"
        for k, v in sorted(mix.items(), key=lambda x: -x[1])[:3]
    ) or "none"
    open_files = ws.get("open_files") or []
    open_s = ",".join(PurePosixPath(p).name for p in open_files[:3]) or "none"
    budget = meta.get("budget_tokens_remaining")
    budget_s = f"{round(budget / 1000)}k" if isinstance(budget, (int, float)) else "na"
    return (
        f"[META] tier={meta.get('user_tier', 'na')}"
        f" lang={meta.get('language_pref', 'na')}"
        f" turn={meta.get('turn_index', 'na')}"
        f" budget={budget_s}"
        f" elapsed={meta.get('elapsed_session_sec', 'na')}"
        f" ci={ws.get('last_ci_status', 'na')}"
        f" dirty={int(bool(ws.get('git_dirty')))}"
        f" loc={ws.get('loc', 'na')}"
        f" mix={mix_s}"
        f" open={open_s}"
    )


def _history_pairs(history):
    """교대 배열 → (user 발화, action 이름, result_summary) 쌍 리스트."""
    pairs = []
    pending_user = None
    for e in history or []:
        role = e.get("role")
        if role == "user":
            pending_user = e.get("content", "")
        elif role == "assistant_action":
            pairs.append((pending_user or "", e.get("name", "?"), e.get("result_summary", "")))
            pending_user = None
    if pending_user is not None:  # 관측상 없지만 방어
        pairs.append((pending_user, "?", ""))
    return pairs


def serialize(sample, mode="full"):
    if mode == "prompt":
        text = sample.get("current_prompt", "")
        return text if isinstance(text, str) else ("" if text is None else str(text))

    meta = sample.get("session_meta") or {}
    pairs = _history_pairs(sample.get("history"))

    if mode == "full":
        lines = [_meta_block(meta)]
        if pairs:
            for i, (u, name, rs) in enumerate(pairs, 1):
                lines.append(f"[H{i}] U: {_clip(u, 200)} => {name} ({_clip(rs, 100)})")
        else:
            lines.append("[H] none")
        lines.append(f"[NOW] {_clip(sample.get('current_prompt'), 400)}")
        return "\n".join(lines)

    if mode == "now_first":
        lines = [f"[NOW] {_clip(sample.get('current_prompt'), 400)}", _meta_block(meta)]
        if pairs:
            for i, (u, name, rs) in enumerate(reversed(pairs), 1):
                lines.append(f"[R{i}] U: {_clip(u, 200)} => {name} ({_clip(rs, 100)})")
        else:
            lines.append("[R] none")
        return "\n".join(lines)

    raise ValueError(f"unknown serialize mode: {mode}")
