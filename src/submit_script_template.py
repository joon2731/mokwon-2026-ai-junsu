"""평가 서버에서 실행되는 추론 스크립트 (제출 zip의 script.py로 복사됨).

self-contained — 서버에는 src/가 없으므로 직렬화 로직을 여기 내장한다.
⚠️ 직렬화는 src/serialize.py의 now_first 모드와 반드시 동일해야 함 (학습-추론 일치).

동작: ./model/fold*/ 에 있는 모든 fold 모델을 로드해 softmax 확률 평균(soft-vote)
→ ./model/bias.json 있으면 log-prob에 가산 → argmax → ./output/submission.csv
"""
import csv
import glob
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

T0 = time.time()
TEST_PATH = "./data/test.jsonl"
SAMPLE_SUB_PATH = "./data/sample_submission.csv"
OUT_PATH = "./output/submission.csv"
MODEL_ROOT = "./model"
MAX_LEN = 512
BATCH = 64


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------- 직렬화 (src/serialize.py now_first와 동일하게 유지) ----------
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
    pairs = []
    pending_user = None
    for e in history or []:
        role = e.get("role")
        if role == "user":
            pending_user = e.get("content", "")
        elif role == "assistant_action":
            pairs.append((pending_user or "", e.get("name", "?"), e.get("result_summary", "")))
            pending_user = None
    if pending_user is not None:
        pairs.append((pending_user, "?", ""))
    return pairs


def serialize_now_first(sample):
    meta = sample.get("session_meta") or {}
    pairs = _history_pairs(sample.get("history"))
    lines = [f"[NOW] {_clip(sample.get('current_prompt'), 400)}", _meta_block(meta)]
    if pairs:
        for i, (u, name, rs) in enumerate(reversed(pairs), 1):
            lines.append(f"[R{i}] U: {_clip(u, 200)} => {name} ({_clip(rs, 100)})")
    else:
        lines.append("[R] none")
    return "\n".join(lines)


# ---------- 데이터 ----------
def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ---------- 추론 ----------
@torch.inference_mode()
def predict_probs(model_dir, texts, device, dtype):
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=dtype)
    model.to(device).eval()

    enc = tok(texts, truncation=True, max_length=MAX_LEN)
    ids_list = enc["input_ids"]
    order = sorted(range(len(texts)), key=lambda i: len(ids_list[i]))
    n_classes = model.config.num_labels
    probs = np.zeros((len(texts), n_classes), dtype=np.float64)

    done = 0
    for s in range(0, len(order), BATCH):
        idx = order[s : s + BATCH]
        batch = tok.pad(
            {"input_ids": [ids_list[i] for i in idx],
             "attention_mask": [enc["attention_mask"][i] for i in idx]},
            return_tensors="pt", pad_to_multiple_of=8)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.float()
        probs[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
        done += len(idx)
        if done % (BATCH * 50) < BATCH:
            log(f"  {done}/{len(texts)} ({done / (time.time() - T0):.1f} samples/s cum)")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return probs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log(f"device={device} dtype={dtype}")

    classes = json.loads(open(os.path.join(MODEL_ROOT, "classes.json"), encoding="utf-8").read())
    bias_path = os.path.join(MODEL_ROOT, "bias.json")
    bias = np.array(json.loads(open(bias_path, encoding="utf-8").read())) if os.path.exists(bias_path) else None

    samples = load_jsonl(TEST_PATH)
    ids = [s.get("id", "") for s in samples]
    texts = [serialize_now_first(s) for s in samples]
    log(f"test samples={len(samples)}")

    fold_dirs = sorted(glob.glob(os.path.join(MODEL_ROOT, "fold*")))
    assert fold_dirs, "no model/fold* directories in zip"
    log(f"fold models: {len(fold_dirs)}")

    probs = np.zeros((len(texts), len(classes)), dtype=np.float64)
    for d in fold_dirs:
        t = time.time()
        probs += predict_probs(d, texts, device, dtype)
        log(f"{d} done in {time.time() - t:.1f}s")
    probs /= len(fold_dirs)

    scores = np.log(np.clip(probs, 1e-9, None))
    if bias is not None:
        assert len(bias) == len(classes)
        scores = scores + bias
        log("applied class bias")
    preds = [classes[i] for i in np.argmax(scores, axis=1)]
    pred_map = dict(zip(ids, preds))

    with open(SAMPLE_SUB_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    n_missing = 0
    for row in rows:
        p = pred_map.get(row["id"])
        if p is None:
            n_missing += 1
        else:
            row["action"] = p
    if n_missing:
        log(f"WARNING: {n_missing} ids without prediction (placeholder kept)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log(f"saved {OUT_PATH} rows={len(rows)}")
    log(f"TOTAL {time.time() - T0:.1f}s, {len(texts) / (time.time() - T0):.1f} samples/s overall")


if __name__ == "__main__":
    main()
