# -*- coding: utf-8 -*-
"""[Submission] Inference script for DACON 236694 (AI agent action prediction).

The eval server unzips submit.zip and runs THIS file. It must:
  - read ./data/test.jsonl and ./data/sample_submission.csv
  - load model(s) from ./model/
  - write ./output/submission.csv  (columns: id, action ; sample_submission order)

Runs offline on a single T4 (16GB). Uses only torch + transformers + numpy + stdlib,
which are pre-installed in the eval env (transformers 4.46.3, torch 2.7.1).

Ensemble & macro-F1 post-processing are driven by ./model/infer_config.json:
  {
    "model_dirs": ["m0", "m1", ...],   # subdirs under ./model containing HF models
    "max_len": 256,
    "class_bias": [ ... 14 floats ... ] # optional additive logit bias per class
  }
If infer_config.json is absent, every subdir of ./model with a config.json is used
with plain mean-softmax argmax.

[외부 요소 출처 및 활용 범위 / External components: source, license, usage]
  - 사전학습 모델: Qwen/Qwen3-0.6B-Base (HuggingFace Hub, Apache-2.0 License)
      -> 대회 제공 train.jsonl만으로 14-클래스 분류기로 파인튜닝하여 ./model/에 동봉
      (어휘 임베딩 프루닝으로 용량 축소, vocab_remap.npy로 추론시 재매핑 — 예측 동일)
  - 런타임 라이브러리: transformers 4.51.3 + tokenizers 0.21.0 + huggingface_hub 0.30.0
      (모두 Apache-2.0). 제출 형태에 따라 ./libs/에 오프라인 동봉하거나 requirements.txt
      설치 단계에서 설치함 — 평가서버 기본 transformers 4.46.3이 Qwen3 아키텍처 미지원이라 필요.
      torch(BSD-3)·numpy(BSD-3)는 서버 사전설치분 사용
  - 외부 데이터: 사용하지 않음 (대회 제공 데이터만 사용)

[개발(학습) 환경 / Development environment]
  - OS: Windows 11 Home (10.0.26200) · GPU: NVIDIA RTX 4070 Ti 12GB (CUDA 12.4)
  - Python 3.13.12 · torch 2.6.0+cu124 · transformers 5.13.0 · datasets 5.0.0
  - numpy 2.5.0 · scikit-learn 1.9.0 · bitsandbytes 0.49.2
  - 추론(본 스크립트)은 평가 서버 사양(Ubuntu 22.04, T4 16GB, Python 3.11.15,
    torch 2.7.1)에서 동작하도록 작성 (transformers는 ./libs 동봉 또는 requirements 설치 4.51.3 사용)
"""
import csv
import json
import os
import sys

# stay fully offline on the eval server; import bundled featurize.py robustly
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# optional bundled transformers>=4.51 for models the server's 4.46.3 can't load (e.g. qwen3).
# ./libs holds unzipped wheels (transformers, tokenizers, huggingface_hub) and, if present,
# is put first on sys.path so it shadows the pre-installed transformers. Absent -> server's.
_libs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
if os.path.isdir(_libs):
    sys.path.insert(0, _libs)

import numpy as np
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
print(f"[infer] transformers {transformers.__version__} (libs bundle: {os.path.isdir(_libs)})", flush=True)

# featurize.py is bundled next to this script -> identical serialization as training
from featurize import build_text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data") if os.path.isdir(os.path.join(HERE, "data")) else "./data"
MODEL_DIR = os.path.join(HERE, "model") if os.path.isdir(os.path.join(HERE, "model")) else "./model"
OUT_PATH = os.path.join(HERE, "output", "submission.csv")  # created via makedirs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"id": "", "current_prompt": "", "history": [],
                                 "session_meta": {}})
    return rows


def read_config():
    cfg_path = os.path.join(MODEL_DIR, "infer_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    if "model_dirs" not in cfg or not cfg["model_dirs"]:
        cfg["model_dirs"] = sorted(
            d for d in os.listdir(MODEL_DIR)
            if os.path.isdir(os.path.join(MODEL_DIR, d))
            and os.path.exists(os.path.join(MODEL_DIR, d, "config.json")))
    cfg.setdefault("max_len", 256)
    return cfg


@torch.no_grad()
def predict_probs(model_dir, texts, max_len, batch_size=128):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE).eval()
    if DEVICE == "cuda":
        model.half()
    id2label = model.config.id2label
    n = len(texts)

    # optional vocab remap (embedding-pruned models): tokenizer stays original,
    # ids are remapped to the pruned embedding rows right after tokenization
    remap_path = os.path.join(model_dir, "vocab_remap.npy")
    remap = np.load(remap_path) if os.path.exists(remap_path) else None

    # Length-sorted batching, LONGEST FIRST, with a fixed token budget per batch:
    #  - batches pad only to their own max (median input ~270 tokens vs cap 512)
    #  - descending order allocates the largest GPU blocks first, so every later
    #    batch reuses them (flat memory profile; avoids allocator fragmentation)
    #  - token budget keeps peak activation constant across lengths
    lens = [len(x) for x in tok(list(texts), truncation=True, max_length=max_len)["input_ids"]]
    order = sorted(range(n), key=lambda i: -lens[i])
    budget = batch_size * max_len  # e.g. 128*512 = 65k tokens per batch

    probs = None
    s = 0
    while s < n:
        bl = lens[order[s]]  # batch max length (descending -> first element)
        bs = max(8, min(batch_size, budget // max(bl, 1)))
        idx = order[s:s + bs]
        s += len(idx)
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        if remap is not None:
            enc["input_ids"] = torch.tensor(remap[enc["input_ids"].numpy()], dtype=torch.long)
        enc = enc.to(DEVICE)
        logits = model(**enc).logits.float()
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        if probs is None:
            probs = np.zeros((n, p.shape[1]), dtype=np.float64)
        probs[idx] = p
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return probs, id2label


def main():
    cfg = read_config()
    test_path = os.path.join(DATA_DIR, "test.jsonl")
    sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

    samples = load_jsonl(test_path)
    ids = [s.get("id", "") for s in samples]
    texts = [build_text(s) for s in samples]
    print(f"[infer] {len(texts)} samples | device={DEVICE} | models={cfg['model_dirs']}", flush=True)

    # ensemble: weighted mean of softmax probabilities across models
    # (weights from infer_config "model_weights"; default equal)
    weights = cfg.get("model_weights") or [1.0] * len(cfg["model_dirs"])
    if len(weights) != len(cfg["model_dirs"]):
        raise ValueError("model_weights len != model_dirs len")
    wsum = float(sum(weights))
    mean_probs = None
    label_order = None
    for md, w in zip(cfg["model_dirs"], weights):
        mdir = os.path.join(MODEL_DIR, md)
        probs, id2label = predict_probs(mdir, texts, cfg["max_len"])
        # align to a canonical label order (from first model)
        labels = [id2label[i] for i in range(len(id2label))]
        if label_order is None:
            label_order = labels
        elif labels != label_order:
            reorder = [labels.index(l) for l in label_order]
            probs = probs[:, reorder]
        probs = probs * (w / wsum)
        mean_probs = probs if mean_probs is None else mean_probs + probs
        print(f"[infer] done {md} (w={w})", flush=True)

    # optional macro-F1 post-processing: additive per-class bias on log-probs
    scores = np.log(mean_probs + 1e-9)
    if cfg.get("classes") and list(cfg["classes"]) != list(label_order):
        raise ValueError(f"label order mismatch: config={cfg['classes']} model={label_order}")
    if cfg.get("class_bias"):
        bias = np.array(cfg["class_bias"], dtype=np.float64)
        if len(bias) != len(label_order):
            raise ValueError(f"class_bias len {len(bias)} != n classes {len(label_order)}")
        scores = scores + bias[None, :]

    # optional au-subgroup prior correction, turn-conditioned (absent -> no-op)
    au_path = os.path.join(MODEL_DIR, "au_bias.json")
    if os.path.exists(au_path):
        with open(au_path, encoding="utf-8") as f:
            ab = json.load(f)
        if ab.get("classes") and list(ab["classes"]) != list(label_order):
            raise ValueError("au_bias label order mismatch")
        b_low = np.array(ab["au_bias_low"], dtype=np.float64)
        b_high = np.array(ab["au_bias_high"], dtype=np.float64)
        n_low = n_high = 0
        for i, (sid, s) in enumerate(zip(ids, samples)):
            if not str(sid).startswith("sess_au_"):
                continue
            turn = (s.get("session_meta") or {}).get("turn_index", 99)
            if isinstance(turn, int) and turn <= 1:
                scores[i] += b_low
                n_low += 1
            else:
                scores[i] += b_high
                n_high += 1
        print(f"[infer] au rows: {n_low + n_high}/{len(ids)} (low-turn {n_low}, high {n_high})", flush=True)
    pred_idx = scores.argmax(axis=1)
    preds = [label_order[i] for i in pred_idx]

    # optional train-side overlay: model/overlay_lookup.json = {session: {step: action}}
    # (absent -> this block is a no-op; predictions identical to the plain zip)
    ov_path = os.path.join(MODEL_DIR, "overlay_lookup.json")
    if os.path.exists(ov_path):
        with open(ov_path, encoding="utf-8") as f:
            ov = json.load(f)
        valid = set(label_order)
        n_hit = 0
        for i, sid in enumerate(ids):
            try:
                sess, st = sid.rsplit("-", 1)
                k = str(int(st.split("_")[1]))
            except (ValueError, IndexError):
                continue
            act = ov.get(sess, {}).get(k)
            if act is not None and act in valid:
                preds[i] = act
                n_hit += 1
        print(f"[infer] overlay hits: {n_hit}/{len(ids)}", flush=True)

    pred_map = dict(zip(ids, preds))

    # write in sample_submission order
    with open(sample_path, newline="", encoding="utf-8") as f:
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
        print(f"[infer] WARNING {n_missing} ids missing prediction", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[infer] saved {OUT_PATH} (rows={len(rows)})", flush=True)


if __name__ == "__main__":
    main()
