# -*- coding: utf-8 -*-
"""여러 XLM-R fold 모델을 일괄 vocab 프루닝 (prune_vocab.py의 검증된 수술을 함수화).

keep-set: train+test 사용 토큰 ∪ specials ∪ 모든 단일문자 piece (히든테스트 안전마진).
각 모델에 대해 [토크나이저 수술 + 임베딩 슬라이스 + fp16 저장 + 동치성 검증] 수행.

Usage: python work\\prune_multi.py xlmr_v2_rdrop_lr4_e4_fold3_best xlmr_v2_rdrop_lr4_e4_fold1_best ...
  -> artifacts/models/<name>_pruned/
"""
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import build_text
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
MAXLEN = 512


def load_jsonl(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_keep(tok):
    texts = [build_text(s) for s in load_jsonl(os.path.join(ROOT, "data", "train.jsonl"))]
    texts += [build_text(s) for s in load_jsonl(os.path.join(ROOT, "data", "test.jsonl"))]
    used = set()
    for i in range(0, len(texts), 2048):
        for idx in tok(texts[i:i + 2048], truncation=True, max_length=MAXLEN)["input_ids"]:
            used.update(idx)
    keep = used | {0, 1, 2, 3, tok.mask_token_id}
    tj = json.load(open(os.path.join(SRC_DIRS[0], "tokenizer.json"), encoding="utf-8"))
    vlist = tj["model"]["vocab"]
    for i, (piece, _score) in enumerate(vlist):
        core = piece[1:] if piece.startswith("▁") else piece
        if len(core) <= 1:
            keep.add(i)
    kept_sorted = sorted(i for i in keep if i < len(vlist))
    print(f"keep-set: used={len(used)} -> kept {len(kept_sorted)} / {len(vlist)}", flush=True)
    return kept_sorted, texts


def prune_one(src, dst, kept_sorted, tj_template):
    old2new = {o: n for n, o in enumerate(kept_sorted)}
    tj = copy.deepcopy(tj_template)
    vlist = tj["model"]["vocab"]
    tj["model"]["vocab"] = [vlist[o] for o in kept_sorted]
    tj["model"]["unk_id"] = old2new[3]
    mask_old = [at["id"] for at in tj.get("added_tokens", []) if at.get("special")]
    for at in tj.get("added_tokens", []):
        at["id"] = old2new.get(at["id"], old2new[3])
    if tj.get("post_processor"):
        for sp in (tj["post_processor"].get("special_tokens") or {}).values():
            if isinstance(sp, dict) and isinstance(sp.get("ids"), list):
                sp["ids"] = [old2new.get(x, x) for x in sp["ids"]]
    os.makedirs(dst, exist_ok=True)
    json.dump(tj, open(os.path.join(dst, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
    for f in ("tokenizer_config.json", "special_tokens_map.json"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            open(os.path.join(dst, f), "w", encoding="utf-8").write(open(p, encoding="utf-8").read())

    model = AutoModelForSequenceClassification.from_pretrained(src)
    emb = model.get_input_embeddings().weight.data
    idx = torch.tensor(kept_sorted, dtype=torch.long)
    new_emb = torch.nn.Embedding(len(kept_sorted), emb.shape[1], padding_idx=old2new[1])
    new_emb.weight.data = emb[idx].clone()
    model.set_input_embeddings(new_emb)
    model.config.vocab_size = len(kept_sorted)
    model.config.pad_token_id = old2new[1]
    model.half()
    model.save_pretrained(dst, safe_serialization=True)
    return old2new


def equivalence(src, dst, texts, n=512):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok1 = AutoTokenizer.from_pretrained(src)
    tok2 = AutoTokenizer.from_pretrained(dst)
    m1 = AutoModelForSequenceClassification.from_pretrained(src).to(dev).eval()
    m2 = AutoModelForSequenceClassification.from_pretrained(dst).to(dev).eval()
    if dev == "cuda":
        m1.half()
        m2.half()
    rng = random.Random(1)
    sample = [texts[i] for i in rng.sample(range(len(texts)), n)]
    agree = 0
    maxdiff = 0.0
    with torch.no_grad():
        for i in range(0, n, 64):
            b = sample[i:i + 64]
            e1 = tok1(b, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
            e2 = tok2(b, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
            assert e1["input_ids"].shape == e2["input_ids"].shape, "tokenization changed!"
            l1 = m1(**e1).logits.float()
            l2 = m2(**e2).logits.float()
            maxdiff = max(maxdiff, (l1 - l2).abs().max().item())
            agree += int((l1.argmax(-1) == l2.argmax(-1)).sum().item())
    del m1, m2
    if dev == "cuda":
        torch.cuda.empty_cache()
    return maxdiff, agree, n


if __name__ == "__main__":
    names = sys.argv[1:]
    assert names, "usage: prune_multi.py <model_dir_name> ..."
    SRC_DIRS = [os.path.join(ART, "models", n) for n in names]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(SRC_DIRS[0])
    kept, texts = compute_keep(tok)
    tj_template = json.load(open(os.path.join(SRC_DIRS[0], "tokenizer.json"), encoding="utf-8"))
    for src in SRC_DIRS:
        dst = src + "_pruned"
        prune_one(src, dst, kept, tj_template)
        md, ag, n = equivalence(src, dst, texts)
        sz = os.path.getsize(os.path.join(dst, "model.safetensors")) / 1e6
        print(f"{os.path.basename(src)} -> pruned {sz:.0f}MB | max|Δlogit|={md:.2e} argmax {ag}/{n}", flush=True)
    print(f"done {time.time()-t0:.0f}s", flush=True)
