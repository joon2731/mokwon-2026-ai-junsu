# -*- coding: utf-8 -*-
"""Qwen 임베딩 프루닝 (BPE 토크나이저 무수술 방식).

XLM-R(Unigram)과 달리 Qwen은 byte-BPE라 tokenizer.json 수술이 위험하다.
대신: 토크나이저는 원본 그대로 두고,
  1) train+test 직렬화 텍스트에서 사용되는 토큰 id를 채굴
  2) keep = 사용 id ∪ 256개 바이트 토큰 ∪ specials
  3) 임베딩을 keep 행만 슬라이스, vocab_remap.npy(원본id→새id, 미보유→공백토큰행) 동봉
  4) script.py가 토크나이즈 직후 input_ids를 재매핑
검증: 원본 vs 프루닝 로짓 동일성 (미보유 토큰 등장률 포함).

Usage: python work\\prune_qwen.py qwen05_smoke_fold1_best
"""
import io
import argparse
import importlib
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
MAXLEN = 512


def load_jsonl(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def main(name, featurizer="featurize", max_len=MAXLEN):
    build_text = importlib.import_module(featurizer).build_text
    src = os.path.join(ART, "models", name)
    dst = src + "_pruned"
    tok = AutoTokenizer.from_pretrained(src)

    texts = [build_text(s) for s in load_jsonl(os.path.join(ROOT, "data", "train.jsonl"))]
    texts += [build_text(s) for s in load_jsonl(os.path.join(ROOT, "data", "test.jsonl"))]
    used = set()
    for i in range(0, len(texts), 2048):
        for ids in tok(texts[i:i + 2048], truncation=True, max_length=max_len)["input_ids"]:
            used.update(ids)
    print(f"사용 토큰: {len(used)}")

    # byte-level BPE의 기본 바이트 토큰(0..255)과 specials 무조건 보존
    keep = set(used) | set(range(256))
    for t in (tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.unk_token_id):
        if t is not None:
            keep.add(int(t))
    space_id = tok(" ", add_special_tokens=False)["input_ids"]
    space_id = space_id[0] if space_id else tok.eos_token_id
    keep.add(int(space_id))

    model = AutoModelForSequenceClassification.from_pretrained(src)
    V = model.get_input_embeddings().weight.shape[0]
    kept = sorted(i for i in keep if i < V)
    old2new = {o: n for n, o in enumerate(kept)}
    print(f"keep: {len(kept)} / {V}")

    # 미보유 id → 공백 토큰의 '새' 행으로 매핑
    remap = np.full(V, old2new[int(space_id)], dtype=np.int32)
    for o, n in old2new.items():
        remap[o] = n

    emb = model.get_input_embeddings().weight.data
    idx = torch.tensor(kept, dtype=torch.long)
    new_emb = torch.nn.Embedding(len(kept), emb.shape[1])
    new_emb.weight.data = emb[idx].clone()
    model.set_input_embeddings(new_emb)
    model.config.vocab_size = len(kept)
    model.config.pad_token_id = old2new[int(tok.pad_token_id)]
    model.half()
    os.makedirs(dst, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    np.save(os.path.join(dst, "vocab_remap.npy"), remap)

    # rope_theta 복원 (4.46.3 호환 — 저장 직후 바로)
    cfg_p = os.path.join(dst, "config.json")
    cj = json.load(open(cfg_p, encoding="utf-8"))
    rp = cj.get("rope_parameters")
    if rp and "rope_theta" not in cj:
        cj["rope_theta"] = rp.get("rope_theta")
        json.dump(cj, open(cfg_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 토크나이저는 원본 그대로 복사
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "special_tokens_map.json", "added_tokens.json"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            open(os.path.join(dst, f), "wb").write(open(p, "rb").read())

    sz = os.path.getsize(os.path.join(dst, "model.safetensors")) / 1e6
    print(f"저장: {dst}  model={sz:.0f}MB")

    # ---- 동치성 검증 ----
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m1 = AutoModelForSequenceClassification.from_pretrained(src).to(dev).eval().half()
    m2 = AutoModelForSequenceClassification.from_pretrained(dst).to(dev).eval().half()
    rng = random.Random(1)
    sample = [texts[i] for i in rng.sample(range(len(texts)), 512)]
    agree = 0
    maxdiff = 0.0
    unseen_tok = 0
    tot_tok = 0
    with torch.no_grad():
        for i in range(0, len(sample), 32):
            b = sample[i:i + 32]
            e = tok(b, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            ii = e["input_ids"].numpy()
            unseen_tok += int((remap[ii] == old2new[int(space_id)]).sum() - (ii == space_id).sum())
            tot_tok += ii.size
            l1 = m1(input_ids=e["input_ids"].to(dev),
                    attention_mask=e["attention_mask"].to(dev)).logits.float()
            ii2 = torch.tensor(remap[ii], dtype=torch.long).to(dev)
            l2 = m2(input_ids=ii2, attention_mask=e["attention_mask"].to(dev)).logits.float()
            maxdiff = max(maxdiff, (l1 - l2).abs().max().item())
            agree += int((l1.argmax(-1) == l2.argmax(-1)).sum().item())
    print(f"동치성: max|Δlogit|={maxdiff:.2e}  argmax {agree}/512  "
          f"미보유토큰 {unseen_tok}/{tot_tok}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="qwen05_smoke_fold1_best")
    ap.add_argument("--featurizer", default="featurize",
                    help="module name under work/ that provides build_text, e.g. featurize_v3")
    ap.add_argument("--max_len", type=int, default=MAXLEN)
    args = ap.parse_args()
    main(args.name, featurizer=args.featurizer, max_len=args.max_len)
