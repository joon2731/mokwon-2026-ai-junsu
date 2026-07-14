# -*- coding: utf-8 -*-
"""1.7B LoRA 교사 → 70k 전체 소프트 로짓 생성 (증류 재료).

실행: python src\\gen_teacher_logits.py --adapter qwen3_17b_lora_fold0_best
산출: da2/artifacts/teacher_logits.npz (ids, logits — train_prepared.parquet 순서)
소요: 4070 Ti에서 ~20분 (1.7B fp16, 길이 내림차순 배칭)

주의(docs/03 R105 예정): 교사가 fold0 홀드아웃 모델이라 fold0 14k는 OOF 로짓,
fold1-4 56k는 in-sample 로짓. 교사 암기 전이는 학생 손실의 CE 앵커(α=0.6)로 억제.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DACON = r"C:\Users\joon2\Desktop\da2"
DA2_ART = r"C:\Users\joon2\Desktop\da2\artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="qwen3_17b_lora_fold0_best")
    ap.add_argument("--base", default=os.path.join(DACON, "pretrained", "Qwen3-1.7B-Base"))
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(DA2_ART, "teacher_logits.npz"))
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(DACON, "artifacts", "train_prepared.parquet"))
    classes = json.load(open(os.path.join(DACON, "artifacts", "classes.json"), encoding="utf-8"))
    adapter_dir = os.path.join(DACON, "artifacts", "models", args.adapter)

    tok = AutoTokenizer.from_pretrained(adapter_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(classes), torch_dtype=torch.bfloat16)
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tok.pad_token_id
        base.config.use_cache = False
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model = model.half().cuda().eval()
    print("teacher merged & loaded", flush=True)

    enc = tok(list(df.text), truncation=True, max_length=args.max_len)
    ids_list = enc["input_ids"]
    order = sorted(range(len(df)), key=lambda i: -len(ids_list[i]))  # 긴 것부터
    logits = np.zeros((len(df), len(classes)), dtype=np.float32)

    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(order), args.bs):
            idx = order[s : s + args.bs]
            batch = tok.pad(
                {"input_ids": [ids_list[i] for i in idx],
                 "attention_mask": [enc["attention_mask"][i] for i in idx]},
                return_tensors="pt", pad_to_multiple_of=8)
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch).logits.float().cpu().numpy()
            logits[idx] = out
            if (s // args.bs) % 50 == 0:
                done = s + len(idx)
                print(f"  {done}/{len(df)} ({done / (time.time() - t0):.0f}/s)", flush=True)

    os.makedirs(DA2_ART, exist_ok=True)
    np.savez(args.out, ids=df.id.values, logits=logits)
    # 빠른 sanity: 교사 로짓의 train 정확도 (in-sample 포함이라 참고용)
    acc = (logits.argmax(1) == df.y.values).mean()
    print(f"saved {args.out}  (train-agree {acc:.4f}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
