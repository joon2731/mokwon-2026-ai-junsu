# -*- coding: utf-8 -*-
"""train.py의 full-data(100%, 홀드아웃 없음) 변형 — 최종 제출 모델 학습용.

train.py와의 차이만:
  - fold 홀드아웃 없이 전체 70k로 학습 (fold 인자 없음)
  - eval/베스트 선택 없음 — cosine 스케줄이 3ep에 완전 anneal되는 지점이 peak라는
    실측(PROGRESS 7/8 에폭 실험)에 근거해 '최종 에폭 = 최종 모델'
  - 저장: {tag}_full_best (OOF 없음)
레시피 자체(옵티마이저/스케줄/가중치)는 train.py와 동일하게 유지할 것.
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)
import pandas as pd

ART = r"C:\Users\joon2\Desktop\da2\artifacts"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=r"C:\Users\joon2\Desktop\da2\pretrained\Qwen3-0.6B-Base")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--weighting", default="sqrt", choices=["none", "balanced", "sqrt"])
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--optim", default="adamw_bnb_8bit")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default=os.path.join(ART, "models"))
    p.add_argument("--tag", default="qwen3_full")
    p.add_argument("--data_path", default=os.path.join(ART, "train_prepared.parquet"))
    p.add_argument("--classes_path", default=os.path.join(ART, "classes.json"))
    p.add_argument("--resume_from_checkpoint", default=None)
    return p.parse_args()


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, label_smoothing=0.0, **kw):
        super().__init__(**kw)
        self.model_accepts_loss_kwargs = False  # train.py와 동일한 GA 계약 유지
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        w = self.class_weights.to(logits.device, logits.dtype) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(
            logits, labels, weight=w, label_smoothing=self.label_smoothing)
        return (loss, outputs) if return_outputs else loss


def main():
    args = get_args()
    set_seed(args.seed)

    df = pd.read_parquet(args.data_path)
    classes = json.load(open(args.classes_path, encoding="utf-8"))
    n_cls = len(classes)
    print(f"FULL-DATA train: n={len(df)}  model={args.model}  max_len={args.max_len}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enc = tok(list(df.text), truncation=True, max_length=args.max_len)

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc, y):
            self.enc, self.y = enc, y
        def __len__(self):
            return len(self.y)
        def __getitem__(self, i):
            d = {k: self.enc[k][i] for k in self.enc}
            d["labels"] = int(self.y[i])
            return d

    ds = DS(enc, df.y.values)

    counts = np.bincount(df.y.values, minlength=n_cls).astype(np.float64)
    if args.weighting == "none":
        cw = None
    else:
        inv = counts.sum() / (n_cls * counts)
        if args.weighting == "sqrt":
            inv = np.sqrt(inv)
        inv = inv / inv.mean()
        cw = torch.tensor(inv, dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_cls,
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)})
    model = model.float()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
        model.config.use_cache = False

    run_dir = os.path.join(args.outdir, f"{args.tag}_run")
    targs = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        lr_scheduler_type="cosine",
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        gradient_checkpointing=args.grad_ckpt,
        optim=args.optim,
        max_grad_norm=1.0,
        eval_strategy="no",
        save_strategy="epoch",       # 크래시 대비 에폭 체크포인트 (재개용)
        save_total_limit=1,
        load_best_model_at_end=False,
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        class_weights=cw, label_smoothing=args.label_smoothing,
        model=model, args=targs, train_dataset=ds,
        data_collator=DataCollatorWithPadding(tok), processing_class=tok,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    save_dir = os.path.join(args.outdir, f"{args.tag}_full_best")
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)
    print("saved model ->", save_dir, flush=True)


if __name__ == "__main__":
    main()
