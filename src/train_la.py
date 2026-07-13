# -*- coding: utf-8 -*-
"""Balanced softmax (in-training logit-adjusted CE) fold0 게이트 실험.

loss = CE(logits + tau * log(prior), labels)  — Menon et al. ICLR'21 Eq.10, tau=1.
class weight는 끔(plain) — 가중과 로짓 보정의 중복 방지. 비교 기준: sqrt CE fold0 0.7679.
실행: python src\\train_la.py --fold 0 --tag qwen3_bsm --grad_ckpt
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

DACON_ART = r"C:\Users\joon2\Desktop\dacon\artifacts"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=r"C:\Users\joon2\Desktop\dacon\pretrained\Qwen3-0.6B-Base")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--optim", default="adamw_bnb_8bit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="qwen3_bsm")
    return p.parse_args()


class LATrainer(Trainer):
    def __init__(self, log_prior=None, tau=1.0, **kw):
        super().__init__(**kw)
        self.model_accepts_loss_kwargs = False
        self.log_prior = log_prior
        self.tau = tau

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        adj = logits + self.tau * self.log_prior.to(logits.device)
        loss = torch.nn.functional.cross_entropy(adj, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    args = get_args()
    set_seed(args.seed)
    df = pd.read_parquet(os.path.join(DACON_ART, "train_prepared.parquet"))
    classes = json.load(open(os.path.join(DACON_ART, "classes.json"), encoding="utf-8"))
    n_cls = len(classes)
    tr = df[df.fold != args.fold].reset_index(drop=True)
    va = df[df.fold == args.fold].reset_index(drop=True)
    print(f"[BSM] fold {args.fold} tau={args.tau}: train={len(tr)} val={len(va)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(texts):
        return tok(list(texts), truncation=True, max_length=args.max_len)

    tr_enc, va_enc = encode(tr.text), encode(va.text)

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc, y):
            self.enc, self.y = enc, y
        def __len__(self):
            return len(self.y)
        def __getitem__(self, i):
            d = {k: self.enc[k][i] for k in self.enc}
            d["labels"] = int(self.y[i])
            return d

    prior = np.bincount(tr.y.values, minlength=n_cls) / len(tr)
    log_prior = torch.tensor(np.log(np.clip(prior, 1e-9, None)), dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_cls,
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)})
    model = model.float()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
        model.config.use_cache = False

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # 추론 시엔 보정 없이 raw argmax (학습 손실에만 prior 반영)
        preds = np.argmax(logits, axis=1)
        return {"macro_f1": f1_score(labels, preds, average="macro"),
                "acc": accuracy_score(labels, preds)}

    run_dir = os.path.join(DACON_ART, "models", f"{args.tag}_fold{args.fold}")
    targs = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=args.grad_ckpt,
        optim=args.optim,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = LATrainer(
        log_prior=log_prior, tau=args.tau,
        model=model, args=targs, train_dataset=DS(tr_enc, tr.y.values),
        eval_dataset=DS(va_enc, va.y.values),
        data_collator=DataCollatorWithPadding(tok), processing_class=tok,
        compute_metrics=compute_metrics)
    trainer.train()

    pred = trainer.predict(DS(va_enc, va.y.values))
    logits = pred.predictions
    preds = logits.argmax(1)
    print(f"MACRO-F1 (fold {args.fold}): {f1_score(va.y.values, preds, average='macro'):.4f}", flush=True)
    os.makedirs(os.path.join(DACON_ART, "oof"), exist_ok=True)
    np.savez(os.path.join(DACON_ART, "oof", f"{args.tag}_fold{args.fold}.npz"),
             ids=va.id.values, logits=logits, y=va.y.values)
    save_dir = os.path.join(DACON_ART, "models", f"{args.tag}_fold{args.fold}_best")
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)
    print("saved ->", save_dir, flush=True)


if __name__ == "__main__":
    main()
