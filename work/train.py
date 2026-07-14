# -*- coding: utf-8 -*-
"""Fine-tune a multilingual transformer for 14-class action prediction.

Reusable for single-fold (Phase 1) and full CV (Phase 2). Saves best model and
out-of-fold logits for later threshold tuning / ensembling.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

ART = r"C:\Users\joon2\Desktop\da2\artifacts"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="xlm-roberta-base")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--weighting", default="sqrt", choices=["none", "balanced", "sqrt"])
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (saves VRAM)")
    p.add_argument("--rdrop", type=float, default=0.0,
                   help="R-Drop alpha (0=off). Two dropout-sampled forwards + symmetric KL; ~2x train time")
    p.add_argument("--optim", default="adamw_torch",
                   help="e.g. adamw_bnb_8bit for xlm-r-large on 12GB (needs bitsandbytes)")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default=os.path.join(ART, "models"))
    p.add_argument("--tag", default="mdeberta")
    p.add_argument("--data_path", default=os.path.join(ART, "train_prepared.parquet"),
                   help="prepared parquet path; default keeps the original V2 training data")
    p.add_argument("--classes_path", default=os.path.join(ART, "classes.json"),
                   help="classes.json path matching the prepared data")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="path to a checkpoint dir to resume training from (HF Trainer). "
                        "Args must match the original run for a clean schedule/optimizer resume.")
    return p.parse_args()


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, label_smoothing=0.0, rdrop_alpha=0.0, **kw):
        super().__init__(**kw)
        # transformers 5.x sets model_accepts_loss_kwargs=True for models whose
        # forward has **kwargs; our compute_loss returns a per-microbatch MEAN
        # and ignores num_items_in_batch, so force the classic mean/GA contract
        # (otherwise loss is not divided by grad-accum steps -> ~2x effective LR).
        self.model_accepts_loss_kwargs = False
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.rdrop_alpha = rdrop_alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()  # loss always in fp32 (safe across dtypes)
        w = self.class_weights.to(logits.device, logits.dtype) if self.class_weights is not None else None
        ce = torch.nn.functional.cross_entropy(
            logits, labels, weight=w, label_smoothing=self.label_smoothing)
        if self.rdrop_alpha > 0 and model.training:
            # R-Drop (NeurIPS'21): second dropout-sampled forward + symmetric KL.
            logits2 = model(**inputs).logits.float()
            ce2 = torch.nn.functional.cross_entropy(
                logits2, labels, weight=w, label_smoothing=self.label_smoothing)
            lp1 = torch.nn.functional.log_softmax(logits, dim=-1)
            lp2 = torch.nn.functional.log_softmax(logits2, dim=-1)
            kl = 0.5 * (
                torch.nn.functional.kl_div(lp1, lp2, log_target=True, reduction="batchmean")
                + torch.nn.functional.kl_div(lp2, lp1, log_target=True, reduction="batchmean"))
            loss = 0.5 * (ce + ce2) + self.rdrop_alpha * kl
        else:
            loss = ce
        return (loss, outputs) if return_outputs else loss


def main():
    args = get_args()
    set_seed(args.seed)

    df = pd.read_parquet(args.data_path)
    classes = json.load(open(args.classes_path, encoding="utf-8"))
    n_cls = len(classes)

    tr = df[df.fold != args.fold].reset_index(drop=True)
    va = df[df.fold == args.fold].reset_index(drop=True)
    print(f"fold {args.fold}: train={len(tr)} val={len(va)}  model={args.model}  "
          f"max_len={args.max_len}  weighting={args.weighting}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:  # decoder LMs (e.g. Qwen) ship without a pad token
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

    tr_ds, va_ds = DS(tr_enc, tr.y.values), DS(va_enc, va.y.values)

    # class weights from training fold
    counts = np.bincount(tr.y.values, minlength=n_cls).astype(np.float64)
    if args.weighting == "none":
        cw = None
    else:
        inv = counts.sum() / (n_cls * counts)
        if args.weighting == "sqrt":
            inv = np.sqrt(inv)
        inv = inv / inv.mean()  # normalize to mean 1
        cw = torch.tensor(inv, dtype=torch.float32)
        print("class weights:", {classes[i]: round(float(inv[i]), 2) for i in range(n_cls)})

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_cls,
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)})
    model = model.float()  # force fp32 master weights (some ckpts load as fp16)
    if getattr(model.config, "pad_token_id", None) is None:
        # decoder-as-classifier (Qwen etc.): pooling needs pad id to find last real token
        model.config.pad_token_id = tok.pad_token_id
        model.config.use_cache = False

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        macro = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        return {"macro_f1": macro, "acc": acc}

    run_dir = os.path.join(args.outdir, f"{args.tag}_fold{args.fold}")
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
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
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

    trainer = WeightedTrainer(
        class_weights=cw, label_smoothing=args.label_smoothing,
        rdrop_alpha=args.rdrop,
        model=model, args=targs, train_dataset=tr_ds, eval_dataset=va_ds,
        data_collator=DataCollatorWithPadding(tok), processing_class=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    print("FINAL:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}, flush=True)

    # OOF logits + per-class F1
    pred = trainer.predict(va_ds)
    logits = pred.predictions
    yv = va.y.values
    preds = logits.argmax(1)
    per_class = f1_score(yv, preds, average=None, labels=list(range(n_cls)))
    print("\nper-class F1:")
    for i, c in enumerate(classes):
        print(f"  {c:18s} {per_class[i]:.3f}")
    print(f"\nMACRO-F1 (fold {args.fold}): {f1_score(yv, preds, average='macro'):.4f}")

    os.makedirs(os.path.join(ART, "oof"), exist_ok=True)
    np.savez(os.path.join(ART, "oof", f"{args.tag}_fold{args.fold}.npz"),
             ids=va.id.values, logits=logits, y=yv)

    # save final model cleanly (best already loaded)
    save_dir = os.path.join(args.outdir, f"{args.tag}_fold{args.fold}_best")
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)
    print("saved model ->", save_dir, flush=True)


if __name__ == "__main__":
    main()
