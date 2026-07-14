# -*- coding: utf-8 -*-
"""Fine-tune Qwen3 with an auxiliary confusion-group softmax CE loss.

E107 (da2): CE_total = CE14 + lambda * CE_group, where CE_group re-softmaxes the
logits ONLY within the confusion group the gold label belongs to. Inference cost
is zero (the head is unchanged) — the aux loss only shapes training gradients so
within-group boundaries (e.g. nav: glob/grep/list/read) get sharper.

Confusion groups (docs/02_plan.md):
  nav      = glob_pattern, grep_search, list_directory, read_file
  verify   = lint_or_typecheck, run_bash, run_tests
  dialogue = ask_user, plan_task, respond_only
  modify   = apply_patch, edit_file, write_file
  web      = web_search  (singleton -> CE_group == 0 for those samples)

Base recipe is the validated full-FT recipe; only --group_ce_lambda is new.
Fork of work/train.py (read-only archive) per da2 file-location rule.
Artifacts still land in artifacts/{models,oof} for package_multi compat.
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

ART = r"C:\Users\joon2\Desktop\da2\artifacts"

# class-name -> confusion group (docs/02_plan.md). web_search is its own singleton.
GROUP_MAP = {
    "glob_pattern": "nav", "grep_search": "nav", "list_directory": "nav", "read_file": "nav",
    "lint_or_typecheck": "verify", "run_bash": "verify", "run_tests": "verify",
    "ask_user": "dialogue", "plan_task": "dialogue", "respond_only": "dialogue",
    "apply_patch": "modify", "edit_file": "modify", "write_file": "modify",
    "web_search": "web",
}
GROUP_TO_ID = {"nav": 0, "verify": 1, "dialogue": 2, "modify": 3, "web": 4}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=r"C:\Users\joon2\Desktop\da2\pretrained\Qwen3-0.6B-Base")
    p.add_argument("--fold", type=int, default=0)
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
    p.add_argument("--group_ce_lambda", type=float, default=0.3,
                   help="weight of the within-group aux CE (0=plain CE14)")
    p.add_argument("--optim", default="adamw_bnb_8bit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default=os.path.join(ART, "models"))
    p.add_argument("--tag", default="qwen3_groupce")
    p.add_argument("--data_path", default=os.path.join(ART, "train_prepared.parquet"))
    p.add_argument("--classes_path", default=os.path.join(ART, "classes.json"))
    return p.parse_args()


class GroupCETrainer(Trainer):
    def __init__(self, class_weights=None, group_ids=None, group_ce_lambda=0.0, **kw):
        super().__init__(**kw)
        # keep the classic mean/grad-accum loss contract (see work/train.py note)
        self.model_accepts_loss_kwargs = False
        self.class_weights = class_weights
        self.group_ids = group_ids            # LongTensor [n_cls]
        self.group_ce_lambda = group_ce_lambda

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()  # loss always fp32
        w = self.class_weights.to(logits.device, logits.dtype) if self.class_weights is not None else None
        ce = torch.nn.functional.cross_entropy(logits, labels, weight=w)
        loss = ce
        if self.group_ce_lambda > 0:
            gid = self.group_ids.to(logits.device)
            sample_group = gid[labels]                              # [B]
            mask = gid.unsqueeze(0) == sample_group.unsqueeze(1)    # [B, n_cls] bool
            masked = logits.masked_fill(~mask, float("-inf"))
            logp = torch.log_softmax(masked, dim=-1)
            nll = -logp.gather(1, labels.unsqueeze(1)).squeeze(1)   # [B]; singleton-group -> 0
            if w is not None:
                sw = w[labels]
                ce_group = (nll * sw).sum() / sw.sum().clamp_min(1e-8)
            else:
                ce_group = nll.mean()
            loss = ce + self.group_ce_lambda * ce_group
        return (loss, outputs) if return_outputs else loss


def main():
    args = get_args()
    set_seed(args.seed)

    df = pd.read_parquet(args.data_path)
    classes = json.load(open(args.classes_path, encoding="utf-8"))
    n_cls = len(classes)

    group_ids = torch.tensor(
        [GROUP_TO_ID[GROUP_MAP[c]] for c in classes], dtype=torch.long)
    grp_sizes = {g: sum(1 for c in classes if GROUP_MAP[c] == g) for g in GROUP_TO_ID}
    print(f"group sizes: {grp_sizes}  lambda={args.group_ce_lambda}", flush=True)

    tr = df[df.fold != args.fold].reset_index(drop=True)
    va = df[df.fold == args.fold].reset_index(drop=True)
    print(f"fold {args.fold}: train={len(tr)} val={len(va)}  model={args.model}  "
          f"max_len={args.max_len}  weighting={args.weighting}", flush=True)

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

    tr_ds, va_ds = DS(tr_enc, tr.y.values), DS(va_enc, va.y.values)

    counts = np.bincount(tr.y.values, minlength=n_cls).astype(np.float64)
    if args.weighting == "none":
        cw = None
    else:
        inv = counts.sum() / (n_cls * counts)
        if args.weighting == "sqrt":
            inv = np.sqrt(inv)
        inv = inv / inv.mean()
        cw = torch.tensor(inv, dtype=torch.float32)
        print("class weights:", {classes[i]: round(float(inv[i]), 2) for i in range(n_cls)})

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
        preds = np.argmax(logits, axis=1)
        return {"macro_f1": f1_score(labels, preds, average="macro"),
                "acc": accuracy_score(labels, preds)}

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

    trainer = GroupCETrainer(
        class_weights=cw, group_ids=group_ids, group_ce_lambda=args.group_ce_lambda,
        model=model, args=targs, train_dataset=tr_ds, eval_dataset=va_ds,
        data_collator=DataCollatorWithPadding(tok), processing_class=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("FINAL:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}, flush=True)

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

    save_dir = os.path.join(args.outdir, f"{args.tag}_fold{args.fold}_best")
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)
    print("saved model ->", save_dir, flush=True)


if __name__ == "__main__":
    main()
