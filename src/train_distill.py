# -*- coding: utf-8 -*-
"""교사 소프트 로짓 증류로 0.6B 학생 full-data 학습.

손실 (딥리서치 ① 반영): L = α·CE_sqrt + (1−α)·T²·KL(student/T || teacher/T),  T=3, α=0.6
학생 레시피는 검증된 full-data 레시피 그대로 (lr 2e-5, 3ep cosine, bs8 ga4, bf16, grad_ckpt).

실행: python src\\train_distill.py --tag qwen3_distill --grad_ckpt
전제: da2/artifacts/teacher_logits.npz (gen_teacher_logits.py 산출)
저장: artifacts/models/{tag}_full_best (기존 프루닝·패키징 경로 재사용 목적)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

DACON_ART = r"C:\Users\joon2\Desktop\da2\artifacts"
DA2_ART = r"C:\Users\joon2\Desktop\da2\artifacts"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=r"C:\Users\joon2\Desktop\da2\pretrained\Qwen3-0.6B-Base")
    p.add_argument("--teacher_npz", default=os.path.join(DA2_ART, "teacher_logits.npz"))
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.6, help="CE 비중 (KL은 1-α)")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--weighting", default="sqrt", choices=["none", "balanced", "sqrt"])
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--optim", default="adamw_bnb_8bit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="qwen3_distill")
    return p.parse_args()


class KDCollator:
    """DataCollatorWithPadding은 비-토크나이저 키(teacher_logits)를 버림 → 수동 처리."""

    def __init__(self, tok):
        self.inner = DataCollatorWithPadding(tok)

    def __call__(self, features):
        t = torch.tensor([f.pop("teacher_logits") for f in features], dtype=torch.float32)
        batch = self.inner(features)
        batch["teacher_logits"] = t
        return batch


class DistillTrainer(Trainer):
    def __init__(self, class_weights=None, temperature=3.0, alpha=0.6, **kw):
        super().__init__(**kw)
        self.model_accepts_loss_kwargs = False
        self.class_weights = class_weights
        self.T = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        t_logits = inputs.pop("teacher_logits")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        w = self.class_weights.to(logits.device, logits.dtype) if self.class_weights is not None else None
        ce = torch.nn.functional.cross_entropy(logits, labels, weight=w)
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(logits / self.T, dim=-1),
            torch.softmax(t_logits.float() / self.T, dim=-1),
            reduction="batchmean") * (self.T ** 2)
        loss = self.alpha * ce + (1 - self.alpha) * kl
        return (loss, outputs) if return_outputs else loss


def main():
    args = get_args()
    set_seed(args.seed)

    df = pd.read_parquet(os.path.join(DACON_ART, "train_prepared.parquet"))
    classes = json.load(open(os.path.join(DACON_ART, "classes.json"), encoding="utf-8"))
    n_cls = len(classes)

    z = np.load(args.teacher_npz, allow_pickle=True)
    t_map = {i: k for k, i in enumerate(z["ids"].tolist())}
    t_logits = z["logits"][[t_map[i] for i in df.id]]
    t_agree = (t_logits.argmax(1) == df.y.values).mean()
    print(f"DISTILL full-data: n={len(df)} T={args.temperature} alpha={args.alpha} "
          f"teacher-agree={t_agree:.4f}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(list(df.text), truncation=True, max_length=args.max_len)

    class DS(torch.utils.data.Dataset):
        def __init__(self):
            self.y = df.y.values
        def __len__(self):
            return len(self.y)
        def __getitem__(self, i):
            d = {k: enc[k][i] for k in enc}
            d["labels"] = int(self.y[i])
            d["teacher_logits"] = t_logits[i].tolist()
            return d

    counts = np.bincount(df.y.values, minlength=n_cls).astype(np.float64)
    if args.weighting == "none":
        cw = None
    else:
        inv = counts.sum() / (n_cls * counts)
        if args.weighting == "sqrt":
            inv = np.sqrt(inv)
        cw = torch.tensor(inv / inv.mean(), dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_cls,
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)})
    model = model.float()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
        model.config.use_cache = False

    run_dir = os.path.join(DACON_ART, "models", f"{args.tag}_run")
    targs = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=args.grad_ckpt,
        optim=args.optim,
        max_grad_norm=1.0,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
        label_names=["labels"],
        remove_unused_columns=False,  # teacher_logits 키가 모델 시그니처에 없어 제거되는 것 방지
    )
    trainer = DistillTrainer(
        class_weights=cw, temperature=args.temperature, alpha=args.alpha,
        model=model, args=targs, train_dataset=DS(),
        data_collator=KDCollator(tok), processing_class=tok)
    trainer.train()

    save_dir = os.path.join(DACON_ART, "models", f"{args.tag}_full_best")
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)
    print("saved ->", save_dir, flush=True)


if __name__ == "__main__":
    main()
