# -*- coding: utf-8 -*-
"""train.py의 LoRA 변형 — 1GB 제출 제한을 받지 않는 로컬 전용 '교사' 학습용 (Qwen3-1.7B).

train.py와의 차이:
  - base 가중치 동결(bf16 로드) + LoRA(attn/MLP) + score 헤드 학습 → 12GB VRAM에 1.7B 수용
  - 옵티마이저는 LoRA 파라미터만 대상 (adamw_torch, lr 1e-4 기본)
  - 저장: adapter + tokenizer (추론/로짓 생성 시 merge_and_unload)
동일: V2 parquet, fold holdout, 에폭별 eval, OOF npz 저장 (게이트 판정용)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

ART = r"C:\Users\joon2\Desktop\da2\artifacts"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=r"C:\Users\joon2\Desktop\da2\pretrained\Qwen3-1.7B-Base")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--weighting", default="sqrt", choices=["none", "balanced", "sqrt"])
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="qwen3_17b_lora")
    p.add_argument("--data_path", default=os.path.join(ART, "train_prepared.parquet"))
    p.add_argument("--classes_path", default=os.path.join(ART, "classes.json"))
    return p.parse_args()


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kw):
        super().__init__(**kw)
        self.model_accepts_loss_kwargs = False
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        w = self.class_weights.to(logits.device, logits.dtype) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def main():
    args = get_args()
    set_seed(args.seed)

    df = pd.read_parquet(args.data_path)
    classes = json.load(open(args.classes_path, encoding="utf-8"))
    n_cls = len(classes)
    tr = df[df.fold != args.fold].reset_index(drop=True)
    va = df[df.fold == args.fold].reset_index(drop=True)
    print(f"[LoRA] fold {args.fold}: train={len(tr)} val={len(va)} model={args.model}", flush=True)

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
        cw = torch.tensor(inv / inv.mean(), dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_cls, torch_dtype=torch.bfloat16,
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)})
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
        model.config.use_cache = False
    # score 헤드는 base와 같은 bf16 유지 (fp32 강제 시 dtype 불일치 — loss는 compute_loss에서 fp32)

    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"])
    model = get_peft_model(model, lcfg)
    if args.grad_ckpt:
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"macro_f1": f1_score(labels, preds, average="macro"),
                "acc": accuracy_score(labels, preds)}

    run_dir = os.path.join(ART, "models", f"{args.tag}_fold{args.fold}")
    targs = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        weight_decay=args.wd,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=args.grad_ckpt,
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
        label_names=["labels"],
    )
    trainer = WeightedTrainer(
        class_weights=cw, model=model, args=targs,
        train_dataset=tr_ds, eval_dataset=va_ds,
        data_collator=DataCollatorWithPadding(tok), processing_class=tok,
        compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    print("FINAL:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}, flush=True)

    pred = trainer.predict(va_ds)
    logits = pred.predictions
    preds = logits.argmax(1)
    print(f"MACRO-F1 (fold {args.fold}): {f1_score(va.y.values, preds, average='macro'):.4f}", flush=True)
    os.makedirs(os.path.join(ART, "oof"), exist_ok=True)
    np.savez(os.path.join(ART, "oof", f"{args.tag}_fold{args.fold}.npz"),
             ids=va.id.values, logits=logits, y=va.y.values)

    save_dir = os.path.join(ART, "models", f"{args.tag}_fold{args.fold}_best")
    trainer.save_model(save_dir)   # adapter + score
    tok.save_pretrained(save_dir)
    print("saved adapter ->", save_dir, flush=True)


if __name__ == "__main__":
    main()
