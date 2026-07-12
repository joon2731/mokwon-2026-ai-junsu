"""트랜스포머 fine-tuning (수동 학습 루프, bf16 autocast).

실행 (리포지토리 루트에서):
  python src\\train_hf.py --model microsoft/mdeberta-v3-base --exp E002 ^
      --mode now_first --max-len 512 --epochs 2 --lr 2e-5 --batch 16 --grad-accum 2

스모크: --limit 2000 --folds 0 --epochs 1
산출물: artifacts/{exp}/oof_probs.npy (train 순서), fold{f}/ (fp16 safetensors), report.json

주의:
- DeBERTa 계열은 fp16 학습이 불안정(overflow) → bf16 autocast 사용 (RTX 4070 Ti 지원).
- OOF는 학습에 쓴 fold 배정(artifacts/splits.csv) 그대로.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from data import ARTIFACTS_DIR, load_folds, load_train
from evaluate import cv_tuned_macro_f1, fast_macro_f1, per_class_f1, tune_biases
from serialize import serialize


class EncodedDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.enc = encodings          # list of dicts (input_ids, attention_mask)
        self.labels = labels

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        item = dict(self.enc[i])
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(y, n_classes, scheme):
    if scheme == "none":
        return None
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    if scheme == "balanced":
        w = len(y) / (n_classes * counts)
    elif scheme == "sqrt":
        w = np.sqrt(len(y) / (n_classes * counts))
    else:
        raise ValueError(scheme)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.inference_mode()
def predict(model, ds, collator, device, batch_size, autocast_dtype):
    """길이순 정렬 배치로 추론 후 원래 순서로 복원한 확률 반환."""
    order = sorted(range(len(ds)), key=lambda i: len(ds.enc[i]["input_ids"]))
    loader = DataLoader(
        [ds[i] for i in order], batch_size=batch_size, shuffle=False,
        collate_fn=collator)
    model.eval()
    chunks = []
    for batch in loader:
        labels = batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            logits = model(**batch).logits
        chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    probs_sorted = np.concatenate(chunks)
    probs = np.empty_like(probs_sorted)
    probs[order] = probs_sorted
    return probs


def train_one_fold(args, fold, texts, y, folds, classes, device):
    set_seed(42 + fold)
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"num_labels": len(classes)}
    if args.attn:
        kw["attn_implementation"] = args.attn
    model = AutoModelForSequenceClassification.from_pretrained(args.model, **kw)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.to(device)

    tr_idx = np.where(folds != fold)[0]
    va_idx = np.where(folds == fold)[0]

    def encode(idx):
        enc = tok([texts[i] for i in idx], truncation=True, max_length=args.max_len)
        return [
            {"input_ids": enc["input_ids"][j], "attention_mask": enc["attention_mask"][j]}
            for j in range(len(idx))
        ]

    ds_tr = EncodedDataset(encode(tr_idx), y[tr_idx].tolist())
    ds_va = EncodedDataset(encode(va_idx), y[va_idx].tolist())
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
    loader = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                        collate_fn=collator, drop_last=True)

    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * 0.06), total_steps)
    w = class_weights(y[tr_idx], len(classes), args.class_weight)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=w.to(device) if w is not None else None)
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32

    model.train()
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels) / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1
                if step % 100 == 0:
                    print(f"  fold {fold} epoch {epoch} step {step}/{total_steps} "
                          f"loss={loss.item() * args.grad_accum:.4f} "
                          f"({step / (time.time() - t0):.2f} step/s, "
                          f"{time.time() - t0:.0f}s)", flush=True)

    probs = predict(model, ds_va, collator, device, args.batch * 4, autocast_dtype)
    score = fast_macro_f1(np.argmax(probs, axis=1), y[va_idx], len(classes))
    print(f"  fold {fold}: macro_f1={score:.5f} ({time.time() - t0:.0f}s)", flush=True)

    if args.save_models:
        out = ARTIFACTS_DIR / args.exp / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        model.half().save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)

    del model
    torch.cuda.empty_cache()
    return va_idx, probs, float(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--mode", default="now_first")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--class-weight", choices=["none", "balanced", "sqrt"],
                    default="sqrt")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--limit", type=int, default=0, help="스모크용 샘플 제한")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-models", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--attn", default=None, help="예: sdpa (xlm-r/bert 계열)")
    ap.add_argument("--grad-ckpt", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.exp}] model={args.model} mode={args.mode} device={device}",
          flush=True)

    samples = load_train()
    fold_of = load_folds()
    if args.limit:
        samples = samples[: args.limit]
    ids = [s["id"] for s in samples]
    folds = np.array([fold_of[i] for i in ids])
    classes = sorted({s["action"] for s in samples})
    cls_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_idx[s["action"]] for s in samples])
    texts = [serialize(s, args.mode) for s in samples]
    print(f"[{args.exp}] n={len(ids)} classes={len(classes)}", flush=True)

    run_folds = [int(f) for f in args.folds.split(",")]
    oof = np.zeros((len(ids), len(classes)), dtype=np.float32)
    covered = np.zeros(len(ids), dtype=bool)
    per_fold = []
    for f in run_folds:
        va_idx, probs, score = train_one_fold(
            args, f, texts, y, folds, classes, device)
        oof[va_idx] = probs
        covered[va_idx] = True
        per_fold.append(score)

    out_dir = ARTIFACTS_DIR / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof_probs.npy", oof)
    (out_dir / "classes.json").write_text(json.dumps(classes), encoding="utf-8")

    mean, std = float(np.mean(per_fold)), float(np.std(per_fold))
    print(f"[{args.exp}] CV macro_f1 = {mean:.5f} +- {std:.5f}", flush=True)

    report = {
        "exp": args.exp,
        "method": f"hf:{args.model}",
        "mode": args.mode,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch * args.grad_accum,
        "class_weight": args.class_weight,
        "folds": run_folds,
        "per_fold": [round(s, 5) for s in per_fold],
        "cv_mean": round(mean, 5),
        "cv_std": round(std, 5),
    }
    # 전 fold를 돌린 경우에만 threshold 튜닝 통계 추가
    if covered.all() and len(run_folds) == len(set(folds.tolist())):
        tuned_pairs = cv_tuned_macro_f1(oof, y, folds)
        tuned = [a for _, a in tuned_pairs]
        bias_full, insample = tune_biases(oof, y)
        report["foldout_tuned"] = [round(s, 5) for s in tuned]
        report["foldout_tuned_mean"] = round(float(np.mean(tuned)), 5)
        report["bias_full_oof"] = [round(float(b), 3) for b in bias_full]
        report["per_class_f1"] = per_class_f1(np.argmax(oof, axis=1), y, classes)
        print(f"[{args.exp}] fold-out tuned = {np.mean(tuned):.5f} "
              f"(in-sample {insample:.5f})", flush=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_dir}\\report.json", flush=True)


if __name__ == "__main__":
    main()
