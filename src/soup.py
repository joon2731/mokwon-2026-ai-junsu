"""Greedy model soup (Wortsman et al. ICML'22) — fold 모델 가중치 평균.

프로토콜 (선택 편향 방지):
  soup 후보 = fold 0..3 모델 (heldout fold 4는 soup에서 제외)
  선택 셋   = fold 4의 val 데이터 — 후보 모델 전원이 학습에서 본 적 없음 → 공정
  절차     = 개별 fold4-val 점수로 정렬 → 최고부터 greedy 추가(점수 안 떨어질 때만)

실행: python src\\soup.py --exp E002 --heldout-fold 4
산출: artifacts/{exp}/soup/ (save_pretrained, fp16) + soup_report.json
"""
import argparse
import json

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from data import ARTIFACTS_DIR, load_folds, load_train
from evaluate import fast_macro_f1
from serialize import serialize
from train_hf import EncodedDataset, predict


def avg_state_dicts(dicts):
    out = {}
    for k in dicts[0]:
        out[k] = torch.stack([d[k].float() for d in dicts]).mean(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--heldout-fold", type=int, default=4)
    ap.add_argument("--mode", default="now_first")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    exp_dir = ARTIFACTS_DIR / args.exp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ho = args.heldout_fold

    samples = load_train()
    fold_of = load_folds()
    classes = sorted({s["action"] for s in samples})
    cls_idx = {c: i for i, c in enumerate(classes)}
    val = [s for s in samples if fold_of[s["id"]] == ho]
    y_val = np.array([cls_idx[s["action"]] for s in val])
    texts = [serialize(s, args.mode) for s in val]
    print(f"heldout fold {ho}: n={len(val)}")

    cand_folds = [f for f in sorted({v for v in fold_of.values()}) if f != ho]
    tok = AutoTokenizer.from_pretrained(exp_dir / f"fold{cand_folds[0]}")
    enc = tok(texts, truncation=True, max_length=args.max_len)
    ds = EncodedDataset(
        [{"input_ids": enc["input_ids"][j], "attention_mask": enc["attention_mask"][j]}
         for j in range(len(val))])
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    def score_state(sd):
        model = AutoModelForSequenceClassification.from_pretrained(
            exp_dir / f"fold{cand_folds[0]}")  # 아키텍처 틀
        model.load_state_dict(sd)
        model.to(device)
        probs = predict(model, ds, collator, device, args.batch, torch.bfloat16)
        del model
        torch.cuda.empty_cache()
        return fast_macro_f1(np.argmax(probs, axis=1), y_val, len(classes)), probs

    # 개별 점수
    states, singles = {}, {}
    for f in cand_folds:
        states[f] = load_file(exp_dir / f"fold{f}" / "model.safetensors")
        s, _ = score_state(states[f])
        singles[f] = s
        print(f"fold{f} alone on heldout: {s:.5f}")

    order = sorted(cand_folds, key=lambda f: -singles[f])
    soup_members = [order[0]]
    best_score, _ = score_state(states[order[0]])
    print(f"seed: fold{order[0]} ({best_score:.5f})")
    for f in order[1:]:
        cand = avg_state_dicts([states[m] for m in soup_members + [f]])
        s, _ = score_state(cand)
        take = s >= best_score - 1e-6
        print(f"try +fold{f}: {s:.5f} -> {'ADD' if take else 'skip'}")
        if take:
            soup_members.append(f)
            best_score = s

    final_sd = avg_state_dicts([states[m] for m in soup_members])
    out = exp_dir / "soup"
    model = AutoModelForSequenceClassification.from_pretrained(exp_dir / f"fold{cand_folds[0]}")
    model.load_state_dict(final_sd)
    model.half().save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    report = {
        "heldout_fold": ho,
        "singles": {f"fold{f}": round(s, 5) for f, s in singles.items()},
        "soup_members": soup_members,
        "soup_score_heldout": round(float(best_score), 5),
    }
    (exp_dir / "soup_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"soup = folds {soup_members}, heldout macro_f1 = {best_score:.5f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
