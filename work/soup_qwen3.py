# -*- coding: utf-8 -*-
"""Stream-average Qwen3 fold checkpoints into one weight-soup model.

Memory-conscious: averages one tensor at a time from safetensors files, then
writes a single fp16 HF checkpoint under artifacts/models/qwen3_smoke_soup5_best.
"""
import argparse
import json
import os
import shutil

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
TMP = os.path.join(ROOT, "scratch_tmp")
os.makedirs(TMP, exist_ok=True)
os.environ.setdefault("TMP", TMP)
os.environ.setdefault("TEMP", TMP)
os.environ.setdefault("TMPDIR", TMP)

import torch
from safetensors.torch import safe_open, save_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="qwen3_smoke")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--out", default="qwen3_smoke_soup5_best")
    return p.parse_args()


def main():
    args = parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    srcs = [os.path.join(ART, "models", f"{args.tag}_fold{f}_best") for f in folds]
    files = [os.path.join(s, "model.safetensors") for s in srcs]
    for p in files:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    dst = os.path.join(ART, "models", args.out)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    print("[soup] stream average:")
    for s in srcs:
        print("  ", s)

    with safe_open(files[0], framework="pt", device="cpu") as f0:
        keys = list(f0.keys())
    print(f"[soup] tensors={len(keys)} folds={len(files)}", flush=True)

    out = {}
    for idx, key in enumerate(keys, start=1):
        acc = None
        for fp in files:
            with safe_open(fp, framework="pt", device="cpu") as f:
                t = f.get_tensor(key).float()
            if acc is None:
                acc = t
            else:
                acc.add_(t)
        acc.div_(len(files))
        out[key] = acc.half().contiguous()
        if idx % 50 == 0 or idx == len(keys):
            print(f"[soup] {idx}/{len(keys)} {key}", flush=True)

    save_file(out, os.path.join(dst, "model.safetensors"))

    # copy HF sidecars from fold0
    for fn in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json",
               "chat_template.jinja", "training_args.bin"):
        p = os.path.join(srcs[0], fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst, fn))

    cfg_p = os.path.join(dst, "config.json")
    cfg = json.load(open(cfg_p, encoding="utf-8"))
    rp = cfg.get("rope_parameters")
    if rp and "rope_theta" not in cfg:
        cfg["rope_theta"] = rp.get("rope_theta")
    json.dump(cfg, open(cfg_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    tcfg_p = os.path.join(dst, "tokenizer_config.json")
    if os.path.exists(tcfg_p):
        tcfg = json.load(open(tcfg_p, encoding="utf-8"))
        if isinstance(tcfg.get("extra_special_tokens"), list):
            tcfg.pop("extra_special_tokens")
            json.dump(tcfg, open(tcfg_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    mb = os.path.getsize(os.path.join(dst, "model.safetensors")) / 1e6
    print(f"[soup] saved -> {dst} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
