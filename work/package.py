# -*- coding: utf-8 -*-
"""Build submit.zip: fp16 model + infer_config + script + requirements."""
import json
import os
import shutil
import zipfile

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
WORK = os.path.join(ROOT, "work")
SUBMIT = os.path.join(ROOT, "submit")

SRC_MODEL = os.path.join(ART, "models", "xlmr_v2_rdrop_lr4_fold0_best")  # V2+R-Drop+LR4e-5 (fold0 0.7231)
BIAS_JSON = os.path.join(ART, "bias_xlmr_len512.json")
MAX_LEN = 512
# split-half cross-fit showed the single-fold-tuned bias does NOT transfer
# (-0.004/-0.001 honest gain) -> ship bias only after it passes the
# cross-fitted gate on full 5-fold OOF (see tune_bias.py).
USE_BIAS = False


def main():
    # fresh submit dir
    if os.path.isdir(SUBMIT):
        shutil.rmtree(SUBMIT)
    m0 = os.path.join(SUBMIT, "model", "m0")
    os.makedirs(m0)

    # 1) model -> fp16
    print("converting model to fp16 ...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(SRC_MODEL)
    model.half()
    model.save_pretrained(m0, safe_serialization=True)
    AutoTokenizer.from_pretrained(SRC_MODEL).save_pretrained(m0)

    # 2) infer_config.json (classes recorded for label-order safety assert)
    cfg = {"model_dirs": ["m0"], "max_len": MAX_LEN,
           "classes": json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))}
    if USE_BIAS:
        cfg["class_bias"] = json.load(open(BIAS_JSON, encoding="utf-8"))["class_bias"]
    json.dump(cfg, open(os.path.join(SUBMIT, "model", "infer_config.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 3) code
    shutil.copy(os.path.join(WORK, "script.py"), os.path.join(SUBMIT, "script.py"))
    shutil.copy(os.path.join(WORK, "featurize.py"), os.path.join(SUBMIT, "featurize.py"))

    # 4) requirements — fast tokenizer loads from tokenizer.json, so nothing
    # extra is needed (verified under transformers 4.46.3 without sentencepiece)
    open(os.path.join(SUBMIT, "requirements.txt"), "w").write(
        "# torch/transformers/numpy pre-installed on the eval server\n")

    # sizes
    total = 0
    for dp, _, fs in os.walk(SUBMIT):
        for f in fs:
            fp = os.path.join(dp, f)
            sz = os.path.getsize(fp)
            total += sz
            if sz > 1_000_000:
                print(f"  {os.path.relpath(fp, SUBMIT):40s} {sz/1e6:8.1f} MB")
    print(f"TOTAL submit/ = {total/1e6:.1f} MB (limit 1000 MB)")

    # 5) zip
    zip_path = os.path.join(ROOT, "submit.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(SUBMIT):
            for f in fs:
                fp = os.path.join(dp, f)
                z.write(fp, os.path.relpath(fp, SUBMIT))
    print(f"zip = {os.path.getsize(zip_path)/1e6:.1f} MB -> {zip_path}")
    print("structure:")
    for dp, _, fs in os.walk(SUBMIT):
        for f in sorted(fs):
            print("   ", os.path.relpath(os.path.join(dp, f), SUBMIT))


if __name__ == "__main__":
    main()
