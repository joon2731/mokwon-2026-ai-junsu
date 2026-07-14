# -*- coding: utf-8 -*-
"""Build a multi-model submit zip (heterogeneous ensemble OK: XLM-R + Qwen ...).

script.py already loads tokenizer+model PER model dir and mean-softmax ensembles,
so this just packages N fp16 models as model/m0, m1, ... + infer_config.json.

Edit MODELS/MAX_LEN below, then:  python work\\package_multi.py [--out submit.zip]

Label-order safety: asserts every model's id2label matches classes.json at
BUILD time (script.py asserts again at runtime).
"""
import argparse
import json
import os
import shutil
import zipfile

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
WORK = os.path.join(ROOT, "work")

# ---- EDIT ME -----------------------------------------------------------
MODELS = [
    ("m0", os.path.join(ART, "models", "qwen3_distill3w_full_best_pruned")),
]
MODEL_WEIGHTS = None  # 단일 모델. (블렌드 실험 시 모델 수와 길이 일치 필수 — 07-14 assert 사고)
MAX_LEN = 512
# ------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submit.zip")
    ap.add_argument("--single_model", default=None,
                    help="override MODELS with one trained model dir under artifacts/models, "
                         "for example qwen3_len384_fold0_best_pruned")
    ap.add_argument("--max_len", type=int, default=None,
                    help="override infer max_len for this package")
    ap.add_argument("--featurize_file", default=os.path.join(WORK, "featurize.py"),
                    help="serialization file to bundle as featurize.py")
    ap.add_argument("--overlay", action="store_true",
                    help="model/overlay_lookup.json 동봉 (train-side overlay probe)")
    ap.add_argument("--au", action="store_true",
                    help="model/au_bias.json 동봉 (au-prior 보정)")
    ap.add_argument("--bundle_tf451", action="store_true",
                    help="transformers 4.51 리눅스 휠을 libs/로 언집 동봉 (qwen3 제출용; "
                         "script.py가 libs/ 있으면 sys.path 주입). 서버 4.46.3이 qwen3 미지원일 때만 필요")
    ap.add_argument("--req_tf451", action="store_true",
                    help="libs/ 동봉 없이 requirements.txt로 transformers 4.51 계열 설치를 요청 "
                         "(서버 패키지 설치 단계 온라인 동작 검증용)")
    args = ap.parse_args()
    if args.bundle_tf451 and args.req_tf451:
        raise SystemExit("--bundle_tf451 and --req_tf451 are mutually exclusive")

    models = MODELS
    max_len = MAX_LEN
    if args.single_model:
        models = [("m0", os.path.join(ART, "models", args.single_model))]
    if args.max_len is not None:
        max_len = args.max_len

    classes = json.load(open(os.path.join(ART, "classes.json"), encoding="utf-8"))
    submit = os.path.join(ROOT, "submit_multi")
    if os.path.isdir(submit):
        shutil.rmtree(submit)
    os.makedirs(os.path.join(submit, "model"))

    names = []
    for name, src in models:
        dst = os.path.join(submit, "model", name)
        print(f"[{name}] {src} -> fp16", flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(src)
        got = [model.config.id2label[i] for i in range(len(model.config.id2label))]
        assert got == classes, f"label order mismatch in {src}:\n{got}\nvs classes.json"
        model.half()
        model.save_pretrained(dst, safe_serialization=True)
        # transformers 5.x는 rope_theta를 rope_parameters로 개명 저장 → 서버(4.46.3)가
        # 기본값 10000으로 오해해 위치인코딩 붕괴 (제출 #10 사고). 옛 키를 복원해 양쪽 호환.
        cfg_p = os.path.join(dst, "config.json")
        cj = json.load(open(cfg_p, encoding="utf-8"))
        rp = cj.get("rope_parameters")
        if rp and "rope_theta" not in cj:
            cj["rope_theta"] = rp.get("rope_theta")
            json.dump(cj, open(cfg_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"  [{name}] rope_theta={cj['rope_theta']} 복원 (4.46.3 호환)")
        # 토크나이저·리맵은 파일 그대로 복사 (프루닝 수술본의 HF 재직렬화 왕복 회피)
        copied = False
        for tf in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "added_tokens.json", "vocab_remap.npy"):
            p = os.path.join(src, tf)
            if os.path.exists(p):
                shutil.copy(p, os.path.join(dst, tf))
                copied = tf == "tokenizer.json" or copied
        if not copied:
            AutoTokenizer.from_pretrained(src).save_pretrained(dst)
        # transformers 5.x는 tokenizer_config에 extra_special_tokens를 list로 저장 →
        # 4.5x대(예: 4.51.3)가 dict로 기대해 첫 인코딩에서 크래시('list' has no 'keys').
        # 키 제거(특수토큰은 tokenizer.json에 잔존해 무손실). XLM-R엔 no-op, 4.46.3에도 무해 → 항상 적용.
        tcfg_p = os.path.join(dst, "tokenizer_config.json")
        if os.path.exists(tcfg_p):
            tj = json.load(open(tcfg_p, encoding="utf-8"))
            if isinstance(tj.get("extra_special_tokens"), list):
                tj.pop("extra_special_tokens")
                json.dump(tj, open(tcfg_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                print(f"  [{name}] extra_special_tokens(list) 제거 (4.51 호환)")
        names.append(name)

    cfg = {"model_dirs": names, "max_len": max_len, "classes": classes}
    if MODEL_WEIGHTS:
        assert len(MODEL_WEIGHTS) == len(names)
        cfg["model_weights"] = MODEL_WEIGHTS
    json.dump(cfg, open(os.path.join(submit, "model", "infer_config.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if args.overlay:
        shutil.copy(os.path.join(ART, "overlay_lookup.json"),
                    os.path.join(submit, "model", "overlay_lookup.json"))
        print("overlay_lookup.json 동봉 (probe build)")
    if args.au:
        shutil.copy(os.path.join(ART, "au_bias.json"),
                    os.path.join(submit, "model", "au_bias.json"))
        print("au_bias.json 동봉 (au-prior 보정)")

    shutil.copy(os.path.join(WORK, "script.py"), os.path.join(submit, "script.py"))
    shutil.copy(args.featurize_file, os.path.join(submit, "featurize.py"))
    req_path = os.path.join(submit, "requirements.txt")
    if args.req_tf451:
        open(req_path, "w", encoding="utf-8").write(
            "transformers==4.51.3\n"
            "tokenizers==0.21.0\n"
            "huggingface_hub==0.30.0\n")
    else:
        open(req_path, "w", encoding="utf-8").write(
            "# torch/transformers/numpy pre-installed on the eval server\n")

    # transformers 4.51 오프라인 번들 (qwen3처럼 서버 4.46.3이 못 읽는 모델용).
    # 휠을 .dist-info 포함 그대로 libs/에 언집 → script.py가 sys.path[0]에 주입해 서버 설치본을 shadow.
    # (pip 미사용 → 오프라인 설치 불확실성 회피. tokenizers는 abi3 리눅스 .so라 T4 py3.11에서 로드됨)
    if args.bundle_tf451:
        libs = os.path.join(submit, "libs")
        os.makedirs(libs, exist_ok=True)
        vend = os.path.join(ROOT, "vendor", "tf451")
        whls = sorted(f for f in os.listdir(vend) if f.endswith(".whl"))
        for whl in whls:
            with zipfile.ZipFile(os.path.join(vend, whl)) as z:
                z.extractall(libs)
        print(f"[libs] transformers 4.51 번들: {whls} -> submit_multi/libs/")

    total = 0
    for dp, _, fs in os.walk(submit):
        for f in fs:
            fp = os.path.join(dp, f)
            sz = os.path.getsize(fp)
            total += sz
            if sz > 1_000_000:
                print(f"  {os.path.relpath(fp, submit):50s} {sz/1e6:8.1f} MB")
    print(f"TOTAL(unzipped) = {total/1e6:.1f} MB (ref only; limit is zip size)")

    zip_path = os.path.join(ROOT, args.out)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(submit):
            for f in fs:
                fp = os.path.join(dp, f)
                z.write(fp, os.path.relpath(fp, submit))
    zmb = os.path.getsize(zip_path) / 1e6
    print(f"zip = {zmb:.1f} MB -> {zip_path}   (제출 한도: zip 1000 MB)")
    if zmb > 990:
        print("!! zip 1GB 근접/초과 — 프루닝 강화 또는 모델 수 축소 필요")


if __name__ == "__main__":
    main()
