# -*- coding: utf-8 -*-
"""Run Qwen3 fold0 with serialization V3, then package submit3 if it improves."""
import datetime as dt
import os
import re
import subprocess
import sys

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
PY = sys.executable
TAG = "qwen3_serial_v3"
BASELINE = 0.7679242648521141
DATA_PATH = os.path.join(ART, "train_prepared_serial_v3.parquet")
LOG_PATH = os.path.join(ART, "qwen3_serial_v3.log")
ERR_PATH = os.path.join(ART, "qwen3_serial_v3.err")
STATUS_PATH = os.path.join(ART, "qwen3_serial_v3_status.txt")
PROGRESS_PATH = os.path.join(ROOT, "PROGRESS.md")


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_status(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    os.makedirs(ART, exist_ok=True)
    with open(STATUS_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def progress(msg):
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n- [{now()}] {msg}\n")


def run(cmd, log_mode="a"):
    write_status("RUN " + " ".join(cmd))
    with open(LOG_PATH, log_mode, encoding="utf-8") as out, open(ERR_PATH, log_mode, encoding="utf-8") as err:
        out.write(f"\n===== {now()} RUN {' '.join(cmd)} =====\n")
        out.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=err, text=True)
        code = p.wait()
    write_status(f"EXIT {code}: {' '.join(cmd)}")
    if code != 0:
        progress(f"V3 직렬화 fold0 실행 실패: exit={code}, command={' '.join(cmd)}")
        raise SystemExit(code)


def parse_macro():
    if not os.path.exists(LOG_PATH):
        return None
    txt = open(LOG_PATH, encoding="utf-8", errors="ignore").read()
    vals = re.findall(r"MACRO-F1 \(fold 0\): ([0-9.]+)", txt)
    if vals:
        return float(vals[-1])
    vals = re.findall(r"'eval_macro_f1': ([0-9.]+)", txt)
    if vals:
        return float(vals[-1])
    return None


def main():
    os.makedirs(ART, exist_ok=True)
    write_status(f"START {TAG} fold0 V3")
    progress("V3 직렬화 fold0 실험 시작: qwen3_serial_v3, max_len=512, 3ep. 로그: artifacts/qwen3_serial_v3.log")

    run([PY, "work\\prepare_data_v3.py", "--out", DATA_PATH], log_mode="w")

    train_cmd = [
        PY, "work\\train.py",
        "--model", "pretrained\\Qwen3-0.6B-Base",
        "--fold", "0",
        "--max_len", "512",
        "--epochs", "3",
        "--bs", "8",
        "--grad_accum", "4",
        "--lr", "2e-5",
        "--warmup", "0.1",
        "--wd", "0.01",
        "--weighting", "sqrt",
        "--precision", "bf16",
        "--grad_ckpt",
        "--optim", "adamw_bnb_8bit",
        "--tag", TAG,
        "--data_path", DATA_PATH,
    ]
    run(train_cmd)

    macro = parse_macro()
    if macro is None:
        progress("V3 직렬화 fold0 학습 완료. Macro-F1 파싱 실패, 로그 확인 필요.")
        write_status("DONE training, macro parse failed")
        return

    delta = macro - BASELINE
    progress(f"V3 직렬화 fold0 완료: CV {macro:.6f} (기존 Qwen3 fold0 {BASELINE:.6f} 대비 {delta:+.6f})")
    write_status(f"DONE training macro={macro:.6f} delta={delta:+.6f}")

    if macro <= BASELINE:
        progress("V3 직렬화 fold0가 기존보다 낮거나 같아서 submit3 패키징은 진행하지 않음.")
        return

    progress("V3 직렬화 fold0가 기존보다 높아서 prune 및 submit3.zip 패키징 진행.")
    run([PY, "work\\prune_qwen.py", f"{TAG}_fold0_best", "--featurizer", "featurize_v3", "--max_len", "512"])
    run([
        PY, "work\\package_multi.py",
        "--single_model", f"{TAG}_fold0_best_pruned",
        "--max_len", "512",
        "--req_tf451",
        "--au",
        "--featurize_file", "work\\featurize_v3.py",
        "--out", "submit3.zip",
    ])
    run([PY, "-m", "zipfile", "-t", "submit3.zip"])
    progress("submit3.zip 생성 및 zip test 완료. 서버 제출 후보로 확인 필요.")
    write_status("DONE submit3.zip")


if __name__ == "__main__":
    main()

