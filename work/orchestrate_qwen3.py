# -*- coding: utf-8 -*-
"""Qwen3-0.6B 자동 오케스트레이터 — GPU 연속 가동, 무인 진행.

흐름:
  1) 4ep fold0(별도 실행중, qwen3_4ep.pid) 완료 대기 (크래시 시 재개 시도)
  2) 5ep fold0 실행
  3) 3ep/4ep/5ep fold0 macro-F1 비교 → 최고 에폭 자동 선택
     (근소차<0.001이면 적은 에폭, 3ep 대비 +0.001 미만이면 3ep 유지 — 노이즈 회피)
  4) 선택된 에폭으로 5-fold(fold1-4) 자동 완성 → 제출용 OOF/모델

모두 skip-if-OOF라 재크래시 후 이 스크립트만 다시 실행하면 남은 지점부터 이어감.
tag: 3ep=qwen3_smoke, 4ep=qwen3_4ep, 5ep=qwen3_5ep.
"""
import ctypes
import os
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import f1_score

ROOT = r"C:\Users\joon2\Desktop\da2"
PY = sys.executable
OOF = os.path.join(ROOT, "artifacts", "oof")
TAG = {3: "qwen3_smoke", 4: "qwen3_4ep", 5: "qwen3_5ep"}

try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception:
    pass

RECIPE = ["--model", "pretrained\\Qwen3-0.6B-Base", "--max_len", "512", "--bs", "8",
          "--grad_accum", "4", "--precision", "bf16", "--lr", "2e-5", "--warmup", "0.1",
          "--wd", "0.01", "--optim", "adamw_bnb_8bit", "--grad_ckpt",
          "--weighting", "sqrt", "--seed", "42"]


def oof_path(tag, fold):
    return os.path.join(OOF, f"{tag}_fold{fold}.npz")


def train(tag, epochs, fold, retries=1):
    if os.path.exists(oof_path(tag, fold)):
        print(f"[orch][skip] {tag} fold{fold} OOF 존재", flush=True)
        return 0
    cmd = [PY, "work/train.py", "--fold", str(fold), "--tag", tag,
           "--epochs", str(epochs)] + RECIPE
    # 기존 체크포인트 있으면 최신에서 재개 (크래시 후 scratch 재시작 방지)
    run_dir = os.path.join(ROOT, "artifacts", "models", f"{tag}_fold{fold}")
    if os.path.isdir(run_dir):
        cks = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")]
        if cks:
            latest = max(cks, key=lambda d: int(d.split("-")[1]))
            cmd += ["--resume_from_checkpoint", os.path.join(run_dir, latest)]
            print(f"[orch] {tag} fold{fold}: checkpoint {latest}에서 재개", flush=True)
    for attempt in range(retries + 1):
        print(f"[orch][start] {tag} fold{fold} {epochs}ep (try {attempt + 1})", flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        print(f"[orch][done] {tag} fold{fold} rc={rc}", flush=True)
        if rc == 0 and os.path.exists(oof_path(tag, fold)):
            return 0
        if attempt < retries:
            print(f"[orch][retry] {tag} fold{fold} 60초 후 재시도", flush=True)
            time.sleep(60)
    return 1


def macro(tag, fold=0):
    d = np.load(oof_path(tag, fold), allow_pickle=True)
    return f1_score(d["y"], d["logits"].argmax(1), average="macro")


def alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return True


# --- 1) 4ep fold0 완료 대기 ---
pidf = os.path.join(ROOT, "artifacts", "qwen3_4ep.pid")
pid = None
if os.path.exists(pidf):
    try:
        pid = int(open(pidf).read().strip())
    except Exception:
        pass
print(f"[orch] 4ep(PID {pid}) fold0 완료 대기...", flush=True)
while not os.path.exists(oof_path("qwen3_4ep", 0)):
    time.sleep(60)
    if pid and not alive(pid) and not os.path.exists(oof_path("qwen3_4ep", 0)):
        print("[orch][경고] 4ep가 OOF 없이 종료(크래시?). 재개 시도.", flush=True)
        if train("qwen3_4ep", 4, 0) != 0:
            print("[orch][중단] 4ep 재개 실패. 종료.", flush=True)
            sys.exit(1)
print("[orch] 4ep fold0 완료.", flush=True)

# --- 2) 5ep fold0 ---
train("qwen3_5ep", 5, 0)

# --- 3) 최고 에폭 선택 ---
scores = {}
for e, tag in TAG.items():
    try:
        scores[e] = macro(tag)
    except Exception as ex:
        print(f"[orch][경고] {tag} fold0 OOF 못읽음: {ex}", flush=True)
print(f"[orch] fold0 macro: " +
      " ".join(f"{e}ep={scores.get(e, float('nan')):.4f}" for e in (3, 4, 5)), flush=True)

if not scores:
    print("[orch][중단] 점수 없음.", flush=True)
    sys.exit(1)
top = max(scores.values())
cands = [e for e in scores if scores[e] >= top - 0.001]   # 최댓값과 0.001 이내
best = min(cands)                                          # 그 중 적은 에폭
if 3 in scores and scores[best] <= scores[3] + 0.001:     # 3ep 대비 개선 미미하면 3ep
    best = 3
print(f"[orch] === 최고 에폭 = {best}ep (tag={TAG[best]}) === 5-fold 진행", flush=True)

# --- 4) 선택 에폭으로 5-fold (fold1-4) ---
fails = []
for f in (1, 2, 3, 4):
    if train(TAG[best], best, f) != 0:
        fails.append(f)
if fails:
    print(f"[orch][경고] 실패 fold: {fails} (재실행 시 이어감)", flush=True)
print(f"[orch] CHAIN DONE — {TAG[best]} {best}ep 5-fold "
      f"({4 - len(fails)}/4 성공). 다음: blend_oof 조립 → 패키징.", flush=True)
