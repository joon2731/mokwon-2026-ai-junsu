# -*- coding: utf-8 -*-
"""4ep fold0(PID는 qwen3_4ep.pid) 완료를 감지하면 5ep fold0을 자동 실행.

- 4ep는 이미 별도 프로세스로 돌고 있음 → 이 스크립트는 OOF 파일이 생길 때까지 대기만.
- 5ep는 3ep/4ep와 동일 레시피, epochs만 5 (fresh 5-epoch 스케줄; 공정 비교용).
- 크래시 대비: 4ep가 OOF 없이 죽으면 중단(5ep 미실행). 재실행 시 skip-if-OOF로 안전.
"""
import ctypes
import os
import subprocess
import sys
import time

ROOT = r"C:\Users\joon2\Desktop\da2"
PY = sys.executable
OOF4 = os.path.join(ROOT, "artifacts", "oof", "qwen3_4ep_fold0.npz")
OOF5 = os.path.join(ROOT, "artifacts", "oof", "qwen3_5ep_fold0.npz")
PIDF = os.path.join(ROOT, "artifacts", "qwen3_4ep.pid")

try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception:
    pass


def alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return True  # 확인 실패 시 살아있다 가정 (대기 지속)


pid = None
if os.path.exists(PIDF):
    try:
        pid = int(open(PIDF).read().strip())
    except Exception:
        pass

print(f"[chain] 4ep(PID {pid}) 완료 대기 시작...", flush=True)
while not os.path.exists(OOF4):
    time.sleep(60)
    if pid and not alive(pid) and not os.path.exists(OOF4):
        print("[chain][중단] 4ep가 OOF 없이 종료됨(크래시 추정). 5ep 미실행. "
              "4ep 재개 후 이 스크립트 다시 실행 필요.", flush=True)
        sys.exit(1)
print("[chain] 4ep 완료 감지! 5ep 시작.", flush=True)

if os.path.exists(OOF5):
    print("[chain][skip] 5ep OOF 이미 존재.", flush=True)
else:
    cmd = [PY, "work/train.py", "--model", "pretrained\\Qwen3-0.6B-Base",
           "--fold", "0", "--tag", "qwen3_5ep", "--max_len", "512", "--epochs", "5",
           "--bs", "8", "--grad_accum", "4", "--precision", "bf16", "--lr", "2e-5",
           "--warmup", "0.1", "--wd", "0.01", "--optim", "adamw_bnb_8bit",
           "--grad_ckpt", "--weighting", "sqrt", "--seed", "42"]
    print("[chain][start] 5ep (8750 스텝, ~5.7h)", flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    print(f"[chain][done] 5ep rc={rc}", flush=True)

print("[chain] CHAIN DONE (4ep -> 5ep)", flush=True)
