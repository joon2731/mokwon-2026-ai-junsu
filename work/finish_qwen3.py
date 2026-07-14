# -*- coding: utf-8 -*-
"""3ep 5-fold 완료 후 자동 제출준비: blend(5-fold CV 로그) → fold0 프루닝 → 4.51 번들 단일 패키징.

사용자 지시: 5-fold 끝 → 결과 로그 → submit_qwen3.zip 준비 → 멈춤. 추가작업 없음.
GPU 겹침 방지 위해 5-fold 드라이버(PID qwen3_5fold_3ep.pid) 종료 후 실행.
"""
import os
import subprocess
import sys
import time

ROOT = r"C:\Users\joon2\Desktop\da2"
PY = sys.executable
PIDF = os.path.join(ROOT, "artifacts", "qwen3_5fold_3ep.pid")


def alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return True


def run(desc, args):
    print(f"\n[finish] >>> {desc}", flush=True)
    rc = subprocess.run([PY] + args, cwd=ROOT).returncode
    print(f"[finish] <<< {desc} rc={rc}", flush=True)
    return rc


pid = None
if os.path.exists(PIDF):
    try:
        pid = int(open(PIDF).read().strip())
    except Exception:
        pass

print(f"[finish] 5-fold 드라이버(PID {pid}) 종료 대기...", flush=True)
while pid and alive(pid):
    time.sleep(120)
print("[finish] 5-fold 드라이버 종료 감지. 제출준비 시작.", flush=True)
time.sleep(10)  # 마지막 파일 flush 여유

# 1) 5-fold OOF 조립 + CV (로그에 macro 출력)
run("blend_oof (3ep 5-fold CV)", ["work/blend_oof.py", "--tags", "qwen3_smoke"])

# 2) fold0 임베딩 프루닝 (제출 크기 940MB로)
run("prune_qwen fold0", ["work/prune_qwen.py", "qwen3_smoke_fold0_best"])

# 3) 4.51 번들 단일 패키징 (+au prior)
rc = run("package submit_qwen3.zip",
         ["work/package_multi.py", "--bundle_tf451", "--au", "--out", "submit_qwen3.zip"])

zp = os.path.join(ROOT, "submit_qwen3.zip")
if os.path.exists(zp):
    print(f"[finish] ✅ 완료: submit_qwen3.zip = {os.path.getsize(zp)/1e6:.0f}MB "
          f"(제출 한도 1000MB). 5-fold CV는 위 blend 로그 참조.", flush=True)
    print("[finish] ⚠ 첫 업로드가 4.51 번들 메커니즘 프로브임(같은/더 나은 점수면 OK).", flush=True)
else:
    print("[finish] ❌ 패키징 실패 — 로그 확인 필요.", flush=True)
print("[finish] STOP.", flush=True)
