import os
import subprocess
import sys

ROOT = r"C:\Users\joon2\Desktop\da2"
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

cmd = [
    sys.executable,
    "work/train.py",
    "--model", "pretrained/Qwen3-0.6B-Base",
    "--fold", "0",
    "--max_len", "384",
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
    "--tag", "qwen3_len384",
]

out_path = os.path.join(ROOT, "artifacts", "qwen3_len384.log")
err_path = os.path.join(ROOT, "artifacts", "qwen3_len384.err")
pid_path = os.path.join(ROOT, "artifacts", "qwen3_len384.pid")

out = open(out_path, "w", encoding="utf-8", buffering=1)
err = open(err_path, "w", encoding="utf-8", buffering=1)

creationflags = 0
if os.name == "nt":
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

proc = subprocess.Popen(
    cmd,
    cwd=ROOT,
    env=env,
    stdout=out,
    stderr=err,
    stdin=subprocess.DEVNULL,
    creationflags=creationflags,
)

with open(pid_path, "w", encoding="utf-8") as f:
    f.write(str(proc.pid))

print(proc.pid)
