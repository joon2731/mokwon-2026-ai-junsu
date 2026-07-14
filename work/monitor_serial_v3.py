# -*- coding: utf-8 -*-
"""Lightweight monitor for qwen3_serial_v3 overnight run."""
import datetime as dt
import os
import re
import time

ROOT = r"C:\Users\joon2\Desktop\da2"
ART = os.path.join(ROOT, "artifacts")
LOG_PATH = os.path.join(ART, "qwen3_serial_v3.log")
ERR_PATH = os.path.join(ART, "qwen3_serial_v3.err")
PID_PATH = os.path.join(ART, "qwen3_serial_v3.pid")
STATUS_PATH = os.path.join(ART, "qwen3_serial_v3_monitor.log")
PROGRESS_PATH = os.path.join(ROOT, "PROGRESS.md")


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_tail(path, n=12000):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n), os.SEEK_SET)
        return f.read().decode("utf-8", errors="ignore")


def is_running(pid):
    if not pid:
        return False
    try:
        import subprocess
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"],
            capture_output=True, text=True, timeout=10)
        return str(pid) in r.stdout
    except Exception:
        return False


def extract_state():
    txt = read_tail(LOG_PATH)
    err = read_tail(ERR_PATH, 4000)
    macro = None
    vals = re.findall(r"MACRO-F1 \(fold 0\): ([0-9.]+)", txt)
    if vals:
        macro = vals[-1]
    evals = re.findall(r"'eval_macro_f1': ([0-9.]+)", txt)
    if evals:
        macro = evals[-1]
    losses = re.findall(r"'loss': ([0-9.]+).*?'epoch': ([0-9.]+)", txt)
    loss = losses[-1] if losses else None
    fatal = ""
    for key in ("Traceback", "RuntimeError", "CUDA out of memory", "Error"):
        if key in err:
            fatal = key
            break
    if macro:
        return f"macro={macro}"
    if loss:
        return f"loss={loss[0]} epoch={loss[1]}"
    if "fold 0:" in txt:
        return "training started"
    if "V3 serialized text" in txt:
        return "data prepared"
    if fatal:
        return f"err={fatal}"
    return "waiting"


def append_progress(msg):
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n- [{now()}] V3 monitor: {msg}\n")


def main():
    last = None
    last_progress = 0.0
    pid = ""
    for _ in range(720):  # up to 12 hours, 1 min interval
        if os.path.exists(PID_PATH):
            raw_pid = open(PID_PATH, encoding="utf-8", errors="ignore").read()
            m = re.search(r"\d+", raw_pid)
            pid = m.group(0) if m else ""
        state = extract_state()
        running = is_running(pid)
        line = f"[{now()}] pid={pid or '?'} running={int(running)} {state}"
        os.makedirs(ART, exist_ok=True)
        with open(STATUS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        t = time.time()
        if state != last or t - last_progress >= 900:
            append_progress(line)
            last = state
            last_progress = t
        if pid and not running and state not in ("waiting", "data prepared", "training started"):
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
