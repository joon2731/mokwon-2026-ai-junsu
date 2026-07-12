"""데이터 로딩 유틸.

모든 스크립트는 리포지토리 루트에서 실행한다: python src\\xxx.py
(data/ 경로가 CWD 기준 상대경로이기 때문)
"""
import csv
import json
from pathlib import Path

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")


def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_train():
    """train 샘플 리스트를 반환. 각 샘플에 정답 'action' 필드를 병합해 둔다."""
    samples = load_jsonl(DATA_DIR / "train.jsonl")
    labels = {}
    with open(DATA_DIR / "train_labels.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]
    for s in samples:
        s["action"] = labels[s["id"]]
    return samples


def load_test():
    return load_jsonl(DATA_DIR / "test.jsonl")


def session_id(sample_or_id):
    """'sess_sim_..._024730-step_08' → 'sess_sim_..._024730' (CV 그룹 키)."""
    sid = sample_or_id if isinstance(sample_or_id, str) else sample_or_id["id"]
    return sid.rsplit("-step_", 1)[0]


def load_folds():
    """artifacts/splits.csv → {id: fold}. splits.py로 생성된 고정 스플릿."""
    fold_of = {}
    with open(ARTIFACTS_DIR / "splits.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fold_of[row["id"]] = int(row["fold"])
    return fold_of
