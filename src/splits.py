"""세션 단위 StratifiedGroupKFold 스플릿을 생성해 artifacts/splits.csv로 고정.

모든 실험은 이 파일의 fold 배정을 그대로 사용한다. 재생성 금지 (seed 42 고정) —
스플릿이 바뀌면 실험 간 CV 비교가 무너진다.

실행: python src\\splits.py  (리포지토리 루트에서)
"""
import csv
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold

from data import ARTIFACTS_DIR, load_train, session_id

N_SPLITS = 5
SEED = 42
OUT = ARTIFACTS_DIR / "splits.csv"


def main():
    samples = load_train()
    ids = [s["id"] for s in samples]
    y = [s["action"] for s in samples]
    groups = [session_id(i) for i in ids]

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_of = {}
    for fold, (_, val_idx) in enumerate(sgkf.split(ids, y, groups)):
        for i in val_idx:
            fold_of[ids[i]] = fold

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "fold"])
        for i in ids:
            w.writerow([i, fold_of[i]])

    # 검증 1: 한 세션이 두 fold에 걸치면 안 됨
    sess_fold = {}
    n_bad = 0
    for i in ids:
        sid = session_id(i)
        if sid in sess_fold and sess_fold[sid] != fold_of[i]:
            n_bad += 1
        sess_fold.setdefault(sid, fold_of[i])
    # 검증 2: fold 크기·클래스 분포 균형
    sizes = Counter(fold_of.values())
    print("fold sizes:", dict(sorted(sizes.items())))
    print("sessions split across folds:", n_bad)
    rare = [c for c, n in Counter(y).items() if n < 2000]
    for c in rare:
        dist = Counter(fold_of[i] for i, a in zip(ids, y) if a == c)
        print(f"rare class {c}: {dict(sorted(dist.items()))}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
