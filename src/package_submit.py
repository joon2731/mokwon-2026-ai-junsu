"""학습 산출물 → 제출 zip 패키징.

실행 (리포지토리 루트에서):
  python src\\package_submit.py --exp E002 --folds 0 --out submit_e002_f0
  python src\\package_submit.py --exp E002 --folds 0,1,2,3,4 --bias --out submit_e002_all

동작:
  submit_{out}/
    ├── model/fold{k}/...      (artifacts/{exp}/fold{k} 복사)
    ├── model/classes.json     (artifacts/{exp}/classes.json)
    ├── model/bias.json        (--bias 시 report.json의 bias_full_oof)
    ├── script.py              (src/submit_script_template.py 복사)
    └── requirements.txt
  → {out}.zip (내용물이 zip 루트에 위치)

zip 후 크기를 출력하고 1GB 초과 시 경고.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from data import ARTIFACTS_DIR

# 로컬 학습 환경과 동일 버전으로 고정 (transformers 5.13은 로컬에서 학습 실패 이력
# 있음 — docs/03 디버깅 기록 참조. 4.51.3으로 학습했으니 서버도 4.51.3)
REQUIREMENTS = """torch==2.6.0
transformers==4.51.3
numpy
sentencepiece==0.2.0
protobuf
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--bias", action="store_true",
                    help="report.json의 bias_full_oof를 bias.json으로 동봉")
    ap.add_argument("--out", required=True, help="출력 이름 (폴더/zip)")
    args = ap.parse_args()

    exp_dir = ARTIFACTS_DIR / args.exp
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True)

    for f in args.folds.split(","):
        src = exp_dir / f"fold{f.strip()}"
        assert src.exists(), f"missing {src}"
        shutil.copytree(src, model_dir / f"fold{f.strip()}")
        print(f"copied {src}")

    shutil.copy(exp_dir / "classes.json", model_dir / "classes.json")
    if args.bias:
        report = json.loads((exp_dir / "report.json").read_text(encoding="utf-8"))
        bias = report.get("bias_full_oof")
        assert bias, "report.json에 bias_full_oof 없음 (전체 fold 학습 필요)"
        (model_dir / "bias.json").write_text(json.dumps(bias), encoding="utf-8")
        print(f"bias.json: {bias}")

    shutil.copy(Path("src/submit_script_template.py"), out_dir / "script.py")
    (out_dir / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    zip_path = Path(f"{args.out}.zip")
    if zip_path.exists():
        zip_path.unlink()
    # PowerShell Compress-Archive 대신 파이썬으로 (루트 배치 보장)
    shutil.make_archive(args.out, "zip", root_dir=out_dir)
    size_gb = zip_path.stat().st_size / (1024 ** 3)
    print(f"\n{zip_path}: {size_gb:.3f} GB" + ("  !! OVER 1GB LIMIT" if size_gb > 1.0 else " (OK)"))
    print(f"드라이런: CLAUDE.md의 스테이징 절차로 {out_dir}\\ 검증 후 제출")


if __name__ == "__main__":
    main()
