# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

데이콘 **2026 AI·SW중심대학 디지털 경진대회 AI부문** 참가 리포지토리 (https://dacon.io/competitions/official/236694).
AI 코딩 에이전트 세션 로그(`current_prompt` + `history` + `session_meta`)를 입력으로 **다음 행동(action)을 14개 클래스 중 하나로 분류**한다. 평가지표는 **Macro-F1**.

## ✅ 예선 종료 (2026-07-15) — 현재는 복기·학습 단계

- **최종 LB 0.77374** (`submit_distill_au07.zip` = Qwen3-0.6B full-data + 3-way 교사 증류 + au_bias×0.7). 본선컷 0.79307 미달
- **📖 복기·비교 분석은 [docs/05_retrospective.md](docs/05_retrospective.md) 부터 읽을 것** — 점수 경로, 기각 목록 전체, R107 오차 진단(천장의 정체), **수상팀 코드 나오면 확인할 8가지 질문**, 재현 절차, 남긴 자산 목록
- 저장소 정리 완료: 실패 실험 산출물 삭제(253GB 회수), **최종 제출물·최종 모델·OOF·학습데이터·코드는 보존** (복기 문서 §7)
- 앞으로의 작업: 수상팀 코드 공개 시 §6 질문에 답을 채우는 비교 분석

## 핵심 제약 — 코드 제출 대회

결과 CSV가 아니라 **zip(model/ + script.py + requirements.txt)을 제출**하고, 평가 서버가 zip을 풀어 `script.py`를 그대로 실행한다. 서버는 `./data/test.jsonl`, `./data/sample_submission.csv`를 제공하고 코드는 `./output/submission.csv`(sample_submission과 동일한 id 순서·컬럼)를 생성해야 한다.

- 서버 환경: **T4 16GB / 3 vCPU / RAM 12GB / Python 3.11.15 / CUDA 12.8**
- 제한: 패키지 설치 ≤10분, 추론 실행 ≤10분, **zip ≤1GB**
- **오프라인 실행** (pip install 외 인터넷 불가) → HuggingFace 모델·토크나이저는 반드시 `save_pretrained`로 zip에 동봉하고 `HF_HUB_OFFLINE=1` + `local_files_only=True`로 로드
- zip 최상위에 `script.py`가 바로 있어야 함 (폴더로 감싸면 실행 실패)
- 서버는 Python 3.11 — 로컬(3.13)과 다르므로 3.11 호환 코드로 작성, sklearn pickle은 requirements에 버전 고정 필수

## 데이터 (data/ — 수정 금지)

- `data/train.jsonl` 70,000샘플 + `data/train_labels.csv` — 스키마·EDA는 [docs/01_data.md](docs/01_data.md)
- `data/test.jsonl`은 **5행짜리 로컬 스텁**(train 샘플 재사용)이다. 실제 테스트셋은 서버에만 있으므로 로컬 성능 판단은 오직 CV로 한다.
- **CV는 반드시 세션 단위 GroupKFold**: id 형식이 `{session_id}-step_NN`이고 같은 세션의 스텝들은 history가 서로 겹친다 (9,429 세션, 세션당 최대 18스텝). 랜덤 split은 심각한 leakage → CV 점수가 LB와 완전히 어긋남.

## 환경 / 명령어

로컬: Windows 11 + PowerShell, Python 3.13 전역 설치, torch 2.6.0+cu124, **transformers 4.51.3 고정**, RTX 4070 Ti 12GB.

⚠️ 이 환경의 함정 (07-12 실측, docs/03 디버깅 기록):
- transformers를 5.x로 올리지 말 것 (학습 실패 이력). 제출 requirements.txt도 4.51.3 고정.
- XLM-R계 250k SPM 토크나이저 모델(xlm-roberta-base, Multilingual-MiniLM)은 이 환경에서 fine-tuning이 안 되는 문제 확인됨 — **새 백본 도입 시 반드시 512샘플 암기 스모크 먼저** (`--limit 512 --epochs 10`, train acc ~100% 확인).
- mdeberta-v3-base는 bs16×512에서 12GB VRAM 초과(공유메모리 스필로 30배 느려짐) → **bs8 + grad_accum** 사용.
- 학습은 bf16 autocast (DeBERTa fp16 불안정). GPU 작업과 CPU 무거운 작업(TF-IDF 등) 동시 실행 금지 — 서로 느려짐.

```powershell
# 학습 (full-data, 검증 레시피) — dacon 파이프라인 사용
python C:\Users\joon2\Desktop\dacon\work\train_full.py --tag <tag> --grad_ckpt
# fold 학습(게이트 실험)은 dacon/work/train.py --fold 0 --tag <tag> ... (레시피는 docs/02 참조)

# 제출 패키징: 반드시 dacon/work의 검증 도구 사용 (직접 zip 만들지 말 것 — docs/04 참조)
python C:\Users\joon2\Desktop\dacon\work\prune_qwen.py <model_dir_name>
python C:\Users\joon2\Desktop\dacon\work\package_multi.py --req_tf451 --au --out submit_xxx.zip

# 제출물 드라이런 — 서버 레이아웃을 스테이징 폴더에 재현해 실행
$stage = "stage"
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory "$stage\data" -Force | Out-Null
# (zip을 $stage에 풀고) 로컬 스텁 데이터 복사 후 실행
Copy-Item data\test.jsonl, data\sample_submission.csv "$stage\data\"
Set-Location $stage; python script.py; Set-Location ..
```

## 리포지토리 구조

- `baseline_submit/` — 주최측 베이스라인 (TF-IDF + LogReg, current_prompt만 사용). 제출 zip 구조의 참조 예시.
- `data/` — 대회 데이터 (위 참조)
- `docs/` — 00 대회정보 · 01 데이터 · 02 전략/계획 · 03 실험로그 · 04 제출 체크리스트

## 이전 프로젝트 자산 (C:\Users\joon2\Desktop\dacon\)

**같은 대회**의 이전 작업 폴더 — LB 0.7677 도달, 검증된 실험 로그(`PROGRESS.md`)와 학습된 모델이 있다. **주 엔진은 여기의 Qwen3-0.6B-Base 파이프라인** (근거·기각 목록은 docs/02_plan.md):
- 학습된 5-fold 모델·OOF: `dacon/artifacts/{models,oof}/qwen3_smoke_*`
- 검증된 코드: `dacon/work/` (train.py, featurize.py=V2 직렬화, script.py, package_multi.py, prune_qwen.py, bench_infer.py)
- 인코더(mdeberta/mbert/xlm-r)는 이 태스크에서 Qwen3 대비 −0.03 이상 열세 — 재실험 금지. 기각된 기법 목록은 docs/02_plan.md 참조.
- 이 리포의 `src/`는 EDA·CV 인프라(splits, evaluate의 threshold 튜닝)로 계속 사용; 트랜스포머 학습은 dacon/work를 쓴다.

## 작업 규칙

- 모든 학습 실험은 OOF 예측을 저장해둔다 — macro-F1용 클래스별 threshold 튜닝과 앙상블 가중 탐색에 재사용한다.
- public LB에 과적합하지 말 것: 본선 규칙에 private score 복원 코드 제출이 명시되어 있어 최종 평가는 private일 가능성이 높다. 의사결정 기준은 GroupKFold CV.
- 제출 전 [docs/04_submission.md](docs/04_submission.md) 체크리스트를 통과시킬 것 (오프라인 로드, zip 크기, 추론 시간, id 순서).
