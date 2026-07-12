# 실험 로그

> 규칙: 학습·제출 실험은 **전부** 여기에 기록한다. CV는 5-fold GroupKFold(session_id, seed 42) macro-F1.
> OOF/테스트 확률은 `artifacts/{exp_id}/`에 저장 (threshold 튜닝·앙상블 재료).

## 요약 보드

| exp | 날짜 | 모델/방법 | 입력 | CV (mean±std) | LB | zip | 추론시간 | 비고 |
|---|---|---|---|---|---|---|---|---|
| E000 | 07-12 | TF-IDF+LogReg (주최측 파리티) | prompt만 | 0.42605±0.0044 | **0.43576** | ~8MB | 로컬 수초 | LB는 주최측 zip 그대로(전체데이터 학습). 튜닝 후 CV 0.43734 |
| E001 | 07-12 | TF-IDF+LogReg, class_weight=balanced | full(v1) | 0.49791±0.0034 | | | | 튜닝 후 0.50854. 컨텍스트 +0.07 |
| SMK_MBERT | 07-12 | bert-base-multilingual-cased 2ep | now_first | 0.379 (6k, fold0만) | | | | 환경 판별용 |
| SMK_KLUE | 07-12 | klue/roberta-base 2ep | now_first | 0.130 (6k) | | | | 영어/코드 혼합에 불리 |
| SMK_MDEB | 07-12 | mdeberta-v3-base 2ep | now_first | 0.187 (6k) | | | | 이전 프로젝트에서도 최종 0.68로 열세 |
| (이전) | ~07-09 | **Qwen3-0.6B-Base 3ep** (dacon/) | V2 | fold0 **0.7679** / 5-fold OOF **0.7701** | **0.7677** (fold0+au) | 839MB | 9:31 | 현재 기준선. dacon/PROGRESS.md |
| E100 | 07-13 | Qwen3-0.6B **full-data** 3ep | V2 | (홀드아웃 없음) | | | | 07-12 밤 학습 중 |

## 제출 기록 (일 10회 제한 — 남은 횟수 관리)

| # | 날짜 | exp | LB | 메모 |
|---|---|---|---|---|
| 1 | 07-12 | 주최측 baseline zip 그대로 | 0.43576 | 스모크 ①. CV(0.426, 80%학습)↔LB(0.436, 100%학습) 오프셋 ~+0.01로 정합 → **CV 신뢰 가능** |

## 실험 상세

### E000 — 주최측 베이스라인 파리티 (prompt-only TF-IDF+LogReg)
- 구성: TF-IDF word(1,2)+char_wb(2,4) → LogReg(C=1), 5-fold GroupKFold. 주최측 pkl 그대로가 아니라 우리 CV로 재학습한 파리티 버전
- CV: [0.43247, 0.42185, 0.43011, 0.42175, 0.42407] → **0.42605 ± 0.0044**
- fold-out threshold 튜닝: **0.43734 ± 0.0010** (+0.011 — 튜닝 파이프라인 동작 확인, 선형 모델에서도 유의미한 이득)
- 결론: prompt 단독은 0.43 수준. 리더보드 0.79와의 갭 = 컨텍스트(history/meta) + 강한 인코더의 몫

### E001 — full-context TF-IDF+LogReg
- 구성: E000과 동일 + 입력을 serialize(full)로 (META+히스토리 6쌍+NOW), class_weight=balanced
- CV: **0.49791 ± 0.0034**, fold-out threshold 튜닝 후 **0.50854 ± 0.0045**
- 결론: 컨텍스트(history+meta)가 선형 모델에서도 +0.07. 남은 갭(0.51→0.79)은 인코더 표현력의 몫

### 디버깅 기록 (07-12 밤) — "트랜스포머가 학습 안 됨" 사건
- 증상: MiniLM/xlm-roberta가 512샘플 암기도 실패 (랜덤 레이블 컨트롤 acc 9%), 실데이터 6k에서 F1 0.04
- 격리 과정: 순수 torch MLP ✓ / 순수 torch TransformerEncoder ✓ / HF 포워드 CPU=GPU ✓ / 옵티마이저-파라미터 identity ✓ / fp32·bf16 동일 / eager·sdpa 동일 / transformers 5.13→4.51.3 다운그레이드해도 동일
- 판별: **mbert(자체 토크나이저)는 정상 학습 궤도, XLM-R계 250k SPM 토크나이저 모델(MiniLM·xlm-r)만 사망** → 원인 완전 규명 대신 "이 환경에서 학습되는 백본 선택"으로 전환 (SMK_* 3파전)
- 교훈: 새 백본은 반드시 512샘플 암기 스모크부터 (train_hf.py --limit 512 --epochs 10)

### SMK_* — 백본 3파전 (6k 샘플, 2ep, 07-12 밤)
- mbert-cased **0.379** / mdeberta 0.187 / klue-roberta 0.130 (동일 조건). mbert만 정상 학습 곡선.
- 단, 직후 이전 프로젝트(dacon/) 발견으로 **인코더 노선 자체가 폐기** — Qwen3-0.6B(0.7679)가 모든 인코더를 +0.03 이상 상회함이 이미 실측돼 있었음. 이 레이스는 "이 환경에서 학습되는 모델" 판별용 기록으로만 의미.

### E100 — Qwen3-0.6B full-data 재학습 (진행 중, 07-12 밤 발사)
- 가설: 이전 프로젝트 제출은 전부 80% fold 모델 → 100% 데이터 재학습으로 +0.003~0.008
- 구성: dacon/work/train_full.py --tag qwen3_full3ep (검증 레시피 그대로, eval 없음, 3ep cosine). 6,561스텝, 저장 `dacon/artifacts/models/qwen3_full3ep_full_best`
- 근거: E000에서 CV(80%) 0.426 vs 주최측 zip(100%) LB 0.436 오프셋 실측, Qwen CV↔LB 갭 0
- 주의: full-data 모델은 정직한 CV가 없음 — 판정은 서버 LB로만. **LB 확인 전 기존 submit.zip(0.7677) 덮어쓰기 금지**
- 결과: (아침에 기록)

### R101~R103 — Qwen3×XLM-R 결합 한계 실험 (07-12 저녁, GPT 세션 · OOF 기반 CPU)
- 배경: oracle 선택기(둘 중 하나라도 맞으면 정답) macro **0.8028** / either-correct 80.06% — 다양성 안에 0.80 존재 (07-12 재현 검증 완료)
- R101 2단계 로지스틱 스태킹(logits+turn/meta, 그룹 CV): **0.7709 (+0.0008)** — 기각
- R102 Qwen 예측 클래스 조건부 XLM-R 재판정: 최고 조합 **+0.00004** — 기각
- R103 TF-IDF char3-5+LinearSVC nav 전문가 오버라이드: **0.7679 → 0.7208** — 기각 (표면 전문가가 문맥 정답 파괴)
- 결론: 결합·후처리로는 oracle 회수 불가 → 더 강한 단일 표현(1.7B 교사 증류) 경로 채택. 주의: 이 실험들은 스크립트 미보존(재현 시 재작성 필요)

### 템플릿 (복사용)

### EXXX — 제목
- 가설:
- 변경점 (vs 직전):
- 구성: 모델 / 입력 직렬화 / max_len / epochs / lr / class weight / seed
- CV: fold별 [ , , , , ] → mean±std
- threshold 튜닝 후 CV:
- LB:
- 결론 / 다음 액션:
