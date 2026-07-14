# LB 0.80 도전 로드맵 (2026-07-12)

## 현재 기준

- 현재 최고 서버 점수: `0.7677217115` (`submit.zip`, Qwen3-0.6B fold0, 9분 31초).
- Qwen3 5-fold OOF: `0.7701255471`.
- 목표: LB `0.8000` (현재 대비 `+0.0323`). 현재 본선권(12위) 기준은 `0.79307` (07-12 20시 리더보드 실측, 20위 0.79131).
- 2026-07-12 현재 전체 70,000건 3epoch 최종 모델 학습 중. 이 모델은 표준 최종화 카드이지만, 단독으로 +0.03을 만들 가능성은 낮음.

## 7/12 새 검증 결과

- Qwen3와 XLM-R 중 하나라도 맞는 비율은 80.06%, 정답을 아는 이상적 모델 선택기의 Macro-F1은 `0.802756`임. 즉 현재 모델들의 오류 다양성 안에 0.8 상한은 존재함.
- 하지만 실제 사용 가능한 결합은 낮음.
  - 단순 확률 블렌드: `0.7737`.
  - Qwen/XLM-R logits + turn/meta 2단계 로지스틱 결합기, 그룹 교차검증: `0.770905` (Qwen 대비 `+0.000779`).
  - Qwen 예측 클래스별 선택적 XLM-R 재판정: 최고 조합 `+0.000041`. 조건부 블렌드 가치 없음.
- TF-IDF char 3-5gram + LinearSVC nav 전용 전문가를 Qwen의 `glob/grep/list/read` 예측에 적용:
  - Qwen fold0 `0.767924` -> `0.720805`로 크게 하락. 선형 표면 전문가 경로 종료.
- 결론: 기존 모델 조합, 후처리, 값싼 전문가로는 +0.03을 만들 수 없음. 더 좋은 단일 표현 모델 또는 강한 교사에서 단일 학생으로 지식 압축이 필요함.

## 실행 우선순위

### 1. Qwen3-Embedding-0.6B 분류 파인튜닝

- 최우선 신규 실험. 기존 Qwen3-0.6B-Base와 같은 0.6B/28층 계열이라 제출 용량과 추론비용이 거의 동일함.
- 공식 모델은 100개 이상 언어, 프로그래밍 언어, code retrieval와 text classification용으로 후학습되어 있음. 현재 Base보다 이 태스크에 맞는 출발점일 가능성이 높음.
- 1차는 V2 직렬화와 기존 3ep 레시피를 그대로 두고 checkpoint만 교체하여 fold0 비교함.
- 게이트: fold0 `0.767924` 대비 최소 `+0.003`, 권장 통과선 `0.773+`.
- 통과하면 5-fold가 아니라 바로 전체 70k 최종 학습을 먼저 제출해 LB를 확인함. instruction 추가는 checkpoint 효과를 확인한 뒤 별도 실험함.
- 예상 시간: 다운로드/호환 스모크 20~40분, fold0 약 3.5시간, 전체 학습 약 4.2시간.

### 2. 혼동집합 강조 손실

- 추론 구조와 비용을 바꾸지 않고 학습 loss만 수정함.
- 기존 14-class CE에 혼동집합 내부 조건부 CE를 추가함.
  - nav: `glob_pattern, grep_search, list_directory, read_file`
  - verify/execute: `lint_or_typecheck, run_bash, run_tests`
  - dialogue/plan: `ask_user, plan_task, respond_only`
  - modify: `apply_patch, edit_file, write_file`
- true class가 속한 집합 내부 logits만 다시 softmax하여 `CE_total = CE14 + lambda * CE_group`로 학습함. 별도 head가 없어 제출 모델 형식과 추론시간은 그대로임.
- lambda 0.2/0.4 중 fold0 한 번으로 판정. 기대 이득은 작지만 현재 약점에 직접 대응함.

### 3. OOF 다중교사 지식 증류

- Qwen3 5-fold와 XLM-R 5-fold OOF soft logits를 교사 목표로 사용함.
- 학생은 단일 Qwen3-Embedding 또는 Qwen3-Base 0.6B이며, hard-label CE와 temperature KL을 함께 사용함.
- 예시: `loss = 0.7 * CE(y) + 0.3 * T^2 * KL(student/T, teacher/T)`, T=2 또는 3.
- 현재 정직한 교사 블렌드가 `0.7737`이라 단독 기대폭은 제한적이지만, 추론비용 없이 앙상블의 클래스 관계를 학생에 전달할 수 있음.
- fold0 게이트 `+0.002`; 미달이면 전체 학습하지 않음.

### 4. Qwen3-1.7B 교사 -> 0.6B 학생 증류

- 0.8을 실제로 노릴 수 있는 가장 큰 카드. 1.7B는 제출하지 않고 로컬 학습/soft-label 생성에만 사용하므로 1GB/10분 제한을 받지 않음.
- QLoRA 또는 제한적 full/LoRA 분류-head 방식으로 fold0 교사를 먼저 학습함. 최신 비교에서도 작은 Qwen3는 생성식 라벨보다 classification-head 파인튜닝이 2~3% 우수하다는 결과가 있음.
- 교사 fold0가 최소 `0.785`, 권장 `0.79+`일 때만 70k logits 생성과 0.6B 증류로 진행함.
- 예상 총시간: 교사 fold0 8~14시간, logits 생성 1~3시간, 학생 3.5~4.5시간. 실패 게이트를 두어 남은 시간을 보호함.

### 5. 보조 카드

- 외부 API/모델로 혼동 클래스와 turn0-1의 라벨 보존 paraphrase 생성. 대회 규칙상 법적 제한 없는 외부 데이터/API 활용은 가능하지만 출처 명시가 필수임. 분포 불일치 위험이 있어 1~4 이후에만 진행함.
- 제출 추론 코드는 이미 길이 정렬 + 동적 token-budget batching을 사용함. 단순 정렬 최적화는 미개척 카드가 아님. T4 16GB용 token budget 확대는 최종 모델에서 서버 프로브 가치가 있으나 점수 자체를 올리지는 않음.

## 현실적 점수 판단

- 전체 70k 재학습만 성공: 대략 0.77대 초중반 예상.
- Embedding checkpoint가 유효하고 혼동 손실/증류까지 통과: 0.78~0.79대 가능. **단 본선권이 0.79307이므로 이 시나리오도 본선권 미달일 수 있음 — 0.79를 실제로 넘으려면 카드 대부분이 상단 적중해야 함.**
- 1.7B 교사가 0.79 이상이고 증류 효율이 좋을 때: 0.80 도전 가능.
- 기존 후처리, soup, max_len384, V3 직렬화, 단순 블렌드, 조건부 XLM-R, TF-IDF 전문가는 재시도하지 않음.

## 근거 자료

- 대회 규칙: https://dacon.io/competitions/official/236694/overview/rules
- Qwen3-Embedding-0.6B 공식 모델: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Qwen3 Embedding 공식 저장소: https://github.com/QwenLM/Qwen3-Embedding
- Qwen3 Embedding 논문: https://arxiv.org/abs/2506.05176
- 작은 Qwen3 classification-head 비교: https://arxiv.org/abs/2607.03801
- 강한 교사 지식 증류: https://proceedings.neurips.cc/paper_files/paper/2022/hash/da669dfd3c36c93905a17ddba01eef06-Abstract-Conference.html

