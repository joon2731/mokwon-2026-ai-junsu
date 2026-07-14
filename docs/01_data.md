# 데이터 스키마 & EDA

> 근거: `data/train.jsonl` 70,000샘플 전수 스캔 (2026-07-12). 재현 스크립트는 필요시 `src/eda.py`로 정리 예정.

## 파일

| 파일 | 내용 |
|---|---|
| `data/train.jsonl` | 70,000샘플. 한 줄 = JSON 객체 한 개 |
| `data/train_labels.csv` | `id,action` — train 정답 |
| `data/test.jsonl` | **로컬 스텁 5행** (train 샘플 재사용). 실제 테스트는 서버에만 존재 |
| `data/sample_submission.csv` | `id,action` — 제출 형식/순서 기준 (로컬은 5행) |

## 샘플 스키마 (train/test 동일)

```jsonc
{
  "id": "sess_sim_20260522_024730-step_08",   // {session_id}-step_{NN}
  "session_meta": {
    "user_tier": "enterprise",                 // enterprise | pro | free
    "language_pref": "en",                     // ko | en | mixed
    "workspace": {
      "language_mix": {"py": 0.82, "yaml": 0.1, ...},  // 언어별 비율 dict
      "loc": 15326,
      "git_dirty": true,
      "open_files": ["src/schemas/types.py", "pyproject.toml"],  // 0개 이상
      "last_ci_status": "passed"               // passed | failed | none
    },
    "budget_tokens_remaining": 131818,
    "turn_index": 8,                           // 1부터 시작, 최대 18 관측
    "elapsed_session_sec": 720
  },
  "history": [                                 // (user, assistant_action) 쌍의 교대 배열
    {"role": "user", "content": "..."},
    {"role": "assistant_action", "name": "read_file",
     "args": {"path": "pyproject.toml"},       // 액션마다 키 다름
     "result_summary": "ok; read pyproject.toml (173L)"}
  ],
  "current_prompt": "bundle's fine. now run the profile tests ..."
}
```

## 레이블 — 14개 클래스 (train 분포)

| action | count | | action | count |
|---|---|---|---|---|
| edit_file | 11,171 | | list_directory | 4,329 |
| grep_search | 9,912 | | ask_user | 2,701 |
| read_file | 9,257 | | plan_task | 2,679 |
| glob_pattern | 5,284 | | lint_or_typecheck | 2,283 |
| respond_only | 5,178 | | write_file | 1,481 |
| run_bash | 5,068 | | web_search | **1,273** |
| apply_patch | 4,823 | | run_tests | 4,561 |

- 불균형 최대 8.8:1 (edit_file vs web_search). **macro-F1이므로 희소 클래스(web_search, write_file, lint_or_typecheck, plan_task, ask_user)의 F1이 점수를 좌우한다.**

## 핵심 구조적 사실

1. **세션 구조**: 70,000샘플 = **9,429 세션** × 스텝(최대 18). 같은 세션의 스텝들은 history가 서로 포함관계 → **랜덤 split 시 leakage. CV는 session_id(= id에서 `-step_NN` 제거) 기준 GroupKFold 필수.**
   - Qwen 계열 실험은 이전 프로젝트 스플릿(`artifacts/train_prepared.parquet`의 fold 컬럼)을 사용 — 기존 OOF/모델과 정합 유지. da2의 `artifacts/splits.csv`는 자체 실험(E000~)용.
1-b. **두 개의 데이터 소스**: id 프리픽스가 `sess_sim_20260522_*` (64,975행, 92.8%)와 **`sess_au_*` (5,025행 / 1,099세션, 7.2%)** 로 나뉜다. au는 레이블 분포가 크게 다르고(read_file 25.7% 등, L1거리 0.42) 히스토리가 짧다. **테스트에선 au 비중 ~15%로 추정**(au prior 제출 실험의 이득이 OOF 예측의 2.4배였던 것에서 역산) → au 턴버킷 prior 보정이 LB +0.0055.
1-c. **숨은 테스트셋 실측**(이전 프로젝트 제출로 확정): **~30k 샘플, 세션당 1스텝만, train 세션과 완전 분리** (train↔test 세션 overlay 프로브 명중 0건 × 2회).
2. **history는 최근 6쌍(12엔트리)으로 캡**: 길이 분포 {0, 2, 4, ..., 12}, turn_index≥7이면 전부 12. user/assistant_action 엄격 교대 (각 242,532개).
3. **history의 action name은 13종만 등장 — `respond_only`는 history에 없음.** ✅ 확인 완료(07-12): **respond_only 턴은 데이터상 항상 세션의 마지막 스텝**이라 다음 스텝이 존재하지 않는다. 비종결 턴은 예외 없이 (prompt, action) 쌍이 다음 스텝 history에 기록됨 (58,326/58,326 확인, 누락 0).
4. **언어**: language_pref = ko 45,028 (64.3%) / en 17,802 (25.4%) / mixed 7,170 (10.2%). 단 **문자 구성 기준으로는 영문 58% / 한글 37%** (코드·경로 혼재, 이전 프로젝트 측정) → 코드 이해력 있는 다국어 모델이 유리.
5. **current_prompt는 짧다**: 평균 61자, 최대 346자, 빈 값 없음. history 발화도 유사하게 짧음 → 6쌍 히스토리 + 메타를 넣어도 512토큰 내 수용 가능할 것 (토큰 길이 분포 측정 필요 ⚠️).
6. user_tier: pro 37,733 / free 20,948 / enterprise 11,319. last_ci_status: passed 28,035 / failed 22,623 / none 19,342.
7. 합성 데이터(시뮬레이터 생성). sim 소스는 전부 `20260522` 날짜 프리픽스, au 소스는 별도 체계(1-b 참조). train/test 분포 일치는 이전 프로젝트 제출로 검증됨(τ 프로브의 OOF 예측과 LB 변화 일치, Qwen CV↔LB 갭 ≈ 0).

## 신호 강도 사전 측정 (train 전체, 인샘플 상한 참고용)

| 규칙 | macro F1 |
|---|---|
| 항상 최빈 클래스(edit_file) | 0.020 |
| P(label \| 직전 action) argmax | 0.137 |
| 리더보드 상위권 (모델) | ~0.79 |

- 직전 액션→다음 액션 전이만으로는 약함 (최고 집중도도 write_file→edit_file 40%). **주 신호는 current_prompt 텍스트**이고, history·meta는 보조 신호.
- 눈에 띄는 전이: edit_file→run_tests(23%), apply_patch→lint_or_typecheck(17%), lint_or_typecheck→apply_patch(25%), list_directory→read_file(25%), 첫 턴(history 없음)→list_directory(20%)/read_file(17%)/plan_task(12%).

## 미해결 질문 (EDA TODO)

- [x] 직렬화 후 토큰 길이 분포 → mdeberta 기준 p99 508(max_len 512가 99.1% 커버). **Qwen3 토크나이저 + V2 직렬화 기준: p50 271 / p99 520 / max 613, >512는 1.28%** (이전 프로젝트 측정) → max_len 512 확정.
- [x] respond_only 직전 턴의 history 표현 → 위 3번: respond_only는 항상 세션 종결 스텝.
- [x] 세그먼트별 성능 (이전 프로젝트, Qwen3 5-fold OOF) → sim 0.760 / au 0.715 / **turn0-1 0.42 (정보 한계로 판정 — 더 강한 모델로도 안 뚫림)** / turn2+ 0.765. 약한 클래스: list 0.47 · read 0.57 · grep 0.62 · ask 0.66 · glob 0.66 · lint 0.69.
- [x] args/result_summary 기여도 → V2 직렬화가 args 키 10종 전부 포함, 전처리 감사에서 "사실상 무손실" 판정. 추가 여지는 표면 단서 태그(verify-cue) 뿐.
- [x] 실제 테스트셋 크기 → **~30k** (서버 실행시간 역산으로 확정).
