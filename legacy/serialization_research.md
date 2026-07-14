# 직렬화 개선 리서치 메모

작성일: 2026-07-09  
목표: Qwen3-0.6B sequence classification 기준으로 입력 직렬화만 바꿔 Macro-F1 개선 가능성을 찾기

## 결론

가장 먼저 해볼 실험은 `current_prompt`를 입력 끝에도 한 번 더 배치하는 **tail anchor 직렬화**가 맞다.

이유는 두 가지다.

1. Qwen3ForSequenceClassification은 마지막 non-pad 토큰 위치의 hidden state로 분류한다.
2. 장문 컨텍스트 연구에서는 중요한 정보가 앞이나 끝에 있을 때 더 잘 쓰이고, 중간에 있으면 성능이 떨어지는 경향이 반복적으로 보고됐다.

현재 V2 직렬화는 다음 형태다.

```text
<current_prompt>
[META] ...
[HIST] newest-first history
```

즉, 모델의 최종 분류 위치가 대체로 history 끝부분에 걸린다. 다음 행동 예측에서 제일 중요한 `current_prompt`가 앞에만 있고, 분류 pooling 위치와 멀다. 그래서 아래 형태가 1순위다.

```text
[CUR] <current_prompt>
[META] ...
[HIST] compact newest-first history
[NEXT_ACTION_FROM] <current_prompt>
```

## 근거

### 1. Qwen3 분류 헤드는 마지막 토큰을 쓴다

Hugging Face Transformers v4.51.3의 Qwen3 구현에서 `Qwen3ForSequenceClassification`은 last non-pad token의 logits를 `pooled_logits`로 사용한다. 공식 GPT2 문서에도 causal model의 sequence classification은 마지막 토큰을 사용한다고 설명되어 있고, Qwen3 코드도 같은 구조다.

영향:

- 입력 끝에 어떤 정보가 오느냐가 중요하다.
- 현재처럼 `[HIST]`가 끝이면 마지막 history 조각 중심으로 분류된다.
- 끝부분에 현재 지시문 또는 예측 타깃 마커를 넣는 것이 구조적으로 맞다.

### 2. 중요한 정보는 앞/끝 배치가 유리하다

Lost in the Middle 논문은 긴 입력에서 관련 정보가 중간에 있을 때 성능이 떨어지고, 앞이나 끝에 있을 때 성능이 높은 경향을 보였다.

영향:

- `current_prompt`를 맨 앞에 두는 기존 설계는 맞다.
- 다만 Qwen3 분류 헤드 특성상 끝에도 다시 놓는 편이 더 안전하다.
- 중간 `[META]`, `[HIST]` 사이에 중요한 신호가 묻히지 않게 해야 한다.

### 3. 포맷과 순서는 실제로 성능을 흔든다

Calibrate Before Use, Fantastically Ordered Prompts, prompt formatting 연구들은 prompt 형식, 예시 순서, 템플릿 선택이 성능을 크게 바꿀 수 있음을 보인다.

영향:

- 직렬화는 단순 문자열 꾸미기가 아니라 모델 입력 분포 자체를 바꾸는 실험이다.
- JSON/YAML/Markdown 같은 전체 포맷 전환도 후보가 될 수 있지만, 지금은 리스크가 크다.
- 우선은 기존 tagged text 형식을 유지하고, 위치/압축/마커만 바꾸는 실험이 낫다.

### 4. 구조화 데이터는 순서 편향이 생긴다

TableFormer, TURL 계열 연구는 표/구조 데이터를 단순 linearization하면 순서와 위치 편향이 생기며, 구조를 명시하는 방식이 유리할 수 있음을 보인다.

영향:

- `[META]`, `[HIST]`, `[LAST_USER]`, `[LAST_ACTION]` 같은 명시적 태그는 유지하는 편이 좋다.
- 필드 순서는 label signal이 강한 것부터 고정하는 편이 좋다.
- 완전한 JSON 전환보다 compact tag format이 현재 모델/토큰 예산에는 더 맞다.

## 로컬 검증

Qwen3 tokenizer 기준 현재 V2 직렬화 길이:

```text
전체 70,000건
mean 265.1 / p50 271 / p75 368 / p90 421 / p95 454 / p99 520 / max 613
>384 tokens: 19.84%
>512 tokens: 1.28%

turn0-1 9,000건
mean 80 / p50 78 / p90 98 / p95 104 / max 146
>384 tokens: 0%
>512 tokens: 0%

turn2+ 61,000건
mean 292.4 / p50 301 / p90 427 / p95 462 / p99 524 / max 613
>384 tokens: 22.77%
>512 tokens: 1.47%
```

구성요소별 평균:

```text
current_prompt: mean 26.3 tokens
meta:           mean 50.0 tokens
history:        mean 187.0 tokens
tail repeat:    mean 30.3 tokens
```

해석:

- `max_len=512`에서는 현재도 잘림이 1.28%뿐이라 길이 문제가 크지 않다.
- `max_len=384`에서는 약 20%가 잘려서 성능 하락 가능성이 크다.
- `current_prompt`를 끝에 반복하면 512 기준 잘림은 약 3.5%로 증가하지만 감당 가능한 수준이다.
- 384 기준에서는 잘림이 약 28.5%까지 올라가므로, tail anchor를 쓰려면 history 압축을 같이 해야 한다.
- turn0-1은 길이가 짧아서 잘림 문제가 아니다. 이 구간은 정보 부족/표현 부족 문제로 봐야 한다.

## 실험 후보

### 1순위: V3 tail anchor

형태:

```text
[CUR] <current_prompt>
[META] tier=... lang=... ci=... dirty=... turn=... loc=... budget=... el=...
[HIST] ...
[NEXT_ACTION_FROM] <current_prompt>
```

기대:

- Qwen3 last-token pooling 구조와 맞는다.
- Lost-in-the-middle 리스크를 줄인다.
- 구현 범위가 작고 기존 학습 코드와 잘 맞는다.

주의:

- `max_len=512`로 먼저 검증하는 것이 맞다.
- 384에 바로 붙이면 history truncation이 커질 수 있다.

### 2순위: turn bucket marker

형태:

```text
[TURN0]
[CUR] ...
```

또는

```text
tb=turn0|turn1|turn2_3|turn4p
```

기대:

- turn0-1 정확도가 낮은데, 이 구간은 길이 문제가 아니므로 상태를 더 직접적으로 알려주는 편이 낫다.
- `turn_index` 숫자보다 bucket/tag가 작은 모델에 더 잘 먹힐 가능성이 있다.

### 3순위: last user / last action 명시

형태:

```text
[LAST_USER] ...
[LAST_ACTION] edit_file path=... -> ...
[HIST] ...
```

기대:

- 현재 history는 newest-first지만 마지막 사용자/도구 행동이 따로 요약되어 있지 않다.
- 다음 행동 예측에서는 직전 사용자 요청과 직전 action name이 매우 강한 단서다.

주의:

- result_summary를 길게 넣으면 노이즈와 토큰 비용이 커진다.
- 마지막 1개 action은 자세히, 오래된 action은 `name(args)` 정도로 압축하는 편이 좋다.

### 4순위: compact history

형태:

```text
[HIST] u: ... || a: read_file path=... || u: ... || a: apply_patch path=...
```

변경:

- history turn 수를 그대로 두되 오래된 action의 `result_summary`는 제거
- assistant action args는 path/pattern/cmd 정도만 유지
- user text는 최근 1~2개만 200자, 나머지는 120자 이하

기대:

- tail anchor 비용을 상쇄한다.
- 384 실험과도 조합 가능하다.

### 5순위: path/code cue feature

형태:

```text
[META] ... open_ext=py,md open_base=train.py,featurize.py prompt_has_path=1 prompt_has_code=0
```

기대:

- `read_file`, `grep_search`, `edit_file`, `run_tests`, `lint_or_typecheck` 같은 클래스 구분에 도움될 수 있다.

주의:

- 너무 많은 파생 피처는 overfit 위험이 있다.
- 우선은 open file 확장자/top basename 정도만 작게 넣는 게 낫다.

### 6순위: action inventory

형태:

```text
[ACTIONS] apply_patch ask_user edit_file glob_pattern grep_search ...
```

기대:

- prompt/verbalizer 연구 관점에서는 label semantics를 넣는 효과가 있을 수 있다.

주의:

- 현재 모델은 label text를 생성하는 게 아니라 classification head로 분류한다.
- 효과는 불확실하고 토큰 비용 대비 우선순위가 낮다.

## 하지 않는 편이 좋은 것

- session_id, sample id 같은 직접 ID성 정보 추가: local CV 과적합 위험이 크다.
- full history를 더 길게 추가: 384/512 모두 이득보다 truncation과 노이즈 위험이 크다.
- JSON/YAML 전체 전환을 첫 실험으로 사용: 연구상 포맷 영향은 있지만, 현재는 기존 V2와 너무 많이 달라져 원인 분석이 어렵다.
- 후처리 bias/LA/동사룰 재시도: 이미 cross-fit에서 효과가 낮거나 음수였다.

## 추천 실행 순서

1. `work/featurize.py`를 V3 실험 파일로 분기한다.
2. `max_len=512`, Qwen3 fold0 3ep로 `tail anchor + turn bucket + compact history`를 한 번에 최소 변경으로 검증한다.
3. fold0 CV가 기존 `0.767924`보다 높으면 5-fold가 아니라 우선 단일 제출 후보 `submit3.zip`로 만든다.
4. fold0에서 `+0.002` 이상이면 추가 fold 또는 5-fold OOF 확인 가치가 있다.
5. fold0에서 같거나 낮으면 tail anchor 단독/compact 단독으로 쪼개서 원인만 확인한다.

## V3 초안

```text
[CUR] {prompt}
[META] tb={turn_bucket} tier={tier} lang={lang} ci={ci} dirty={dirty} loc={loc} budget={budget} el={elapsed} open={open_files} lm={language_mix}
[LAST_USER] {last_user}
[LAST_ACTION] {last_action_compact}
[HIST] {compact_history}
[NEXT_ACTION_FROM] {prompt}
```

현재 기준으로는 `max_len=512`가 맞다. 384는 제출 결과가 좋게 나오면 별도 분기로 보되, V3는 history 압축 없이 바로 384에 얹으면 위험하다.

## 참고자료

- Lost in the Middle: How Language Models Use Long Contexts: https://arxiv.org/abs/2307.03172
- Hugging Face GPT2ForSequenceClassification docs: https://huggingface.co/docs/transformers/model_doc/gpt2
- Hugging Face Transformers Qwen3 v4.51.3 source: https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py
- Calibrate Before Use: Improving Few-Shot Performance of Language Models: https://arxiv.org/abs/2102.09690
- Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity: https://arxiv.org/abs/2104.08786
- Does Prompt Formatting Have Any Impact on LLM Performance?: https://arxiv.org/abs/2411.10541
- TableFormer: Robust Transformer Modeling for Table-Text Encoding: https://arxiv.org/abs/2203.00274
- TURL: Table Understanding through Representation Learning: https://arxiv.org/abs/2006.14806

