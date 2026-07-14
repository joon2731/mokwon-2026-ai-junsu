# DACON 236694 — 진행 로그 (AI 에이전트 다음 행동 예측)

## [2026-07-12] 목표 LB 0.80 신규 경로 조사

- 상세 실행안: `ROADMAP_080.md`.
- 현재 Qwen3 OOF `0.770126`, 서버 LB `0.767722`; 목표 0.80까지 약 `+0.0323` 필요.
- Qwen3/XLM-R 모델선택 oracle은 `0.802756`이지만 실제 블렌드 `0.7737`, 2단계 로지스틱 결합 `0.770905`(+0.000779), 예측 클래스 조건부 XLM-R `+0.000041`로 기존 두 모델 결합만으로는 0.8 불가.
- 신규 CPU 검증: TF-IDF+LinearSVC nav 전문가를 Qwen fold0에 붙이면 `0.767924 -> 0.720805`. 표면형 전문가는 사용하지 않음.
- 최우선 신규 카드: `Qwen3-Embedding-0.6B`를 기존 V2/3ep 레시피로 fold0 파인튜닝. 공식적으로 multilingual/code retrieval/text classification 후학습 모델이며 Base와 같은 0.6B 계열이라 제출 비용을 유지할 가능성이 높음.
- 다음 카드: 혼동집합 내부 조건부 CE(추론비용 0) -> OOF 다중교사 증류 -> Qwen3-1.7B 교사를 0.6B로 증류. 1.7B는 제출하지 않고 학습 교사로만 사용함.
- 대회 규칙 재확인: 법적 제한 없는 사전학습 모델, 외부 데이터, API 활용 가능. 주요 외부 요소의 출처/활용 범위 기재 필수. 예선 종료 2026-07-15 10:00.

## [2026-07-12] 진행 중 - Qwen3 전체 70k 3ep 최종 모델 학습

- 목적: 기존 최고 Qwen3 fold0 모델이 학습에 사용하지 않은 검증 fold 14,000건까지 포함하여, 전체 학습 데이터 70,000건으로 최종 단일 모델을 학습함.
- 학습 코드: `work/train_full.py` (2026-07-12 생성). fold holdout/eval/OOF 없이 전체 데이터로 학습하고 마지막 3epoch 모델을 저장함.
- 설정: V2 `train_prepared.parquet`, Qwen3-0.6B-Base, max_len=512, 3epoch, bs=8, grad_accum=4, lr=2e-5, warmup=0.1, wd=0.01, weighting=sqrt, bf16, grad_ckpt, adamw_bnb_8bit, seed=42.
- 실행: Python PID `35544`, 시작 2026-07-12 18:40 KST. 로그 `artifacts/qwen3_full3ep.log`.
- 총 스텝: `6561` (70,000건 전체 데이터 x 3epoch). 2026-07-12 19:34 기준 `1426/6561` = 21.7%, epoch 약 0.65.
- 동작 확인: RTX 4070 Ti GPU util 100%, VRAM 8.4GB, 약 2.2~2.4초/step. 오류 파일 없음. 현재 추정 종료 시각 22:45 KST 전후.
- 저장 예정: `artifacts/models/qwen3_full3ep_full_best`. 에폭별 크래시 복구 체크포인트는 `artifacts/models/qwen3_full3ep_run`에 저장됨.
- 완료 후 필요한 작업: 저장 완료 확인 -> 기존 Qwen 프루닝 경로로 동치성 검증 -> `submit.zip`과 같은 V2/max_len512/au/requirements 구성으로 별도 제출본 패키징. 기존 `submit.zip`은 서버 LB 0.7677217115가 확인된 기준본이므로 새 모델의 서버 결과 확인 전에는 덮어쓰지 않음.
- 주의: 전체 데이터 모델은 정직한 OOF/CV를 새로 계산할 수 없음. 기대 근거는 fold별 검증 완료 후 전체 70k를 같은 레시피로 재학습하는 표준 최종화이며, 실제 개선 여부는 서버 LB로만 판정함.

## [2026-07-09] 완료 - Qwen3 V3 직렬화 fold0

- 실험명: `qwen3_serial_v3`
- 목적: 기존 V2(`prompt -> META -> HIST`) 대신 V3 tail anchor 직렬화로 fold0 CV 개선 확인.
- 구성: Qwen3-0.6B-Base, fold0, max_len=512, 3epoch, bs8, grad_accum4, lr2e-5, warmup0.1, wd0.01, weighting=sqrt, bf16, grad_ckpt, adamw_bnb_8bit.
- V3 구조: `[CUR] prompt -> [META] -> [LAST_USER] -> [LAST_ACTION] -> compact [HIST] -> [NEXT_ACTION_FROM] prompt`
- 기준점: 기존 Qwen3 fold0 CV `0.7679242649`, 기존 최종 LB `0.7677217115`.
- 길이 검증: Qwen3 tokenizer 기준 V3 `>512` 비율 2.02%, `>384` 비율 19.87%, turn0-1은 `>512` 0%.
- 실행 PID: runner `48376`, monitor `42900`.
- 로그: `artifacts/qwen3_serial_v3.log`, `artifacts/qwen3_serial_v3.err`, `artifacts/qwen3_serial_v3_monitor.log`, `artifacts/qwen3_serial_v3_status.txt`
- 상태: 2026-07-09 02:05 KST 시작, 05:43 KST 종료.
- 결과: fold0 CV `0.766900`, 기존 Qwen3 fold0 CV `0.7679242649` 대비 `-0.001024`.
- epoch별 CV: epoch1 `0.7073`, epoch2 `0.7613`, epoch3 `0.7669`.
- 결론: V3 tail anchor + last user/action + compact history 조합은 기존 V2보다 약간 낮음. 자동 조건대로 `submit3.zip` 패키징은 진행하지 않음.
- 산출물: `artifacts/models/qwen3_serial_v3_fold0_best`, `artifacts/oof/qwen3_serial_v3_fold0.npz`.
- 현재 최종 제출 후보는 계속 `submit.zip` 유지.
- 추가 판단: 사용자가 서버 분포 확인용 제출을 요청하여 V3도 제출 패키지 생성.
- `prune_qwen.py qwen3_serial_v3_fold0_best --featurizer featurize_v3 --max_len 512` 완료. 원본/pruned 동치성 max|Δlogit|=0, argmax 512/512.
- `submit3.zip` 생성 완료: Qwen3 V3 fold0 pruned + max_len512 + au + requirements 설치형 + V3 featurize 동봉. zip size `838,529,123` bytes, `python -m zipfile -t submit3.zip` OK.
- 주의: V3는 CV가 V2보다 낮으므로 기본 최종본은 여전히 `submit.zip`. `submit3.zip`은 서버 분포 확인용 1회 제출 후보.
- au 관련 정리: "Qwen은 au가 필요 없음"이 아니라 "Qwen은 au를 모델 자체가 XLM-R보다 잘 맞춰서 au-aware 재학습 우선순위가 낮음"이 정확함. 기존 `au_bias.json` prior는 Qwen3 fold0 OOF에서도 `0.767924 -> 0.769731`로 `+0.001807` 개선, V3 fold0에서도 `0.766850 -> 0.768629`로 `+0.001779` 개선되어 제출 패키지에는 유지.
- 제출 결과: `submit3.zip`은 서버에서 10분 초과로 실패. CV도 V2보다 낮고 런타임도 실패했으므로 V3 직렬화 패키지는 사용하지 않음. 최종 후보는 계속 `submit.zip`.

## [2026-07-09] 직렬화 개선 딥 리서치 요약

- 상세 정리 파일: `serialization_research.md`
- 핵심 결론: Qwen3ForSequenceClassification은 마지막 non-pad token을 pooling해서 분류하므로, 현재 V2처럼 끝이 `[HIST]`로 끝나는 구조는 개선 여지가 큼.
- 1순위 실험: `max_len=512` 기준 V3 tail anchor. 형태는 `[CUR] prompt -> [META] -> [LAST_USER]/[LAST_ACTION] -> compact [HIST] -> [NEXT_ACTION_FROM] prompt`.
- 근거: Hugging Face Qwen3 v4.51.3 source, GPT2 sequence classification docs, Lost in the Middle, prompt formatting/order sensitivity 논문들.
- 로컬 길이 검증: V2는 512 초과 1.28%, 384 초과 19.84%. prompt tail repeat 추가 시 512 초과는 약 3.5%라 감당 가능하지만, 384는 약 28.5%까지 올라가 history 압축 없이는 위험.
- turn0-1은 길이 문제가 아님. token length가 max 146으로 384/512에서 전혀 잘리지 않으므로, turn bucket marker와 last user/action 명시가 더 유효한 방향.
- 다음 작업 후보: `work/featurize.py`를 바로 덮지 말고 V3 실험용으로 분기한 뒤 fold0 3ep CV를 기존 Qwen3 fold0 0.767924와 비교.

## [2026-07-09] submit2 max_len384 제출 결과

- 제출 파일: `submit2.zip`
- 구성: Qwen3-0.6B 3ep fold0 + au + requirements 설치형, `max_len=384`
- 서버 결과: LB `0.7626078771`, 런타임 `9분 11초`
- 기존 최종 `submit.zip` 결과: LB `0.7677217115`, 런타임 `9분 31초`
- 해석: 런타임은 약 20초 줄었지만 LB가 약 `-0.0051` 하락. max_len384 단독은 최종본으로 쓰기 어렵고, 현재 최종은 계속 `submit.zip` 유지.
- 의미: 384는 속도 확보 수단으로는 확인됐지만 점수 손실이 큼. 직렬화 V3는 384가 아니라 `max_len=512`에서 먼저 검증하는 편이 맞음.

## [2026-07-08] 제출 파일명 규칙

- `submit.zip`: 현재 최종 제출본. 지금은 `submit_qwen3_req.zip`과 같은 Qwen3-0.6B 3ep fold0 + au + `requirements.txt` 설치형 제출본이며, LB 0.7677217115 / 런타임 9:31로 확인됨.
- `submit2.zip`: 현재 진행 중인 Qwen3 `max_len=384` 실험이 통과하면 이 이름으로 패키징함.
- `submit3.zip`: 이후 직렬화 변경(V3) 실험이 통과하면 이 이름으로 패키징함.
- 기존 긴 이름(`submit_qwen3_req.zip`, `submit_qwen3_soup.zip` 등)은 실험 추적용으로만 남기고, 실제 제출 후보는 `submit`, `submit2`, `submit3` 순서로 관리함.

## [2026-07-08] 진행 중 - Qwen3 max_len 384 fold0 실험

- 목적: Qwen3 단일 모델 입력 길이를 512에서 384로 줄였을 때 CV 손실과 추론 런타임 절감 가능성을 확인함.
- 이유: 현재 최고 제출은 9:31/10분으로 여유가 약 30초뿐임. Qwen3 x XLM-R 블렌드는 OOF 0.7737로 좋지만 런타임/용량 제약에 걸림. max_len 384가 점수를 크게 깎지 않으면 블렌드 여지가 생김.
- 태그: `qwen3_len384`
- 설정: Qwen/Qwen3-0.6B-Base, fold0, max_len 384, 3epoch, bs8, grad_accum4, lr2e-5, warmup0.1, wd0.01, weighting=sqrt, bf16, grad_ckpt, adamw_bnb_8bit.
- 기준 비교: 기존 `qwen3_smoke` fold0 max_len512 CV 0.7679242649 / LB 0.7677217115 / 제출 런타임 9:31.
- 판정 기준: fold0 CV 손실이 -0.002~-0.004 이내면 계속 볼 가치 있음. -0.008 이상이면 블렌드 이득으로 회복이 애매함.
- 상태: 학습 완료. 시작 2026-07-08 21:37 KST, 종료 2026-07-09 00:35 KST 전후. 산출물 `artifacts/models/qwen3_len384_fold0_best`, OOF `artifacts/oof/qwen3_len384_fold0.npz`.
- 결과: epoch1 CV 0.7053, epoch2 CV 0.7591, epoch3/final CV **0.7639**. 기존 max_len512 fold0 CV 0.767924 대비 **-0.0040**.
- 판정: 점수 손실이 작지는 않지만 사전 기준(-0.002~-0.004 이내) 경계선. 런타임 절감 확인용 `submit2.zip` 후보로 패키징 진행.
- 패키징: `qwen3_len384_fold0_best_pruned` 생성 완료. 원본/프루닝 동치성 max|Δlogit|=0, argmax 512/512. `submit2.zip` 생성 완료(`--max_len 384 --req_tf451 --au`), zip 839.25MB, zip test OK, infer_config max_len=384 확인.
- 첫 시도는 `Qwen/Qwen3-0.6B-Base` 허브 이름으로 로드하다가 네트워크 차단으로 모델 config 조회 실패. 로컬 경로 `pretrained\Qwen3-0.6B-Base`로 수정함.
- 확인: foreground debug `epochs=0.01`은 학습 루프/eval/save까지 정상 완주함. 모델/데이터 문제는 아님.
- 주의: Python detached launcher는 `_ctypes` DLL 로드 오류가 나서 사용하지 않음. `cmd` 래퍼 방식으로 재시작.
- 로그: `artifacts/qwen3_len384.log`, `artifacts/qwen3_len384.err`, wrapper `work/run_qwen3_len384.cmd`.

## [2026-07-08] 제출 실험 메모 - Qwen3 requirements 설치형

- 목적: 서버의 온라인 패키지 설치 단계에서 `transformers==4.51.3` 설치가 되는지 검증함. 기존 `libs/` 동봉 방식 대신 `requirements.txt` 설치 방식 사용.
- 생성 파일: `submit_qwen3_req.zip`
- 모델/설정: 기존 최고 제출과 동일한 `qwen3_smoke_fold0_best_pruned` 단일 Qwen3-0.6B 3ep fold0 + `au_bias.json` + max_len 512.
- 차이점: `libs/` 미포함, `requirements.txt`에 `transformers==4.51.3`, `tokenizers==0.21.0`, `huggingface_hub==0.30.0` 기록.
- 검증: zip 무결성 OK, 필수 파일 포함 OK, `libs/` 미포함 확인. **서버 제출에서도 정상 작동 확인됨**(패키지 설치 단계로 transformers 4.51.3 계열 설치 가능).
- 용량: zip 839.3MB. 기존 `submit_qwen3.zip` 853.1MB 대비 약 14MB 감소.
- 결과: `requirements.txt` 설치형 Qwen3 제출 경로 정상 작동 확인. 이제 Qwen3 제출은 `libs/` 동봉 방식과 서버 설치 방식 둘 다 가능함.
- 함의: `libs/`를 빼면 zip 약 14MB를 절약할 수 있음. 단, 이 절감만으로는 Qwen3+XLM-R 전체 블렌드 탑재에는 부족하므로 추가 프루닝/양자화/입력 길이 축소와 같이 봐야 함.
- 다음 행동: 향후 새 제출은 `--req_tf451` 경로를 기본 후보로 두고, 블렌드/용량 확보 실험에서 `libs/` 제거분을 같이 반영함.

> 최종 업데이트: 2026-07-08 · 우리 **LB 0.7677** 🏆 (Qwen3-0.6B 3ep fold0 단일+au+requirements 설치형, 런타임 9:31, 이전 0.7591 대비 +0.0086) · 본선컷 **0.785↑**(계속 상승 중 — **추격 안 함**) · **목표 LB 0.80 / 최소 0.79** (고정) · D-7 · **엔진=Qwen3-0.6B 확정. ✅ 3ep 5-fold 완료(7/8): OOF CV 0.7701** [0.7679/0.7717/0.7682/0.7705/0.7722]. **현재 최고 제출본=`submit.zip` LB 0.7677 / 9:31.** `submit_qwen3_soup.zip`은 LB 0.7603으로 기각. **다음 후보=블렌드/속도여유 확보 쪽, soup 재시도 금지.**
    - **[✅ 제출 결과 7/8] submit.zip → LB 0.7677 🏆 (신기록, +0.0086 vs 0.7591), 런타임 9:31.** 핵심 검증: ①**requirements 설치형 transformers 4.51.3 서버서 작동 확정**(qwen3 채점됨) ②**추론 9:31<10분 통과**(bench 15.4분은 비관적, 실측 9:31 — 단 여유 ~30초뿐) ③CV fold0 0.7679→LB 0.7677 **갭 −0.0002≈0**(Qwen 갭 재확인). **함의**: 런타임 여유 30초라 **블렌드(XLM-R +2.7분)·soup 다모델은 10분 초과 → 불가**. 단일 0.6B가 T4 실질 천장. 더 올리려면 max_len 단축으로 여유 확보 후 blend, 또는 soup.
    - **[❌ 제출 결과 7/8] submit_qwen3_soup.zip → LB 0.7602664582, 런타임 9:35.** 5-fold weight-soup는 zip/런타임 규정은 통과했지만 **성능이 기존 단일 0.7677보다 −0.0075 하락** → **기각**. 결론: fold 가중치 평균은 이 태스크/레시피에서 일반화 이득을 담지 못하고 모델을 흐림. **package_multi.py 기본값은 다시 qwen3_smoke_fold0_best_pruned(현재 최고 제출본)로 복구.**
    - **[규정 체크 7/8]** ✅1GB(853MB) ✅오프라인 ✅설치(pip無, libs언집) ✅**외부요소 기재 수정**(script.py 헤더가 XLM-R로 잘못돼있어 → Qwen3-0.6B-Base Apache-2.0 + 4.51번들 명기로 수정, zip 재생성) ✅**추론시간 9:31<10분 실측 통과**. bench_infer 로컬추정 30k=15.4분은 비관적이었음. 단, 여유는 약 30초뿐이라 추가 모델/긴 max_len은 위험.
    - **[블렌드 7/8] Qwen3×XLM-R honest = 0.7737** (Qwen3단일 0.7701 +0.0036, w=0.62/0.38, 게이트 통과). **단 Qwen3 905MB+XLM-R 200MB+libs >1GB → 현 제출 미포함**(단일). 블렌드 넣으려면 추가 프루닝/int8 필요(추가작업).
>
> **[0.80 도전 현실 — 7/7 갱신]** 엔진 탐색 종료: **Qwen3-0.6B가 T4 천장**(1.5B·9B·30B·480B·Qwen3.5 전부 속도/크기/버전 봉쇄 — 하단 '모델 사다리' 참조). Qwen3 fold0 0.7679, 5-fold OOF 투영 ~0.772 → LB ~0.77~0.78 예상. **0.79~0.80은 스트레치**(에폭+5-fold+블렌드가 다 맞아야). 상세: 하단 '🎯 0.80 로드맵' 섹션.

## ▶ 여기서 재개 (RESUME POINT)

### [7/8 Codex 이어받음] Qwen3 5-fold weight-soup 제출본 생성 → 제출 후 기각
- handoff 확인 완료: 현재 LB 0.7677217115, submit_qwen3.zip runtime 9:31, Qwen3 5-fold OOF 0.7701, Qwen3×XLM-R blend 0.7737(단 1GB/10분 벽).
- **weight-soup 작업 완료**: `work/soup_qwen3.py`로 `qwen3_smoke_fold0~4_best` 5개 checkpoint를 텐서 단위 스트리밍 평균 → `artifacts/models/qwen3_smoke_soup5_best` 생성.
- `prune_qwen.py qwen3_smoke_soup5_best` 완료 → `artifacts/models/qwen3_smoke_soup5_best_pruned`(model.safetensors 904.7MB). 원본 soup vs pruned 동치성: max|Δlogit|=0, argmax 512/512.
- python work/package_multi.py --bundle_tf451 --au --out submit_qwen3_soup.zip 완료. **submit_qwen3_soup.zip=853.156MB, zip test OK, 핵심 파일/libs/au/vocab_remap 포함.**
- soup sanity: fold0 첫 3000행 macro 0.7947(누수 포함 sanity라 정직 CV 아님, collapse 없음 확인용).
- **제출 결과**: LB **0.7602664582**, runtime **9:35** → 기존 `submit_qwen3.zip` LB 0.7677217115보다 **−0.0075**. **soup 기각**, 재제출 금지. `package_multi.py` 기본값은 현재 최고 모델인 `qwen3_smoke_fold0_best_pruned`로 되돌림.


### 🔧 재부팅 복구노트 (하드리부팅 후 세션)
- **무엇이 중단됐나**: `qwen3_smoke` = **Qwen3-0.6B-Base** fold0 스모크(신규 실험, 이전 세션에서 progress.md 갱신 전 시작). 파일 타임: 체크포인트 16:03 저장 → 마지막 로그 17:10에 **하드리부팅**(전원/OS 크래시, `.err`에 traceback 없음 — epoch2 eval 36% 진행 중 전원 끊김). 코드 버그 아님.
- **살아남은 것 / 잃은 것**: epoch1 체크포인트 `artifacts/models/qwen3_smoke_fold0/checkpoint-1750`만 생존 = **eval_macro_f1 0.7131**(epoch1). epoch2~3 유실(최종 미도달). train.py에 `--resume_from_checkpoint` 없음 → 이어하려면 코드 2줄 추가 또는 처음부터 재실행(~3h, seed 고정이라 결정적).
- **🔑 발견**: Qwen3-0.6B **epoch1 0.7131 > Qwen2.5-Coder-0.5B epoch1 0.6974 (+0.0157)** — 동일 레시피 비교. Qwen2.5는 ep1→ep3 +0.0597 상승(→0.7571)했으므로 Qwen3 완주 시 **~0.77+ 기대**(단 Qwen3-Base는 코드 특화 아님 → 불확실).
- **✅ [transformers 4.51 핀 경로 검증 완료 — 3관문 전부 통과] Qwen3 오프라인 제출 실현 가능.** (검증: scratchpad/gate_c_fixed.py, 격리 venv451=4.51+시스템torch, 리눅스휠=scratchpad/whl·whl2)
  - **Gate A(지원)**: `Qwen3ForSequenceClassification`가 transformers 4.51.3에 존재 + AutoModelForSequenceClassification 매핑 등록 ✓ (서버 4.46.3은 qwen3 미지원 확정 → 4.51 핀 필수)
  - **Gate B(번들)**: 오프라인 휠 = transformers 4.51.3(10.4MB) + tokenizers 0.21.0(3.0MB, `abi3`라 py3.11 OK) + huggingface_hub 0.30.0(0.48MB) = **총 ~14MB**. 1GB 예산 무시 가능, 로컬휠 설치 ~1분. 델타는 tokenizers/hub 2개뿐(safetensors 등 나머지는 서버 4.46.3이 이미 충족)
  - **Gate C(정확성)**: 4.51 재현 argmax vs 5.13 OOF — **XLM-R f3 99.90% / Qwen2.5 f1 99.85% (둘 다 PASS)**, Qwen3 ep1 로드·추론 정상(macro 0.729@2k, 기록 0.713 정합). **단 Qwen(2.5·3)은 5.13-저장 config 때문에 수정 2개 필수**: ① `tokenizer_config.json`의 `extra_special_tokens`(list) **제거**(4.51 `_set_model_specific_special_tokens`가 dict 기대 → 크래시; 토큰 자체는 tokenizer.json에 잔존해 무손실) ② `config.json`에 top-level `rope_theta=1000000` **복원**(없으면 4.51이 기본 10000으로 읽어 rope 붕괴 — 4.46.3 수정과 동일). **XLM-R(roberta)은 무수정.**
  - **✅ 패키저 4.51 대응 완료(학습창 중 구현)**: `package_multi.py`에 ① tokenizer `extra_special_tokens`(list) 제거 추가(rope 복원 옆, XLM-R엔 no-op·4.46.3 무해라 항상 적용) + `--bundle_tf451` 플래그(vendor/tf451 리눅스휠 3개를 `.dist-info`째 libs/로 언집). `script.py`는 libs/ 있으면 `sys.path.insert(0,libs)`로 서버 4.46.3 shadow(없으면 서버 것 사용 → 양쪽 zip 안전). 휠 영구보관 `vendor/tf451/`(14MB, 언집57.5MB→zip~20MB). 문법·구조 검증 완료. **미검증(제출 필요)**: 리눅스 .so sys.path 주입이 서버서 실제 작동하는지 → **4.51 번들+기존 0.7591 모델로 프로브 제출 1회로 확인**(같은 점수 나오면 메커니즘 OK).
  - **⚠ 1GB 예산 제약(신규)**: pruned Qwen3-0.6B ~940MB(본체 0.45B + 임베딩프루닝 47MB) → **Qwen3+XLM-R+libs ≈ 1.16GB > 1GB**. ⇒ **첫 qwen3 제출은 단일**(≈960MB, 빠름). 블렌드하려면 Qwen3 int8 or 추가 프루닝 필요.
  - **⚠ 속도 재검증 필수**: Qwen3-0.6B는 0.5B보다 ~20% 무거움. 0.5B 단일이 서버 실측 5:56(#11)이라 0.6B 단일 ~7min 예상(OK) 이나 **5-fold 앙상블은 10분 초과 → 불가**. ⇒ 제출형은 0.5B와 동일하게 **Qwen3 단일(+XLM-R 블렌드)** 구조. **fold0 완주만으로 첫 LB 판독 가능(5-fold 불필요).** 반드시 `bench_infer.py`로 T4 실측 확인 후 제출.
  - **➡ 판정: qwen3 완주 GO** — fold0 3epoch 마저(재실행 ~3h, 또는 train.py에 resume 2줄 추가해 checkpoint-1750부터 ~2h). epoch1이 이미 0.5B ep1 대비 +0.0157.
  - **[진행 중] checkpoint-1750부터 재개 학습** — PID **7316**(artifacts/qwen3_resume.pid), 로그 `artifacts/qwen3_resume.{log,err}`(loss는 stdout 버퍼링으로 지연, tqdm은 .err에 라이브). 인자: max_len512·3ep·bs8·ga4·lr2e-5·warmup0.1·wd0.01·adamw_bnb_8bit·grad_ckpt·sqrt(training_args.bin 대조 일치). **✅ 완료(7/7 20:57): fold0 MACRO-F1 = 0.7679** (Qwen2.5-Coder 0.7571 대비 **+0.0108**, Qwen2.5 5-fold OOF 0.7617도 상회). `_best` + OOF 저장됨. **⇒ "신세대>코더" 확정, 엔진=Qwen3-0.6B-Base.** 세그먼트 gain: au 0.724→0.787(+0.063)·web_search +0.057·turn2+ +0.008 / **turn0-1 0.424→0.425(+0.001)=더 좋은 모델로도 안 뚫리는 벽**. nav클러스터: list +0.020·read +0.010·grep +0.009. **Qwen3 5-fold OOF 투영 ≈ 0.772.**
    - **[⏸ 중단됨 7/7 21시, 사용자 지시] Qwen3-0.6B 5-fold 체인 보류** — `work/overnight_qwen3.py`(fold1-4, fold0와 동일레시피) 발사했다가 **사용자 요청으로 즉시 중단**(레시피 먼저 확정 목적). fold0 스모크 레시피가 튜닝 안 된 상태라 **5-fold 전에 하이퍼파라미터 결정 대기 중**. 재개 시: `python work\overnight_qwen3.py`(~3.2h/fold×4≈13h, skip-if-OOF). 튜닝안: epochs 3→4(XLM-R서 +0.0047), max_len 512→640(잘림 0%, +25%시간), lr 2e-5 유지 등.
    - **[7/7 밤] 4ep 테스트: Colab 무료T4 = 실패(24s/it, 46h) → 로컬로 전환.** 무료 Colab은 T4뿐이라 이 레시피(bs8·max512·grad_ckpt·4ep 7000스텝)엔 너무 느림(최적화해도 ~12h+, 세션 12h/유휴90분 제한에 미완). **로컬 4070Ti = 2.37s/it → ~4.6h로 T4의 1/10.** Colab 패키지(`colab_qwen3_4ep/`+zip)는 Pro A100/L4용으로 보존.
    - **[진행중 7/7] 로컬 4ep fold0 학습** — PID **2452**(artifacts/qwen3_4ep.pid), 로그 `artifacts/qwen3_4ep.{log,err}`, **PYTHONUNBUFFERED=1로 loss 실시간(모니터 fix)**. tag=`qwen3_4ep`(3ep의 qwen3_smoke와 분리). ETA ~4.3h(2.28s/it). **✅ 모니터 fix 검증됨**(PYTHONUNBUFFERED로 loss 실시간 로그 확인, step100 `loss 3.174`). 재크래시: 위 재개 명령에서 `--tag qwen3_4ep --epochs 4`로.
    - **[에폭 실험 종결 7/8]** fold0 결과: **3ep 0.7679 > 4ep 0.7616(−0.0063)**. 원인=cosine-to-zero 스케줄이 3에폭에 완전 anneal될 때 peak 최고, 에폭 늘리면 anneal 퍼져 peak↓. **에폭 늘리기 = 손해 확정.** 5ep는 같은 이유로 더 낮을 게 자명 → **사용자 판단으로 스킵**(오케스트레이터 8104 폐기). ⇒ **최종 레시피 = 3ep**(qwen3_smoke).
    - **[진행중 7/8] Qwen3 3ep 5-fold(fold1-4)** — `work/overnight_qwen3.py`, PID **23936**, 로그 `artifacts/qwen3_5fold_3ep.{log,err}`. fold0(0.7679) 있음 + fold1-4 학습중(~3.2h/fold, 4folds≈13h, skip-if-OOF). 완료 후 자동 제출준비.
    - **[진행중 7/8] finish-체인 (무인 제출준비)** — `work/finish_qwen3.py`, PID **16608**, 로그 `artifacts/finish_qwen3.{log,err}`. 5-fold 드라이버 종료 감지 → ①`blend_oof --tags qwen3_smoke`(5-fold CV 로그) → ②`prune_qwen qwen3_smoke_fold0_best` → ③`package_multi --bundle_tf451 --au --out submit_qwen3.zip`(Qwen3 3ep fold0 단일, package_multi MODELS 이미 설정). **결과물: `submit_qwen3.zip`(≈960MB) + blend 로그의 5-fold CV.** ⚠첫 업로드=4.51 메커니즘 프로브. **사용자 지시로 여기서 멈춤 — 결과 나빠도 추가작업 없음.**
    - **⚙️ 크래시 시 복구(현재)**: `python work\overnight_qwen3.py`(5-fold 마저) → 완료되면 `python work\finish_qwen3.py`(제출준비). 둘 다 skip-if-OOF.
    - (구 단순 5ep 체인 chain_5ep_after_4ep.py는 오케스트레이터로 대체·종료함)
    - **⚙️ 재크래시 시 재개(현재 기준, 복붙)**: **오케스트레이터만 다시 실행**하면 됨 — 완료된 fold은 skip, 진행중이던 fold은 최신 checkpoint에서 자동 재개(train() 개선됨):
      ```
      python work\orchestrate_qwen3.py
      ```
      (단독 fold을 수동 재개하려면: `python work\train.py ...동일레시피... --tag <qwen3_4ep|qwen3_5ep|qwen3_smoke> --epochs <4|5|3> --resume_from_checkpoint artifacts\models\<tag>_fold<N>\checkpoint-<최신>`)
      (프로세스 죽었는지 확인: `tasklist | findstr python` / GPU: `nvidia-smi`. 학습은 분리 프로세스라 이 대화창 닫혀도 계속 돎.)
  - **[Qwen3-Coder 조사, 7/7] 소형 코더 없음 → 노선 무변**: Qwen3-Coder는 **480B-A35B, 30B-A3B MoE뿐**(둘 다 T4 불가). Qwen3 dense는 전부 general(0.6/1.7/4/8/14/32B), 1.7B도 1.5B처럼 속도벽 → **T4에선 Qwen3-0.6B-Base가 천장**.
  - **[30B-A3B 판정] 봉쇄**: 총 30.5B(활성3.3B). MoE라 라우팅 전 30B 전체 상주 필요 → **int4도 16.8GB = 제출 1GB의 17배 초과, T4 16GB·로컬 12GB 둘 다 초과**. 직접 사용 불가. 증류 teacher만 이론상 가능(클라우드/오프로딩 필요, 대형 프로젝트·이득 불확실).
  - **[모델 사다리 확정, 7/7] T4 제출 가능 = ~0.8B dense가 천장**: 9B dense int4 5GB(1GB 5배 초과)+속도 15배 → 봉쇄. 2B/4B/1.5~1.7B = 속도벽(#30). **MoE 코더(30B/480B) = 메모리벽.** ⇒ 직접 제출 후보는 **≤~0.8B dense뿐**.
  - **[Qwen3.5-0.8B 조사 → 기각, 7/7]** config.json 확인: **멀티모달 VLM**(`Qwen3_5ForConditionalGeneration`, model_type `qwen3_5`, image/video/vision 토큰 보유·vocab 248k) + **transformers 4.57.0.dev0 요구**(검증본 4.51.3 훨씬 초과, dev 미출시). ⇒ 텍스트분류 부적합 + 제출 번들 재검증 고위험 → **기각.** **엔진 = Qwen3-0.6B-Base 확정.** (Qwen3.5 라인 전체가 VLM+4.57dev일 가능성 높아 재조사 무가치. 신세대 추격은 여기서 종료.) ⇒ "코더 이점 검증"은 곧 **Qwen3-0.6B-Base(general·신세대) vs Qwen2.5-Coder-0.5B(coder·구세대, 0.7571)** 비교 = 지금 fold0가 답함. epoch1 이미 Qwen3-Base 우세(+0.0157) → **신세대 이득이 코더 특화를 상쇄한다는 가설**. fold0 확정 시 판정. (완전 격리 원하면 Qwen2.5-0.5B-Base 추가학습으로 coder-vs-base 직접측정 가능하나, fold0가 실용답을 주면 불필요)
- **✅ 나머지 전부 온전**: `submit.zip`(876MB, LB 0.7591) testzip OK · Qwen 5-fold(qwen05_smoke_fold0~4)+OOF · XLM-R 5-fold+OOF · 블렌드+au(OOF 0.7690) · qwen05_rdrop_fold0(#29, 0.7592)+OOF 모두 생존. **손실 없이 재개 가능.**

- **[7/8 새벽] Qwen 5-fold 완성 + 블렌드 zip 준비 완료.** 폴드 [0.7571/0.7681/0.7553/0.7628/0.7646] = **OOF 0.7617**. **Qwen×XLM-R 블렌드 정직 cross-fit 0.7667** (w=0.627/0.373) + 블렌드 위 au 턴버킷 **+0.0024** → **최종 OOF 0.7690**.
- **제출 대기: `submit.zip`** (876MB, 명명규칙: submit.zip=실전 최종 / 번호붙은 zip=테스트용) = Qwen fold1 임베딩프루닝(988→737MB, 비트동일) + XLM-R fold3 프루닝 + 가중블렌드 + au. **5.13/4.46.3 양쪽 스텁 검증 예측 동일 확인.** 예상 LB **0.766~0.770**, 서버 ~7.5분 예상. 제출 후 결과로 갭 재확인.
- **그 다음 카드**: au-업웨이트 Qwen 재학습, Qwen 폴드 교체 실험(f1↔f4), 3-way 블렌드(속도 예산 확인 필요), Qwen3-0.6B(transformers 핀 교체 리스크)
- **공부키트**: `study/`+`study.zip`(팀원 Colab 실습, lr 2e-5·ep4 검증됨), `제출_공부키트/`(최종코드+공부노트.pdf+제출물) 완성
- **현재 최고 레시피**: `xlm-roberta-base` · max_len 512 · **R-Drop α=1.0** · **LR 4e-5** · **4 epoch** · bs8 ga4 bf16 · sqrt 클래스가중. → **5-fold 완성** [0.7278/0.7390/0.7285/0.7400/0.7388] = **OOF CV 0.7348** (sim-only 0.7412 / au 0.5899). LB 예상 ~0.718. 제출본 `submit.zip`은 아직 3ep 모델(LB 0.7063) — 갱신 필요.
- **갭 공식(검증됨)**: `LB ≈ CV − 0.018`. 후처리 bias/LA는 이득 0으로 **기각**(USE_BIAS=False).
- **[07-06 밤 자동화 진행 중]** overnight2(PID 54024: base4ep ✅0.7278 → large 시도) → 종료되면 overnight3(PID 59996)가 fold1~4를 자동 체인 학습 (같은 태그 `xlmr_v2_rdrop_lr4_e4` → 아침에 5-fold 완성 예정). 로그: artifacts/overnight{2,3}.log
- **다음 할 일(아침)**: `python work\status.py` → **MORNING.md** 따라: ① large/5-fold 결과 확인 ② **Qwen2.5-Coder-0.5B fold-0 스모크** (준비 100%: 가중치 `pretrained/`, train.py 디코더 지원, 클래스배선·토큰길이 검증완료. 게이트 CV≥0.74) ③ `bench_infer.py`로 T4 10분 판정 ④ `blend_oof.py`→`package_multi.py`→제출
- **그 다음 카드**: pruned 앙상블(`work/prune_vocab.py`; **Qwen도 vocab 152k라 프루닝 유효** 988→~790MB), self-distill(5-fold OOF 확보 후).
- **주의**: featurize는 **V2**(V1 학습 모델과 비호환). 코드설명=`team_guide.pdf`, 팀공유=`team_share.zip`(4.8MB).
- **규정 체크(07-06, 원문 대조)**: 위반 없음. 사전학습모델/외부데이터 "법적 제한 없는 것" 허용 — XLM-R=MIT ✓, Qwen은 **0.5B/1.5B(Apache-2.0)만** 사용(3B는 연구용 라이선스 금지), KLUE=CC-BY-SA(출처 표기). **할 일: 최종 제출 문서에 "외부 요소 출처·활용 범위" 섹션 기재 의무** (규정 명시).
- **[7/6 밤 재분석]** MORNING.md §8 필독 — sess_au 발견(테스트 조성 확인 필요), train-side overlay 프로브 대기, 턴0-1 약점, 천장 상향(0.80 사정권). 목표 LB 0.80으로 상향.
- **새 세션 시작 문구**: "PROGRESS.md와 MORNING.md 읽고 이어가자."

## 🎯 0.80 로드맵 (재부팅 복구 세션 종합, 파일 전수 재독)
현재 **0.7591** → 컷 0.7807(**+0.022**) → 목표 **0.80(+0.041)**. 갭공식상 5-fold OOF **0.817** 필요.
- **★ 재프레이밍(중요)**: "노이즈 천장 ~0.76"은 **아티팩트로 판명**. 그 순도(0.71~0.73)는 skeleton(프롬프트만·숫자마스킹·히스토리/메타 폐기) 중복군에서만 관측됨. **전체 직렬화 텍스트는 100% 유니크(중복 0), skeleton+last1 in-sample 천장 0.9918.** ⇒ 문맥완비 행엔 0.76 근방 노이즈 바닥이 측정된 적 없음 → **0.78+ 물리적 차단 없음.** 경로 = 엔진 교체 + 세그먼트 레버 합산.

### 실제로 점수 올리는 레버 (미개척 · 기각 안 됨 · OOF 정직검증 전제)
| 우선 | 레버 | 타깃 세그먼트 | 헤드룸(macro) | 상태 |
|---|---|---|---|---|
| 🥇 | **Qwen3-0.6B 엔진** | 전역 | **+0.02~0.04** | **학습 중(PID 7316)**. epoch3>0.7571(0.5B)이 채택 관문. 제출=4.51 핀(검증완료) |
| 🥉 | ~~turn0-1 공략~~ **벽 확정·강등**(7/8) | 저턴 12.9% macro 0.42 | +0.001~0.003 | **Qwen3(강한모델)도 turn0 0.424→0.425=안움직임** → 모델용량 아니라 정보한계(히스토리 없어 액션 under-determined). 유일카드=turn0 오버샘플 fold0 1회 테스트(기대낮음). **0.80의 답 아님.** 산수: 13%를 0.42→0.55 올려도 전체 +0.003뿐 |
| 🥈 | **verify 클러스터 표면단서 강화**(학습측 재직렬화: pytest/eslint/tsc/컴파일 토큰 강조) | lint 0.69(run_tests/bash는 이미 0.82) | +0.005~0.015 | ⚠ 후처리 동사룰 기각 → 학습측만. lint만 약함 |
| ◽ | ~~au-aware 재학습~~ **우선순위↓** | sess_au 7.2% | 소 | **Qwen는 au 이미 macro 0.71**(XLM-R 0.55와 딴판) → 재학습 이득 작음. au 추론 prior만 유지 |
| 🥉 | **Qwen 5-fold soup**(가중평균, 1모델 추론비용) | 전역 | +0.007 | 속도예산 확인 후 |
| ◽ | per-class threshold(F/2, 보정 후) on OOF | 희소클래스(web/list/read) | +0.005~0.015(불확실) | 저렴하나 posthoc 스캔 다수 기각 이력 → 회의적 |

### 기각됨 (반복 금지)
GBM 구조화피처 스택(#6 raw보다↓, "모델이 이미 텍스트로 읽음") · scalar-τ LA/14-param bias(cross-fit≤0) · **후처리 동사룰**("다시 빌드/린트/테스트"→고정매핑; **7/7 Qwen OOF 실측 −0.0065**, len≤20서 정답파괴438≫복구79 — 모델이 이미 문맥으로 더 잘 맞힘. verify 공략은 **학습측만** 유효) · **오버레이/크로스스텝 히스토리 누수**(히든테스트=세션당 1스텝, train과 완전분리, 제출#5·#7 둘 다 0건) · 세션번호(L1 0.05) · kNN/exact-dup(0%) · markov(0.14)

> **★ 후처리 레버 전면 소진 확정**(bias/LA/threshold/동사룰 전부 cross-fit≤0). ⇒ 0.79~0.80은 **오직 학습측(엔진 확장 + au-aware/turn0/verify 재학습) + 블렌드**로만. GPU 필요 → qwen3 완료 후 순차 재학습이 유일 경로.

### 🔬 Qwen 실측 진단 (7/7, 5-fold OOF 0.7617 — 학습레버 타깃 확정)
- **세그먼트**: sim 0.760 / **au 0.715**(XLM-R 0.55 대비 급상승 — Qwen이 au를 잘 함) / **sim turn0-1 0.419**(n≈8k) vs turn2+ 0.765 / au turn0-1 0.447.
- **클래스**(약→강): list 0.47·read 0.57·grep 0.62·ask 0.66·glob 0.66·lint 0.69·plan 0.70 | web 0.73·run_tests 0.82·run_bash 0.83 | apply 0.96·edit 0.98·write 0.99·respond 1.00.
- **0.80 산수**: 하위 7개(평균 ~0.62)를 **0.75로** 올리면 macro (0.75·7+0.88·7)/14 = **0.815 → LB 0.80**. ⇒ **전장은 nav+intent 클러스터, 특히 turn0-1.** au는 이미 됨(레버 아님).

### 목표 0.79~0.80 경로 (컷 추격 X · 0.79 최소선)
0.7591 → **0.79는 +0.031, 0.80은 +0.041 = 큰 폭.** 단일 레버론 불가 → **엔진 확장 + 세그먼트 레버 전량 스택**이 필수:
- **엔진**: Qwen3-0.6B **5-fold 앙상블**(단일 아님). 0.5B는 5fold가 단일 대비 +0.005 → Qwen3 5fold OOF 목표 ~0.78+. (1GB/속도상 5모델 동시탑재 불가 → **fold-soup(가중평균 1모델)** 또는 상위 2fold)
- **스택**(각 5-fold OOF 정직 cross-fit honest>best+0.002 통과분만): verify클러스터 표면단서(#11, **+0.03 최대 미개척**) · au-aware 재학습 · turn0 오버샘플 · 동사룰
- **블렌드**: Qwen3 + XLM-R(+Qwen2.5) 다양성
**전부 상단 적중해야 0.80. 0.79는 엔진 성공 + 2~3레버로 사정권.** 미달 시 더 큰 엔진 강제(1.5B 속도돌파 재검토: 배치최적화·int4·max_len축소 조합).

## 1. 문제 정의
- **과제**: AI 코딩 에이전트의 **다음 행동**을 14개 클래스 중 하나로 예측 (단일 라벨)
- **평가**: **Macro-F1** (14개 클래스 균등 평균 → 소수 클래스가 점수 좌우)
- **제출**: 코드 제출형. `submit.zip = model/ + script.py + requirements.txt`
  - `script.py`가 `./data/{test.jsonl,sample_submission.csv}` 읽고 `./output/submission.csv` 생성
  - 제약: **≤1GB (zip 압축파일 기준 — 풀린 용량 무관, 7/8 확인), 설치 ≤10분, 추론 ≤10분, 오프라인, UTF-8, 하루 10회**
- **평가환경**: Ubuntu 22.04, **T4 16GB**, 3 vCPU/12GB, Python 3.11.15, torch 2.7.1, transformers 4.46.3
- **로컬 개발**: RTX 4070 Ti 12GB, torch 2.6, transformers 5.13 (⚠ 버전 상위 → 핀 검증 필요)
- **일정**: 예선 마감 2026-07-15 10:00 / 본선(top12) 코드+발표 07-20

## 2. 데이터 현황
- train **70,000행 / 9,429 세션** (id=`sess_sim_날짜_세션-step_턴`), 세션당 평균 7.4스텝
- 언어: language_pref **ko 64% / en 25% / mixed 10%** — 단, 프롬프트 **문자 구성은 영문 58% / 한글 37%** (코드 혼재)
- 클래스 분포 (불균형): edit_file 11171 · grep_search 9912 · read_file 9257 · glob_pattern 5284 · respond_only 5178 · run_bash 5068 · apply_patch 4823 · run_tests 4561 · list_directory 4329 · ask_user 2701 · plan_task 2679 · lint_or_typecheck 2283 · write_file 1481 · web_search 1273
- 각 샘플: `session_meta`(메타) + `history`(유저↔행동 로그) + `current_prompt`(현재 지시)
- **테스트셋 크기 추정 ~3만~4만** (추론 660샘플/초@4070Ti → ~240@T4, 서버 3분에서 역산)

## 3. 실험 로그 (fold-0 기준, session-grouped CV)

| # | 실험 | macro-F1 | 메모 |
|---|---|---|---|
| 1 | mDeBERTa-v3 **bf16** | ❌ NaN | disentangled attention이 mixed precision서 overflow |
| 2 | XLM-R base, max_len **256** | 0.6933 | 안정. e1 0.623→e2 0.685→e3 0.693 |
| 3 | + bias 후처리 | 0.7001 | +0.007 |
| 4 | XLM-R base, max_len **512** | **0.7071** | 문맥 레버 +0.014 (작지만 확실). **공정한 현재 최고 CV** |
| 5 | + bias 후처리 | ~~0.7126~~ | ⚠️ **무효 판정** — split-half cross-fit서 −0.004/−0.001 (과적합). USE_BIAS=False로 전환 |
| 6 | GBM 스태킹 (probs+구조화피처) | 0.6979 | ❌ raw보다 낮음 — 피처 무효 |
| 7 | mDeBERTa-v3 **fp32**, 256 | 0.6799 (e3) | ❌ XLM-R(0.6933)보다 나쁨 |
| 8 | XLM-R + mDeBERTa 앙상블 | 0.7091 | +0.002뿐 → **버림** (mDeBERTa 4.5배 느림) |
| — | **제출 #1 (XLM-R-512+bias)** | **LB 0.6901** | CV 0.7126 → LB 0.6901 = **갭 −0.02** |
| 9 | **[밤샘 7/5→6]** V2직렬화+LR픽스, fold0 plain | 0.7023 | V1 0.7071 대비 −0.005 (V2와 LR픽스 교락, 노이즈 범위) |
| 10 | **fold0 + R-Drop(α=1.0)** | **0.7133** | **+0.011 확정** — R-Drop 채택 |
| 11 | **xlmr_v2_rdrop 5-fold 전체** | **평균 0.7128** [0.7133/0.7151/0.7047/0.7153/0.7154] | 70k OOF 확보. 재시도/스킵 0회 |
| 12 | 후처리 스캔 (70k OOF) | 이득 없음 | scalar-τ LA: τ*=0.10≈0 (sqrt weight가 이미 보정 흡수, 검증 예측 적중). 14-param bias: cross-fit **−0.0013** → 재확인 기각. **plain argmax로 제출** |
| — | **제출 #2 (V2+R-Drop fold0, 클린)** | **LB 0.6944** | CV 0.7133 → 갭 −0.019 확정. 신기록 |
| — | 제출 #3 (τ=0.4 프로브) | LB 0.6916 | OOF 예측(−0.0021)과 일치(−0.0028) → **테스트 분포시프트 없음 확정**, OOF 나침반 검증됨 |
| 13 | 데이터 채굴 (템플릿/규칙 재랭킹) | 기각 | 중복 15%는 이미 만점 클래스, 단서 재랭킹 정직이득 +0.0007뿐 (상한 +0.018은 정답파괴 미반영) |
| 14 | **fold0 LR 4e-5 × R-Drop** | **0.7231** | **+0.0098 확정 승리** — GA버그 시절의 실효 4e-5가 정답이었음. R-Drop이 고LR 안정화. **현 최고 레시피** |
| — | **제출 #4 (LR4×R-Drop fold0)** | **LB 0.7063** 🏆 | 갭 −0.0168. 갭 공식 3연속 검증 (−0.017/−0.019/−0.017) → **OOF로 LB 예측 가능 확정** |
| 15 | 세션 오버레이 (step k 정답 = step k+1 history) | train 100.00% (60,553건) | **히든 테스트엔 세션당 1스텝만** → 제출 #5가 #4와 소수점 10자리 동일 = 0건 적용. 가설 종결, 손해 0. 오버레이는 zip에 무해하게 잔류 |
| 16 | **[밤샘 7/6→7] fold0 4에폭 (LR4×R-Drop)** | **0.7278** | **+0.0047 신기록** — e1 0.666→e2 0.704→e3 0.720→e4 0.728 (수렴 곡선상 4ep=스윗스팟). **5-fold는 4에폭으로 확정**. tag=xlmr_v2_rdrop_lr4_e4 |
| 17 | **[밤샘 7/6→7] XLM-R-large fold0** (s42·LR8e-6·R-Drop·8bit Adam) | 0.7218 | 붕괴 없이 1차 성공(200분). 단 **base 4ep(0.7278)에 −0.006 패배**, 동에폭 base(0.7231)와 동급인데 비용 3.5배 → **주력 기각, 앙상블 다양성 멤버 후보만** (아침 blend_oof로 판정). tag=xlmr_large_s42 |
| 18 | **[밤샘 7/7] e4 레시피 5-fold 전체** | **0.7348** (concat 70k) | [0.7278/0.7390/0.7285/0.7400/0.7388] 재시도 0회. 3ep 5-fold(0.7128) 대비 **+0.022**. **sim-only 0.7412 / au 0.5899**. fold0 기준 large 블렌드 +0.003(in-sample, 한계적). LB 예상 ~0.718 |
| — | **제출 #6 (4ep fold0 단일, 클린)** | **LB 0.7142** 🏆 | CV 0.7278 → 갭 **−0.0136**. 갭 4차 검증 (범위 −0.014~−0.019, 평균 −0.016) |
| — | 제출 #7 (train-side overlay 프로브) | LB 0.7142 (동점) | **명중 0건 → 히든테스트는 train 세션과 완전 분리 확정.** overlay 양방향(#15 테스트내부 + train-side) 모두 영구 종결. 다음 zip부터 lookup 미동봉 |
| 20 | **au-prior 보정** (au 행에만 τ=0.25 logit bias) | OOF cross-fit: au +0.0138, ALL +0.0023 | 5 fold 전부 τ=0.25 일치, au flip 8% |
| — | **제출 #8 (fold0 + au_bias)** | **LB 0.7197** 🏆 (+0.0055) | **테스트에 au 존재 확정, 이득이 OOF 예측의 2.4배 → 테스트 au 비중 ~15% 추정 (train 7.2%의 2배).** 갭 −0.010으로 축소. **au_bias 전 zip 기본 탑재 + au 전용 학습 카드 우선순위 상향** |
| 21 | **au-prior v2 (턴 조건부: 0-1/2+ 버킷별 bias, τ=1.0)** | OOF ALL **0.7395** (v1 +0.0024) | au 턴0-1 재앙지대 0.326→**0.495** 회복. cross-fit 정직 통과. script.py v2 포맷 반영 — **다음 zip부터 적용** |
| 22 | **Qwen2.5-Coder-0.5B fold0 스모크** (512·3ep·lr2e-5·8bitAdam·ckpt·R-Drop無) | **0.7571** | **게이트(0.74) 완파, 인코더 +0.0293.** 이득처 = verify/comm 클러스터 (web_search +0.15, lint +0.09, run_tests +0.08 — 발견 #11 적중). nav는 ±0 (진짜 벽). **디코더 노선 확정, tag=qwen05_smoke** |
| 23 | XLM-R × Qwen fold0 블렌드 | **0.7634** (w=0.37/0.63) | 상호보완 확인. 단 T4 예산상 2모델 동시 탑재 불가 → 블렌드는 속도 해결 후 |
| 24 | **vocab 프루닝 검증 완료** (`prune_multi.py`) | 556→**208MB**, Δlogit=0.00 | 사용 9,194토큰+단일문자 마진=23,248 유지. 비트 단위 동일 확인 → 3-fold 앙상블 1GB 통과 |
| 25 | **추론 최적화**: 길이정렬(내림차순)+토큰예산 배칭 | XLM-R 1.6배↑, Qwen 절벽(10배 저하) 해소 | 오름차순 정렬 시 VRAM 파편화+WDDM 스필 절벽 발견 → 긴 배치 먼저로 해결. T4 환산: XLM-R 185/s, **Qwen 64.5/s (30k=9.3분 ✓ / 40k=11.8분 ❌)** |
| — | **제출 #9 (XLM-R 3-fold 프루닝 앙상블 + au v2)** | **LB 0.7342** 🏆 (+0.0145) | 실행 4:57 → **테스트 ≈ 30k 확정** (Qwen 단독 T4 예산 통과 판정). OOF 0.7395 대비 갭 **−0.0053** — 폴드 앙상블 보너스가 OOF 미반영 + au 이득 테스트 2배 효과. 갭 붕괴 추세 (−0.019→−0.005) |
| — | 제출 #10 (Qwen fold0, **사고**) | LB 0.6818 (실행 6:01) | **원인: transformers 5.13이 config의 rope_theta→rope_parameters로 개명 저장 → 서버 4.46.3이 기본값 10000으로 오해 → RoPE 붕괴.** 로컬 4.46.3 재현 0.6676 → config에 rope_theta 복원 → **0.7639, 5.13과 99.7% 일치.** 패키저에 자동 복원 영구 반영. 부산물: Qwen T4 실측 6:01 (30k 여유 확인) |
| — | **제출 #11 (Qwen fold0, rope_theta 복원판)** | **LB 0.7576** 🏆🏆 (+0.0234, 실행 5:56) | **CV 0.7571 → 갭 +0.0005 = 소멸.** 단일 모델이 3-fold 앙상블 격파. 컷까지 −0.0231. **Qwen 2모델 동시탑재는 시간상 불가** → 남은 카드: fold soup(1모델 비용) · au-Qwen bias(공짜) · Qwen+XLM-R프루닝 블렌드(~8분 ✓) → 0.765~0.775 사정권 |
| 26 | **[7/7밤~7/8] Qwen 5-fold 전체** | **OOF 0.7617** [0.7571/0.7681/0.7553/0.7628/0.7646] | 재시도 0회, 총 7.4h. XLM-R 5-fold 대비 +0.027 |
| 27 | **Qwen×XLM-R 블렌드** (70k 정직 cross-fit) | **0.7667** (w=0.627/0.373) | 게이트 통과(+0.005). 블렌드 위 au 턴버킷 재적합 **+0.0024 → 최종 OOF 0.7690** (au는 low-turn만 τ=1, high는 τ=0 — Qwen이 au를 원래 잘 다룸) |
| 28 | **Qwen 임베딩 프루닝** (`prune_qwen.py`, BPE 무수술 id-리맵 방식) | 988→**737MB**, Δlogit=0.00 | 토크나이저 원본 유지 + vocab_remap.npy로 추론시 재매핑. 미보유 토큰 0. script.py에 리맵 훅 + model_weights 지원 추가 |
| — | **제출 #12 (Qwen f1 + XLM-R f3 블렌드 + au)** | **LB 0.7591** 🏆 (+0.0015, 실행 7:34) | OOF 0.7690 대비 **갭 −0.010** — 예측 하회. **교훈: fold별 OOF 차이는 모델 품질이 아니라 fold 난이도** (XLM-R·Qwen 동일 패턴으로 증명) → "최고 fold 선택"은 무의미, 블렌드 이득도 OOF +0.005 중 ~+0.002만 실전 이전 |
| 29 | **Qwen + R-Drop fold0** (tag=qwen05_rdrop) | **0.7592** (원판 0.7571, +0.0021) | ❌ 게이트(0.762) 미달. R-Drop이 Qwen엔 XLM-R만큼(+0.011) 안 먹힘 → **R-Drop 5-fold 보류**. 0.5B 노선 천장 ~0.76 확정 → 0.80은 판 키우기(1.5B) 필요 |
| 30 | **[7/8] Qwen2.5-Coder-1.5B 제약 프로브** (`quant_probe.py`) | ❌ **속도벽 확정** | T4 30k 추정: fp16 19분·int8 27분(역양자화 더 느림)·int4 20분. max_len 256+낙관(factor 2.1)도 13분 → **10분 한도 통과 불가**. 용량(int4 0.85GB)은 OK지만 속도가 벽. **1.5B 직접 제출 물리적 불가.** 0.80 정공법 봉쇄 → 증류/Qwen3/0.5B극한 중 택 |
| 31 | **[7/8] Qwen3-0.6B-Base fold0 스모크** (512·3ep·lr2e-5, qwen05와 동일 레시피) | epoch1 **0.7131** (⚠ 재부팅으로 중단, 미완주) | **Qwen2.5-0.5B epoch1(0.6974) 대비 +0.0157** = 유망. Qwen2.5는 ep1→ep3 +0.06 상승 이력 → 완주 시 ~0.77+ 기대(불확실). 단 **config model_type=qwen3 / tf 4.51 → 서버 4.46.3 로드 불가**: 제출하려면 transformers≥4.51 핀 or 증류 teacher용. checkpoint-1750만 생존 |
| 19 | **[밤샘 7/6→7] 전방위 재분석** (보고만, 상세 MORNING.md §8) | — | ①**sess_au 발견**: train 7.2%가 별도 정책 소스 (라벨분포 L1 0.42, au macro 0.55 vs **sim 0.7372**) — id prefix로 테스트 판별 가능한 엣지 ②**train-side overlay**: 스텁 5/5 복원 — 프로브 1회로 확정 가능(제로리스크) ③**턴0-1 = 최대 약점** (sim macro 0.44, 13%) ④천장 상향: "노이즈 천장" 결론은 편향 — 0.80 사정권 ⑤기각: 마르코프 0.14·kNN 0.19·세션번호=노이즈 |

### 512 모델 클래스별 F1 (약점 진단)
잘함: respond_only 1.00 · write_file 0.98 · edit_file 0.96 · apply_patch 0.93
못함(탐색 클러스터): **list_directory 0.48 · read_file 0.52 · lint 0.52 · web_search 0.55 · grep 0.60**

## 4. 핵심 발견 (데이터/실험으로 검증)
1. **탐색 클러스터(read/grep/glob/list, 데이터 41%)가 F1 ~0.5 천장** — 오류의 99%가 클러스터 내부. "무슨 종류인지"는 알지만 "정확히 뭔지"를 못 가림
2. **이 천장은 노이즈/애매성 때문 (비가역)** — 프롬프트-라벨 연결이 느슨한 케이스 다수. 학계도 agent tool-selection을 "다중 정답"으로 봄(집합 F1). 모델 키워도 못 뚫음
3. **긴 문맥(history)은 텍스트로 넣으면 소폭 도움(+0.014)**, 구조화 피처로 넣으면 무효(트랜스포머가 이미 텍스트로 읽음)
4. **CV가 LB보다 ~0.02 높음** → "LB ≈ CV − 0.02"로 환산
5. **XLM-R > mDeBERTa** on this task (벤치마크 우위가 전이 안 됨)
6. 텍스트는 **어휘상 안 특수함** (UNK 0%) — TAPT 기대치 낮춤
7. **추론 여유 7분** → 2-모델 앙상블 가능
8. **[2026-07-05 검증] 단일폴드 14-param bias 튜닝은 무효** — in-sample +0.0055는 전부 과적합, 정직한 cross-fit 이득 ≤0. 공정 CV=0.7071. CV−LB 갭 −0.017은 폴드 분산(±0.008@2σ)+분포 시프트
9. **[검증] train.py grad-accum 버그 수정** — transformers 5.13에서 custom compute_loss가 num_items_in_batch 무시 → loss가 GA로 안 나눠져 실효 LR ~2배였음. `model_accepts_loss_kwargs=False`로 수정 (이전 런들은 이 조건에서 학습된 것)
10. **[검증] featurize V1이 loc/budget/elapsed를 미직렬화** — loc은 turn과 독립(corr 0.02)인 라벨 신호. V2로 추가 (+상한 완화 200/110/80, lm top-2). **V2부터 기존 모델과 비호환 — 재학습 전 재패키징 금지**
11. **[검증] verify 클러스터(lint 0.52)는 nav와 달리 노이즈 아님** — 오류가 run_tests(160)/run_bash(53)로 가는 내부 혼동, pytest/eslint/tsc 표면 단서로 분리 가능 → ~+0.03 macro 공략처. comm 클러스터도 유사(+0.03)
12. **[07-06 전처리 감사] V2 직렬화는 사실상 무손실 확정** — args 키 10/10 전부 포함(버려지는 키 0), history는 데이터 자체가 최대 6턴이라 커버 100%, clip 포화 ≤0.3%(prompt max 346<400), 512토큰 잘림 3.8%뿐(max 674). 유일 손실 = language_mix top-2 제한(전 샘플이 3~5개 언어 보유, 미미). **⇒ 0.78 갭은 전처리 원인 아님. mmBERT 장문맥 카드도 무의미(674토큰 넘는 입력 자체가 없음).** 남는 가설: 디코더 LLM · 학습 레시피 · 앙상블. 감사 스크립트: scratchpad/preproc_audit.py
13. **[천장 수학] 실제 nav 평균 0.558** (0.5 아님). 0.776 도달엔 중간 6클래스(run_bash·run_tests·plan_task·ask_user·web_search·lint) 평균 0.632→0.793 필요. 헤드룸 상한 ~0.87, 현실적 CV 0.74~0.755

## 5. 산출물 위치
- 학습/추론 코드: `work/` (train.py, featurize.py=공유 직렬화, script.py=제출추론, tune_bias.py, run_cv.py, package.py)
- 아티팩트: `artifacts/` (train_prepared.parquet, oof/, models/, bias_*.json, 분석 txt들)
- 제출 파일: `submit.zip` (489MB, fp16 XLM-R-512 + bias)
- 프로젝트 메모리: `~/.claude/.../memory/`

## 6. 방법 메뉴 (deep-research 검증 완료: 25주장 중 19확정/6폐기)
**헤드라인: 은탄환 없음. 대부분 이미지/합성노이즈/영어 벤치라 우리 task엔 각 +0~2pt로 축소 예상. 경로 = 규율있는 앙상블 + logit adjustment/threshold.**

| 우선 | 방법 | 근거 | 비고 |
|---|---|---|---|
| 🥇 | **앙상블 (multi-seed 우선)** | Kaggle 우승: 3-seed가 multi-fold 이김 | 재학습 필요 |
| 🥇 | **Logit adjustment** (τ·log prior) | Menon ICLR'21 | 제로비용, train빈도만, 히든테스트 생존 |
| 🥈 | **KLUE 한국어 인코더 앙상블 멤버** | YNAT KLUE-BERT 85.7 > XLM-R 83.5 | mDeBERTa 대체. 단 우리 58% 영어라 검증 필요 |
| 🥈 | **R-Drop** | +1.3F1 (텍스트 실험) | 플러그인, 추론비용 0 |
| 🥈 | **AWP** (>FGM) | AT-BERT 우승, 최신 우승자 AWP | +0.3~0.6, 학습에 얹기 |
| 🥉 | **calibration→threshold** | Lipton'14 (F/2 규칙, 보정 전제) | temperature scaling 먼저 |
| ◽ | 노이즈 강건 손실(GCE q=0.7/SCE) | 이미지선 8~17pt, 텍스트선 0~2pt | 우리 노이즈는 instance-dependent → 불확실 |
| ❌ | ModernBERT · LoRA소형LLM · e5임베딩 · EDA · pseudo-labeling | 딥리서치로 기각/봉쇄 | 안 함 |

**추천 순서**: ① logit adjustment(즉시,제로비용) → ② KLUE 인코더 테스트 → ③ multi-seed + R-Drop/AWP 재학습 → ④ calibration+threshold. 각 단계 OOF 검증.

## 6b. [2026-07-05] 외부 계획 적대적 검증 결과 (13-agent workflow)
| 항목 | 판정 | 행동 |
|---|---|---|
| SRA (+3~6%) | ❌ 라벨 누수 확정 | 기각. (대안: label-agnostic kNN 보간, 소이득 실험로만) |
| SigmoidF1 | ❌ 멀티라벨 논문 오적용, 단일라벨 근거 음성 | 기각. 굳이면 CE+λ·softF1 블렌드 저LR 1에폭만 |
| Nelder-Mead | ❌ 과적합 악화 | 기각. 좌표상승 유지 + 프로토콜 교체 |
| Logit adjustment | ✅ 단 class-weight와 중복 주의 | 포스트혹 스칼라 τ 그리드(0~1.5)부터. 재학습 시엔 weight 제거+LA-loss τ=1 |
| soft self-distill | ✅ | 5-fold OOF 확보 후 1라운드, α=0.7(nav 0.5) |
| R-Drop→FGM→AWP | ✅ 순서 확정 | R-Drop α=1.0 먼저. Fast-AWP는 기각(저자 폐기) |
| mmBERT | 🔶→⬇ | requirements에 transformers==4.56.2 핀으로 사용 가능. 단 [07-06 감사] **장문맥 보너스 무효**(입력 p99=562, max 674토큰) → 순수 모델 다양성 가치만 남음, 우선순위 하락 |
| KLUE-RoBERTa | 🔶 | 다양성 멤버로 1회 테스트. 게이트: 앙상블>공정기준선 && 불일치율≥10%. 둘이 780MB로 1GB OK |

**수정 완료(코드)**: train.py(GA 버그+기본모델), featurize.py(V2), script.py(OUT_PATH·label순서 assert), package.py(USE_BIAS=False·classes 기록·requirements 정리), tune_bias.py(그리드 축소·tie-break·cross-fit 게이트)

## 7. 확정 로드맵 (검증 반영)
1. **V2 직렬화 + 수정 트레이너로 5-fold XLM-R-512 + R-Drop** 재학습 (~4h) → 70k OOF 확보
2. OOF 70k로: 스칼라-τ LA 그리드 + cross-fit bias 게이트 + temperature scaling
3. **KLUE fold-0** 다양성 테스트 (게이트 통과 시 앙상블 편입)
4. mmBERT fold-0 테스트 (선택)
5. self-distill 1라운드 (+0.002~0.005)
6. verify/comm 클러스터 오류 분석 → 표면 단서 공략 (직렬화/재랭킹)
7. 앙상블 fp16 패키징(≤1GB) + 4.46.3 미러 검증 → 제출
**기대치**: 현실적 CV 0.74~0.755 (LB −0.015~0.02 감안 시 0.72~0.74). 0.776은 6~7단계가 크게 터져야 가능

## 7. 현재 상태 & 다음
- **진행 중**: deep-research 심층 리서치 (더 나은 방법 탐색), mDeBERTa fold-0 마무리
- **다음**: 리서치 결과로 방법 확정 → FGM+앙상블부터 스택 쌓기 → 각 단계 CV 검증
- **냉정한 현실**: 컷 0.776은 top-12 수준. 스택이 다 맞아떨어져야 근접. 노이즈 천장은 인정하고 "넘을 수 있는 부분" 최대화
- **[07-06 밤 리더보드 정찰]** top-20이 0.776~0.794에 밀집 (12위 컷 0.7807). 코드공유엔 공식 TF-IDF 베이스라인 2개뿐 → 0.78은 공개코드가 아니라 상위팀들의 독자 도달. 우리 인코더 천장 추정(LB 0.72~0.74)과 +0.05 괴리 → **nav 클러스터가 "비가역 노이즈"라는 우리 결론이 틀렸을 가능성 높음** (그들은 nav를 0.65~0.75로 가르는 중). 유력 후보: ① 디코더 LLM 파인튜닝(≤0.5B, 코드 사전학습+장문맥) ② 장문맥 인코더(mmBERT 8k — history 전체 투입) ③ 직렬화/피처 누락 신호. 토크게시판은 JS 렌더링이라 미확인.

- [2026-07-09 02:05:48] V3 직렬화 fold0 실험 시작: qwen3_serial_v3, max_len=512, 3ep. 로그: artifacts/qwen3_serial_v3.log

- [2026-07-09 02:05:48] V3 monitor: [2026-07-09 02:05:48] pid=﻿48376 running=0 waiting

- [2026-07-09 02:06:26] V3 monitor: [2026-07-09 02:06:26] pid=48376 running=1 training started

- [2026-07-09 02:21:29] V3 monitor: [2026-07-09 02:21:29] pid=48376 running=1 training started

- [2026-07-09 02:36:32] V3 monitor: [2026-07-09 02:36:32] pid=48376 running=1 training started

- [2026-07-09 02:51:36] V3 monitor: [2026-07-09 02:51:36] pid=48376 running=1 training started

- [2026-07-09 03:06:39] V3 monitor: [2026-07-09 03:06:39] pid=48376 running=1 training started

- [2026-07-09 03:21:42] V3 monitor: [2026-07-09 03:21:42] pid=48376 running=1 training started

- [2026-07-09 03:36:45] V3 monitor: [2026-07-09 03:36:45] pid=48376 running=1 training started

- [2026-07-09 03:51:49] V3 monitor: [2026-07-09 03:51:49] pid=48376 running=1 training started

- [2026-07-09 04:06:52] V3 monitor: [2026-07-09 04:06:52] pid=48376 running=1 training started

- [2026-07-09 04:21:55] V3 monitor: [2026-07-09 04:21:55] pid=48376 running=1 training started

- [2026-07-09 04:36:59] V3 monitor: [2026-07-09 04:36:59] pid=48376 running=1 training started

- [2026-07-09 04:52:02] V3 monitor: [2026-07-09 04:52:02] pid=48376 running=1 training started

- [2026-07-09 05:07:05] V3 monitor: [2026-07-09 05:07:05] pid=48376 running=1 training started

- [2026-07-09 05:22:08] V3 monitor: [2026-07-09 05:22:08] pid=48376 running=1 training started

- [2026-07-09 05:37:11] V3 monitor: [2026-07-09 05:37:11] pid=48376 running=1 training started

- [2026-07-09 05:39:12] V3 monitor: [2026-07-09 05:39:12] pid=48376 running=1 macro=0.7669

- [2026-07-09 05:43:41] V3 직렬화 fold0 완료: CV 0.766900 (기존 Qwen3 fold0 0.767924 대비 -0.001024)

- [2026-07-09 05:43:41] V3 직렬화 fold0가 기존보다 낮거나 같아서 submit3 패키징은 진행하지 않음.
