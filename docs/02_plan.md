# 전략 & 실행 계획

> 07-12 개정: 이전 작업 폴더 `C:\Users\joon2\Desktop\dacon\`(같은 대회, LB 0.7677에서 중단)의
> 실험 결과를 반영. 엔진은 Qwen3-0.6B-Base로 확정, 인코더 노선은 제외. 근거 원문: `dacon/PROGRESS.md`.

## 목표와 현실

- 목표: macro F1 ≥ **0.793** (12위선). 07-12 리더보드: 1위 0.798, 12위 0.79307, 20위 0.789.
- 출발점: 이전 프로젝트 최고 **LB 0.7677** (Qwen3-0.6B fold0 + au prior, 추론 9:31). 필요 개선폭 **+0.016**.
- 마감: **7/15 10:00**. 제출 일 10회.

## 확정 사실 (이전 프로젝트 실측 — 재검증·재시도 불필요)

**엔진 서열** (fold0 CV): Qwen3-0.6B-Base **0.7679** (5-fold OOF 0.7701) > Qwen2.5-Coder-0.5B 0.7571 > XLM-R base 최적화 0.7278 > mdeberta 0.68 (bf16는 NaN). 1.5B 이상은 T4 추론 속도벽으로 제출 불가 → **0.6B dense가 천장**.

**확정 레시피**: Qwen3-0.6B-Base · V2 직렬화(`dacon/work/featurize.py`) · max_len 512 · **3ep** (4ep는 −0.006, cosine anneal 특성) · bs8 ga4 · lr 2e-5 · warmup 0.1 · wd 0.01 · sqrt 클래스가중 · bf16 · grad_ckpt · adamw_bnb_8bit · seed 42.

**숨은 테스트**: ~30k 샘플 · 세션당 1스텝 · train 세션과 완전 분리(overlay 프로브 2회 확정) · au 세그먼트(`sess_au_*`, train 7.2%)가 테스트에선 ~15% 추정. **au 턴버킷 prior는 LB +0.0055 — 항상 탑재**(`au_bias.json`).

**CV↔LB 갭**: Qwen ≈ 0 (0.7571→0.7576). XLM-R은 −0.017이었음.

**서버**: Ubuntu 22.04 · T4 · torch 2.7.1 · transformers 4.46.3 기본 → Qwen3는 requirements로 **4.51.3 설치**(검증됨, torch 재설치 불필요). Qwen3-0.6B 단일 추론 **9:31/10분 — 여유 30초, 2모델 블렌드 불가**. 추론은 길이 내림차순 정렬 배칭 필수(오름차순은 VRAM 절벽). 5.13으로 저장한 모델은 rope_theta·extra_special_tokens 픽스 필요(`package_multi.py`에 내장) — 4.51.3으로 학습하면 해당 없음.

**기각 목록 (재시도 금지)**: uniform fold-soup(LB −0.0075) · GBDT 스태킹 · 후처리 bias/LA/threshold(sqrt 가중 학습 모델에선 cross-fit ≤0) · 동사 룰(−0.0065) · tail-anchor 직렬화 V3(−0.001) · max_len 384(LB −0.005) · 마르코프/kNN/세션번호 · overlay.

**재사용 자산**: 학습된 5-fold 모델+OOF(`dacon/artifacts/`), train.py·script.py·package_multi.py·prune_qwen.py·bench_infer.py(`dacon/work/`), 로컬 가중치(`dacon/pretrained/Qwen3-0.6B-Base`), train_prepared.parquet(70k, fold 포함).

## 이전 프로젝트 결론 검증 (07-13 새벽, OOF 재계산 — scratchpad/verify_prior.py)

**재현 확인된 것**
- Qwen3 5-fold OOF **0.7701 정확히 재현**. 클래스별 F1 서열도 일치 (list 0.477 최약 ~ respond 1.000).
- Qwen3×XLM-R 블렌드: 고정 w=0.60에서 **0.7735** (그들 honest 0.7737과 일치) → 블렌드 가치 +0.0034는 실재.
- turn0(히스토리 없음) 구간이 어렵다는 것, 후처리 소진, 1.5B 속도벽(eager 기준)은 타당.

**정정된 것**
- PROGRESS의 "au 0.715 / turn0-1 0.42"는 **Qwen2.5 시절 수치**. Qwen3 5-fold 기준으로 재계산하면 **au 0.7975 (sim 0.7637보다 오히려 높음)**, turn1-2 0.599. au는 더 이상 약점이 아님 — au_bias의 Qwen3 OOF 이득도 +0.0018로 축소 (단 LB +0.0055 실측이 있으므로 탑재는 유지).

**논리적 소프트스팟 (그들 결론이 설정에 종속적인 부분)**
1. **"2모델 블렌드 불가"는 eager PyTorch 한정.** ONNX Runtime(fp16, T4)을 시도한 적 없음 — Qwen3 추론이 1.3배만 빨라져도(9:31→~7분) XLM-R(2:40)과의 블렌드가 10분 안에 들어옴. 블렌드 가치 +0.003 재현됐으므로 유일하게 남은 앙상블 경로.
2. **sqrt 클래스가중은 Qwen에서 ablation된 적 없음** (XLM-R 시절 설정 승계).
3. **V3 직렬화 실패(−0.001)는 3가지 변경(tail anchor + last_action + compact hist) 번들 실험** — 개별 레버는 미검증. verify-cue, turn-bucket 태그는 시도 자체가 없었음.

## 남은 레버

| 레버 | 기대 | 근거/비고 |
|---|---|---|
| **full-data 재학습** | +0.003~0.008 | 이전 제출은 전부 80% fold 모델. E000에서 80%CV↔100%LB 오프셋 +0.01 실측. 07-12 밤 실행 중 |
| **ONNX 추론 가속 → Qwen3×XLM-R 블렌드** | +0.002~0.004 | 블렌드 OOF 이득 +0.0034 재현 확인. optimum 내보내기 → 로짓 동일성 검증 → 로컬 속도 실측 → T4 환산 ≤7분이면 채택. 서버에 onnxruntime-gpu 설치 필요(10분 예산 내) |
| **Qwen3 lr 4e-5 fold0 프로브** | 0~+0.01 | XLM-R에선 lr 2e-5→4e-5가 +0.0098. Qwen3는 2e-5만 시도됨. 게이트: fold0 0.7679 대비 +0.002 |
| 블렌드 증류 (teacher=Qwen3×XLM-R OOF) | +0.002~0.003 | 앙상블 이득의 85~90% 회수(문헌). ONNX 실패 시 보험. GPU 밤 슬롯 후보 |
| au prior | (이미 확보) | 탑재 유지. Qwen3 OOF 이득은 +0.0018로 작아졌으나 LB +0.0055 실측 존중 |

기각 (07-13 새벽 정량 분석, scratchpad/cue_analysis.py): **verify-cue 직렬화** — lint 정답 중 표면 단서가 있는 샘플(n=985)은 이미 recall 0.76~0.78로 소화되고 있고, 취약 구간(recall 0.589, n=1,041)은 단서 자체가 없어 태그로 줄 정보가 없음. 단서가 있어도 정답이 run_bash/lint/run_tests로 삼분(23/23/21%)되어 결정력 부족. 기대 이득 +0.001~0.003 < 게이트.

주의: 선형 모델(E000/E001)에서 확인한 threshold 튜닝 +0.011은 **약한 모델에만 유효**. sqrt 가중으로 학습된 Qwen에는 이득 0이 실측돼 있으므로 기대하지 말 것.

## 일정 (실질 D-2.5)

| 시점 | 작업 |
|---|---|
| 07-12 밤 | ✅ train_full.py 작성·디버그 → **qwen3_full3ep 학습 발사** (~4.5h) |
| 07-13 오전 | full 모델 prune_qwen → package_multi(au) → 스테이징 드라이런 → **LB 프로브 제출** (기대 0.770~0.776) |
| 07-13 낮 | **lr 4e-5 fold0 프로브**(~3.2h). 병행(CPU): ONNX 내보내기·로짓 동일성·속도 실측 → 블렌드 타당성 판정 |
| 07-13 밤 | lr 게이트 통과 → lr4e-5 full-data(~4.5h). 실패 시: ONNX 블렌드 성립이면 **XLM-R full-data**(블렌드 파트너 강화, ~3h) / ONNX도 실패면 **블렌드 증류 full-data** (teacher=Qwen3×XLM-R OOF, 단일 모델에 +0.002~3) |
| 07-14 | 최고 조합 패키징·제출. 예비 제출로 검증. 최종 후보 확정 |
| 07-15 오전 | 마감(10:00) 전 최종 제출물 선택 확인 |

## 리스크

| 리스크 | 대응 |
|---|---|
| full-data 이득이 0에 가까움 | verify-cue가 주 레버로 승격. 둘 다 실패 시 0.7677 부근 정체 — 목표 미달 가능성 인정 |
| verify-cue 게이트 실패 | 매몰 3.2h로 끊고 lr 스윕 전환 |
| 4.51.3 학습 모델의 서버 호환 | 로컬 4.51.3 = 서버 설치 버전과 동일이라 리스크 낮음. 제출 전 스테이징 드라이런 필수 |
| 추론 10분 초과 | full 모델도 단일 Qwen3-0.6B라 9:31 동일 예상. bench_infer.py로 사전 확인 |
| 마감이 7/15 **오전 10시** | 7/14 밤까지 모든 제출 완료, 당일 아침은 확인만 |

## 결정 로그

| 날짜 | 결정 | 근거 |
|---|---|---|
| 07-12 | CV = 5-fold GroupKFold(session), seed 42. 단 Qwen 계열은 이전 프로젝트 스플릿(parquet의 fold 컬럼)을 그대로 사용 | 그쪽 OOF/모델과 정합 유지 |
| 07-12 | 학습 정밀도 bf16 (mdeberta fp16/bf16 불안정) | 이전 프로젝트 NaN 실측 + 리서치 |
| 07-12 | 인코더 노선(mdeberta/mbert/klue) 제외, 엔진 = Qwen3-0.6B-Base | 이전 프로젝트 실측: 인코더 최고치 대비 +0.03 이상 |
| 07-12 | 트랜스포머 학습은 dacon/work 파이프라인 사용, da2/src는 EDA·평가용 | 검증된 코드 재사용, 마감 3일 |
| 07-12 | full-data 재학습을 1순위 레버로 | 이전 제출이 전부 80% fold 모델, 갭 실측 +0.01 |
| 07-12 | transformers 4.51.3 로컬 고정 (5.13 금지) | 5.13 학습 실패 이력(da2 docs/03 디버깅 기록) + 서버 설치 버전과 일치 |
