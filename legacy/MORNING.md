# ☀️ 아침 브리핑 — 2026-07-07 (D-8)

> **밤새 전부 완료 (재시도·크래시 0회)**: base 4ep ✅0.7278 → large ✅0.7218 → **5-fold 완성 ✅ OOF CV 0.7348** (sim-only **0.7412**) → **Qwen 스모크 자동 발사** (진행 중, `artifacts/qwen_smoke.log` — §2 명령은 이미 실행됐으니 결과만 확인).

## ⚡ 밤새 최종 스코어보드

| 항목 | 결과 |
|---|---|
| 5-fold (e4) | [0.7278 / 0.7390 / 0.7285 / 0.7400 / 0.7388] → **OOF 0.7348** (3ep 5-fold 대비 +0.022) |
| sim / au 분리 | **sim 0.7412** / au 0.5899 (au가 전체 −0.0064 잠식 — §8 발견 1) |
| XLM-R-large fold0 | 0.7218 (붕괴 없이 1차 성공) — 주력 기각, fold0 블렌드 +0.003 한계적 |
| LB 예상 | OOF 0.7348 − 갭 0.017 ≈ **0.718** + 폴드 앙상블 보너스 |

## 0. 일어나서 제일 먼저

```
python work\status.py
```

밤새 로그 요약 + 모든 OOF 점수 + GPU 상태가 한 번에 나옴. 볼 것 3가지:
1. **large 결과** — `summary: ... large=0.xxxx` 줄. 0.55 이하면 붕괴(재시도 이력 확인), 0.73+면 앙상블 멤버 후보
2. **overnight3 진행도** — fold 1~4 중 몇 개 끝났나 (fold당 ~73분)
3. **에러 여부** — Traceback/RETRY 줄

## 1. 확정된 사실 (어젯밤 기준)

| 항목 | 값 |
|---|---|
| base 4에폭 fold-0 | **CV 0.7278** (3ep 0.7231 대비 +0.0047, 신기록) |
| 5-fold 레시피 | 4에폭 확정, overnight3가 자동 진행 중 |
| 전처리/후처리 | 더 짤 것 없음 확정 (PROGRESS 발견 #12, 후처리 전부 기각) |
| 리더보드 | 본선컷(12위) **0.7807**, 1위 0.7936. 인코더 천장 ~0.73대 → **0.78은 노선 전환 필요** |
| Qwen 준비 | ✅ 가중치 다운로드·train.py 디코더 지원·클래스 배선 검증 완료. 토큰 p99=520 (max_len 640이면 잘림 0%) |

## 2. 오늘의 메인 이벤트: Qwen 스모크 (GPU 비는 대로)

```
python work\train.py --model pretrained\Qwen2.5-Coder-0.5B --fold 0 --tag qwen05_smoke --max_len 512 --epochs 3 --bs 8 --grad_accum 4 --precision bf16 --lr 2e-5 --warmup 0.1 --optim adamw_bnb_8bit --grad_ckpt --weighting sqrt
```

- 예상 소요 ~2.5–3.5시간, VRAM ~6GB (여유)
- **판정 게이트**: fold-0 CV **≥ 0.74** → LLM 노선 확정 (5-fold 확장 + vocab pruning 패키징) / **≤ 0.72** → 가설 기각, 인코더 마무리
- overnight3가 아직 돌고 있으면: 기다렸다 하는 게 기본. 급하면 `taskkill /PID <overnight3 PID> /F` + 현재 fold의 train 자식 프로세스도 종료 (진행 중 fold는 나중에 수동 재실행 필요 — status.py로 어느 fold까지 됐는지 확인)
- 참고: LR 대안 1e-5(보수)/3e-5(공격). 첫 에폭 loss가 ln(14)≈2.64에서 안 내려오면 LR 낮춰 재시도

## 3. 스모크 직후 반드시: 추론 속도 실측 (제출 가능성 판정)

```
python work\bench_infer.py --model artifacts\models\qwen05_smoke_fold0_best
```

- T4 환산 후 **30k/40k 샘플 예상 분수** 출력 (한도 10분, 로드 오버헤드 ~1.5분 감안)
- 초과 시 카드: max_len 512→448, 길이 정렬 배칭(script.py 개조), 대안으로 fp16 배치 키우기
- **fp16 안정성 체크**(제출은 T4=fp16 강제): 같은 5k 샘플에서 fp32 vs fp16 argmax 일치율 ≥99.9% 확인

## 4. 5-fold 완성 후 (overnight3 끝나면)

```
python work\blend_oof.py --tags xlmr_v2_rdrop_lr4_e4          # 5-fold CV 확정
python work\blend_oof.py --tags xlmr_v2_rdrop_lr4_e4,qwen05_smoke   # (스모크 후) 블렌드 이득 확인 — fold0 공통분만으로 계산됨
```

- 갭 공식: **LB ≈ 5-fold OOF CV − 0.017** (3연속 검증됨)
- 블렌드 채택 기준: cross-fit honest > best single + 0.002

## 5. 제출 카드 (하루 10회, 오전 권장 1~2발)

| 카드 | 명령 | 기대 LB | 비고 |
|---|---|---|---|
| A. 4ep fold-0 단일 (즉시 가능) | `package_multi.py` (MODELS에 m0=`xlmr_v2_rdrop_lr4_e4_fold0_best`) → 업로드 | **~0.711** | 갭 공식 4번째 검증 겸 안전 신기록 |
| B. 폴드 앙상블 2~3개 | 5-fold 완성 후 MODELS에 상위 폴드 2~3개 | ~0.715–0.72 | bench로 10분 한도 먼저 확인 (모델당 T4 ~3분) |
| C. Qwen 단일/블렌드 | 스모크 통과 시 | ? | vocab pruning(988→~790MB) 필요할 수 있음 |

- package_multi.py는 **라벨 순서를 빌드 타임에 assert** — 이종 모델 섞어도 안전
- script.py는 다중 모델 mean-softmax 이미 지원 (수정 불필요)

## 6. 알아둘 리스크

1. **T4는 bf16 미지원** → 제출 추론은 fp16 (script.py가 이미 half()). Qwen fp16 안정성은 §3에서 확인
2. 서버 transformers 4.46.3의 Qwen2 분류 지원은 4.37+라 OK — 패키징 후 **4.46.3 미러 검증**(기존 절차) 필수
3. Qwen 라이선스: 0.5B/1.5B = Apache 2.0 ✓ (**3B는 금지** — 연구용 라이선스)
4. max_len 640이면 잘림 0%지만 컴퓨트 +25% — 스모크는 512, 본학습 때 640 고려

## 8. 🔬 밤샘 전방위 재분석 보고 (지시: 실행 없이 보고만 / 목표 LB 0.80)

> 원자료: `artifacts/analysis_0707*.txt` · 스크립트: `work/analysis_0707*.py` (전부 CPU, 학습 무간섭)

### 발견 1 — ★★★ train 안에 정책이 다른 두 번째 소스: `sess_au_*` (7.2%)
- 70k 중 **5,025행(1,099세션)** 이 `sess_au_XXXXXX_YYY` 형식 (나머지 64,975행은 전부 `sess_sim_20260522_*`)
- **라벨 분포가 근본적으로 다름** (L1 거리 0.42): read_file 25.7% (sim 12.3%의 2배), glob 1.8%·list 2.2% (sim의 ¼), 히스토리 평균 2.5턴 (sim 3.5), ko 62%
- **현 모델(e4) 성능: au macro 0.55 vs sim 0.7372** — au가 전체 macro를 −0.009 깎는 중. 특히 **au 턴0-1은 acc 0.233 (사실상 랜덤)**
- 시사점:
  1. 우리 진짜 sim 성능은 **0.7372** (지금까지 혼합 지표로 과소 인지)
  2. 테스트에서 **id prefix만으로 au/sim 판별 가능** → au 전용 prior 보정·전용 학습은 상위권도 놓쳤을 수 있는 **차별화 엣지**
  3. 히든테스트 조성이 관건 (스텁은 5/5 sim): sim-only → au는 학습 노이즈 (제외/다운웨이트 실험), 혼재 → au 처리로 macro 수확
  4. 조성 확인: script.py에 au 카운트 print 추가 → 제출 상세 페이지에서 stdout 보이는지 확인

### 발견 2 — ~~train-side overlay~~ ❌ 프로브 결과 기각 확정 (제출 #7 = #6 동점, 명중 0건 — 히든테스트는 train 세션과 완전 분리)
- 로컬 테스트 스텁 **5/5의 세션이 train에 존재**, step K+1 행의 history 마지막 액션 = step K 정답 → **5/5 복원 성공** (단, 스텁 자체가 train 복사본이라 히든테스트 일반화는 미확정)
- 메커니즘은 발견 #15에서 이미 train 100% 검증됨 (60,553건). #15는 "테스트 내부" 겹침(0건)만 확인했고 **"train↔테스트" 겹침은 미검증 경로**
- **1회 제출로 확정, 제로 리스크**: train→{세션:{step:행동}} 조회 테이블을 zip에 동봉, 명중 시 override / 명중 0이면 예측 불변 (카드 A와 점수 비교로 판독)
- 작업량 ~30분. 명중 시 이득 = 겹침 비율에 비례 (0%면 가설 영구 종결)

### 발견 3 — ★★ 턴 0-1이 최대 약점 세그먼트 (전체의 13%)
- sim 턴0-1: **macro 0.439 / acc 0.606** vs sim 턴2+: 0.735/0.764 — 히스토리 없는 순수 의도분류 구간
- 카드: 저턴 샘플 가중↑, 턴 조건 분기, Qwen의 의도이해력 최대 수혜 예상 구간

### 발견 4 — ★★ 천장 재추정: "노이즈 천장" 결론은 하향 편향이었음
- 프롬프트 중복그룹(n=2–9, sim) 라벨 순도 0.71–0.73 → 프롬프트만으론 확률적이나, 모델은 이미 sim acc 0.745 → **히스토리+메타가 실제로 추가 신호를 주는 중이고 상위권 0.79는 그걸 더 뽑은 것**
- 기존 발견 #2("모델 키워도 못 뚫음")·현실적 CV 0.74~0.755 추정 → **상향 수정 필요**. 0.80은 (디코더 + au 처리 + 턴0 공략 + 앙상블 + overlay 명중 시) 이론적 사정권
- 산수: LB 0.80 ≈ sim-CV 0.817. 합산 경로: Qwen +0.02~0.04(가설) · 5fold +0.007 · au +0.005~0.01 · verify/comm +0.01~0.03 · 앙상블 +0.005 → **0.77~0.81** (상단은 전부 적중 시)

### 재확인 후 기각
- 액션 마르코프만: macro 0.14 (신호는 텍스트에 있음) · TF-IDF kNN/기억화: 0.19 + 교차 exact-dup 0건 · 세션번호 신호: L1 0.053 = 샘플링 노이즈 기대치와 동일 · 결과상태 조건부(grep 0match→glob 24% 등): 실재하나 이미 텍스트로 모델에 노출 · 웹 공개 힌트: 전무

### 아침 권장 실행 순서 (전부 미실행 — 확인 후 승인)
1. **overlay 프로브** zip 제작·제출 (30분, 결정적) — 제출 ①
2. **Qwen 스모크** (~3h, 게이트: sim-macro ≥0.75)
3. **au 카드 2종**: au 제외 fold0 재학습(~75분) / au-prior 추론 보정(OOF로 무료 검증)
4. 5-fold 완성 → blend → **카드 A(4ep fold0) 제출 ②** (갭 공식 4차 검증)

## 7. 어젯밤 새로 생긴 파일

- `work/overnight3.py` — 체인 드라이버 (실행 중)
- `work/status.py` — 아침 대시보드
- `work/blend_oof.py` — OOF 조립·블렌드·honest 검증
- `work/package_multi.py` — 다중 모델 패키징 (라벨 순서 assert)
- `work/bench_infer.py` — T4 추론 시간 예측
- `pretrained/Qwen2.5-Coder-0.5B/` — 가중치 (Apache 2.0)
- train.py: 디코더 지원 2줄 (XLM-R 경로 무영향), script.py: 규정 기재 헤더
