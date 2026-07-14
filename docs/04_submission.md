# 제출 체크리스트 & 패키징

> 07-12 개정: 제출 경로를 이전 프로젝트의 검증된 도구(`work/`)로 통일.
> 서버 실측치는 이전 프로젝트 제출 12회에서 확보된 값.

## 패키징 — work/package_multi.py 사용

Qwen 제출은 직접 zip을 만들지 말고 검증된 패키저를 쓴다 (rope_theta 복원, tokenizer
`extra_special_tokens` 픽스, 라벨 순서 assert, vocab_remap, au_bias 동봉이 전부 자동):

```powershell
# 1) 임베딩 프루닝 (988MB → ~740MB, 로짓 비트 동일 검증 포함)
python work\prune_qwen.py <model_dir_name>
# 2) 패키징 (MODELS 변수를 대상 모델로 수정 후)
python work\package_multi.py --req_tf451 --au --out submit_xxx.zip
# 3) T4 추론 시간 추정
python work\bench_infer.py --model artifacts\models\<name>
```

zip 구조: `model/` + `script.py` + `requirements.txt` (+ `au_bias.json`, `vocab_remap.npy`, `featurize.py`) — 전부 zip 루트.

## requirements.txt (검증된 조합 — 임의 변경 금지)

서버에 torch 2.7.1·transformers 4.46.3이 기본 설치돼 있다. **torch는 requirements에 넣지 않는다**
(재설치 낭비·리스크). Qwen3에 필요한 것만:

```
transformers==4.51.3
tokenizers==0.21.0
huggingface_hub==0.30.0
```

(이 조합으로 서버 설치·채점 통과 실적 있음. 로컬 학습도 4.51.3으로 통일했으므로 5.13 시절의
rope_theta/extra_special_tokens 수술은 새 모델엔 불필요 — 단 package_multi가 알아서 no-op 처리.)

## 제출 전 체크리스트

**모델/패키징**
- [ ] zip ≤ 1GB (압축 기준. Qwen3-0.6B pruned 단일 ≈ 840~960MB)
- [ ] prune 후 로짓 동일성 확인 출력 봤는가 (max|Δlogit|=0)
- [ ] au_bias.json 동봉 (+0.0055 LB 실측 — 뺄 이유 없음)
- [ ] script.py 헤더의 외부요소 출처 기재가 실제 모델과 일치 (Qwen3-0.6B-Base, Apache-2.0) — 규정 의무

**추론 (script.py는 dacon 검증본 사용)**
- [ ] fp16 (T4는 bf16 미지원), `HF_HUB_OFFLINE=1`
- [ ] 길이 **내림차순** 정렬 배칭 (오름차순은 VRAM 파편화로 10배 느려짐 — 실측)
- [ ] 예상 추론 시간: Qwen3-0.6B 단일 30k ≈ **9:31** (여유 30초뿐 — max_len·배치 건드리면 재실측)
- [ ] sample_submission id 순서·컬럼 유지, 14개 레이블 문자열 일치

**드라이런 (매 제출 전)**
- [ ] 스테이징 폴더에서 로컬 test.jsonl 스텁으로 완주 (CLAUDE.md 절차)
- [ ] output/submission.csv 행수·헤더 확인

**제출 관리**
- [ ] 오늘 제출 횟수 확인 (일 10회), docs/03 제출 기록 갱신
- [ ] 제출 zip 보관 (재현성)
- [ ] **마감 7/15 10:00 — 7/14 밤까지 모든 제출 완료, 15일 아침은 확인만**

## 서버 실측치 (이전 프로젝트에서 확보 — 재측정 불필요)

| 항목 | 값 | 근거 |
|---|---|---|
| 테스트셋 크기 | **~30k** | XLM-R 3-fold 실행 4:57에서 역산 |
| transformers 4.51.3 설치 | **통과** (10분 내) | submit_qwen3_req.zip 채점 완료 |
| Qwen3-0.6B 처리량 (T4) | ~64.5 samples/s → 30k ≈ 9:31 | 제출 실측 |
| XLM-R base 처리량 (T4) | ~185 samples/s | 제출 실측 |
| 2모델 블렌드 | **시간 초과로 불가** | 9:31 + XLM-R 2:40 > 10분 |
