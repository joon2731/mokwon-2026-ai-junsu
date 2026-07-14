# 07-12 밤샘 체인 v2 (목표 0.80 재편): full-data 종료 대기 -> Embedding fold0 -> 1.7B LoRA 교사 fold0
# lr4e-5 슬롯은 1.7B 교사에 양보 (ROADMAP_080 4순위 승격). 실행:
#   Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File work\chain_0712_night2.ps1" -WindowStyle Hidden
Set-Location C:\Users\joon2\Desktop\dacon
$env:PYTHONUNBUFFERED = "1"

Wait-Process -Id 35544 -ErrorAction SilentlyContinue

# 슬롯 A: Qwen3-Embedding-0.6B checkpoint 교체 fold0 (게이트 0.7679+0.003)
python work\train.py --model pretrained\Qwen3-Embedding-0.6B --fold 0 --tag qwen3emb_smoke `
  --max_len 512 --epochs 3 --bs 8 --grad_accum 4 --lr 2e-5 --warmup 0.1 --wd 0.01 `
  --optim adamw_bnb_8bit --grad_ckpt --weighting sqrt --precision bf16 *> artifacts\qwen3emb_smoke.log

# 슬롯 B: 1.7B LoRA 교사 fold0 (게이트 0.785)
# B-1: 메모리 검증 마이크로런 (~5분)
python work\train_lora.py --tag q17b_micro --epochs 0.01 --bs 8 --grad_accum 4 --grad_ckpt *> artifacts\q17b_micro.log
if ($LASTEXITCODE -eq 0) {
    python work\train_lora.py --tag qwen3_17b_lora --epochs 3 --bs 8 --grad_accum 4 --grad_ckpt *> artifacts\qwen3_17b_fold0.log
} else {
    # bs8 OOM 시 bs4 폴백
    python work\train_lora.py --tag q17b_micro4 --epochs 0.01 --bs 4 --grad_accum 8 --grad_ckpt *> artifacts\q17b_micro4.log
    if ($LASTEXITCODE -eq 0) {
        python work\train_lora.py --tag qwen3_17b_lora --epochs 3 --bs 4 --grad_accum 8 --grad_ckpt *> artifacts\qwen3_17b_fold0.log
    }
}
