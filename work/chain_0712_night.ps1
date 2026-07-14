# 07-12 밤샘 체인: full-data(PID 35544) 종료 대기 -> Embedding fold0 -> lr4e-5 fold0
# 실행: Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File work\chain_0712_night.ps1" -WindowStyle Hidden
Set-Location C:\Users\joon2\Desktop\dacon
$env:PYTHONUNBUFFERED = "1"

Wait-Process -Id 35544 -ErrorAction SilentlyContinue

# 슬롯 A: Qwen3-Embedding-0.6B checkpoint 교체 fold0 (ROADMAP_080 1순위, 게이트 0.7679+0.003)
python work\train.py --model pretrained\Qwen3-Embedding-0.6B --fold 0 --tag qwen3emb_smoke `
  --max_len 512 --epochs 3 --bs 8 --grad_accum 4 --lr 2e-5 --warmup 0.1 --wd 0.01 `
  --optim adamw_bnb_8bit --grad_ckpt --weighting sqrt --precision bf16 *> artifacts\qwen3emb_smoke.log

# 슬롯 B: Base + lr 4e-5 fold0 (da2 docs/02 레버, 게이트 0.7679+0.002)
python work\train.py --model pretrained\Qwen3-0.6B-Base --fold 0 --tag qwen3_lr4_smoke `
  --max_len 512 --epochs 3 --bs 8 --grad_accum 4 --lr 4e-5 --warmup 0.1 --wd 0.01 `
  --optim adamw_bnb_8bit --grad_ckpt --weighting sqrt --precision bf16 *> artifacts\qwen3_lr4_smoke.log
