$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
Set-Location "C:\Users\joon2\Desktop\dacon"

python work\train.py `
  --model pretrained\Qwen3-0.6B-Base `
  --fold 0 `
  --max_len 384 `
  --epochs 3 `
  --bs 8 `
  --grad_accum 4 `
  --lr 2e-5 `
  --warmup 0.1 `
  --wd 0.01 `
  --weighting sqrt `
  --precision bf16 `
  --grad_ckpt `
  --optim adamw_bnb_8bit `
  --tag qwen3_len384 `
  1> artifacts\qwen3_len384.log `
  2> artifacts\qwen3_len384.err

"EXIT_CODE=$LASTEXITCODE" | Set-Content artifacts\qwen3_len384.exit
