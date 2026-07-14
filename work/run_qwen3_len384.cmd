@echo off
cd /d C:\Users\joon2\Desktop\dacon
set PYTHONUNBUFFERED=1

python work\train.py ^
  --model pretrained\Qwen3-0.6B-Base ^
  --fold 0 ^
  --max_len 384 ^
  --epochs 3 ^
  --bs 8 ^
  --grad_accum 4 ^
  --lr 2e-5 ^
  --warmup 0.1 ^
  --wd 0.01 ^
  --weighting sqrt ^
  --precision bf16 ^
  --grad_ckpt ^
  --optim adamw_bnb_8bit ^
  --tag qwen3_len384 ^
  1> artifacts\qwen3_len384.log ^
  2> artifacts\qwen3_len384.err

echo EXIT_CODE=%ERRORLEVEL%> artifacts\qwen3_len384.exit
