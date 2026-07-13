# mmBERT-base 5-fold OOF 학습 (dacon train.py 재사용, SDPA, V2 직렬화)
# 목적: Qwen3와의 오류 다양성 측정용 OOF 확보. 단독 성능은 부차적.
# 실행: Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File C:\Users\joon2\Desktop\da2\src\train_mmbert_folds.ps1" -WindowStyle Hidden
Set-Location C:\Users\joon2\Desktop\dacon
$env:PYTHONUNBUFFERED = "1"
foreach ($f in 0,1,2,3,4) {
  python work\train.py --model pretrained\mmBERT-base --fold $f --tag mmbert_v2 `
    --max_len 512 --epochs 3 --bs 16 --grad_accum 2 --lr 3e-5 --warmup 0.06 --wd 0.01 `
    --weighting sqrt --precision bf16 --grad_ckpt *> artifacts\mmbert_v2_fold$f.log
}
