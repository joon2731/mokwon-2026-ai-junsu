# 07-13 밤 체인: 3-way 교사 재증류 (full-data) -> balanced softmax fold0 게이트
# 실행: Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File C:\Users\joon2\Desktop\da2\src\chain_0713_night.ps1" -WindowStyle Hidden
Set-Location C:\Users\joon2\Desktop\da2
$env:PYTHONUNBUFFERED = "1"

# 슬롯 A: 3-way 교사(0.7754) 증류 full-data (~4.5h)
python src\train_distill.py --tag qwen3_distill3w --teacher_npz artifacts\teacher_logits_3way.npz `
  --grad_ckpt *> artifacts\qwen3_distill3w.log

# 슬롯 B: balanced softmax fold0 게이트 (~3.2h, 기준 0.7679)
python src\train_la.py --fold 0 --tag qwen3_bsm --grad_ckpt *> artifacts\qwen3_bsm_fold0.log
