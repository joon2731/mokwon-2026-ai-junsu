# 재증류(3-way) 완료 감지 -> CPU 프루닝 -> 패키징 -> CPU 드라이런 자동 체인
# 실행: Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File C:\Users\joon2\Desktop\da2\src\auto_package_3w.ps1" -WindowStyle Hidden
$log = 'C:\Users\joon2\Desktop\da2\artifacts\qwen3_distill3w.log'
$deadline = (Get-Date).AddHours(4)
while ((Get-Date) -lt $deadline) {
  if (Select-String -Path $log -Pattern 'saved ->' -Quiet) { break }
  if (Select-String -Path $log -Pattern 'Traceback' -Quiet) { exit 1 }
  Start-Sleep 120
}

$env:PYTHONUNBUFFERED = '1'
$env:CUDA_VISIBLE_DEVICES = ''   # BSM 학습과 GPU 경합 금지 — 전부 CPU

Set-Location C:\Users\joon2\Desktop\dacon
python work\prune_qwen.py qwen3_distill3w_full_best *> C:\Users\joon2\Desktop\da2\artifacts\prune_3w.log
python work\package_multi.py --single_model qwen3_distill3w_full_best_pruned --req_tf451 --au --out submit_distill3w.zip *>> C:\Users\joon2\Desktop\da2\artifacts\prune_3w.log
Move-Item submit_distill3w.zip C:\Users\joon2\Desktop\da2\submit_distill3w.zip -Force

# CPU 드라이런
Set-Location C:\Users\joon2\Desktop\da2
Remove-Item stage3w -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory stage3w\data -Force | Out-Null
Expand-Archive submit_distill3w.zip -DestinationPath stage3w -Force
Copy-Item data\test.jsonl, data\sample_submission.csv stage3w\data\
Set-Location stage3w
python script.py *>> C:\Users\joon2\Desktop\da2\artifacts\prune_3w.log
Set-Location C:\Users\joon2\Desktop\da2
Remove-Item stage3w -Recurse -Force
"AUTO_PACKAGE_DONE" | Out-File artifacts\prune_3w_done.txt