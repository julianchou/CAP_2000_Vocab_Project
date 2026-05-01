$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
& "C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\python.exe" "-u" "C:\\Project\\CAP_2000_Vocab_Project\\scripts\\run_stage_schedule_background.py" "--root" "C:\\Project\\CAP_2000_Vocab_Project" "--workspace-root" "C:\\Project\\CAP_2000_Vocab_Project\\workspaces\\vocab" "--job-id" "sched_stage5_ep13_14_20260426_145110" "--profile-id" "vocab" "--stages-path" "C:\\Project\\CAP_2000_Vocab_Project\\config\\vocab\\stages.yaml"
exit $LASTEXITCODE
