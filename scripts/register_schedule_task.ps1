param(
    [Parameter(Mandatory = $true)][string]$TaskName,
    [Parameter(Mandatory = $true)][string]$Command,
    [Parameter(Mandatory = $true)][string]$Arguments,
    [Parameter(Mandatory = $true)][string]$StartAt,
    [string]$Description = ""
)

$ErrorActionPreference = "Stop"

function Write-RegisterResult {
    param(
        [string]$Mode,
        [bool]$WakeToRun,
        [string]$Note
    )

    $payload = @{
        ok          = $true
        task_name   = $TaskName
        mode        = $Mode
        wake_to_run = $WakeToRun
        note        = $Note
    }
    Write-Output ($payload | ConvertTo-Json -Compress)
}

$startTime = [datetime]::Parse($StartAt)
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $Command -Argument $Arguments
$trigger = New-ScheduledTaskTrigger -Once -At $startTime
$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 23)

$errors = @()

try {
    $principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType S4U -RunLevel Limited
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description $Description `
        -Force | Out-Null
    Write-RegisterResult -Mode "s4u" -WakeToRun $true -Note "registered with wake-to-run"
    exit 0
} catch {
    $errors += "s4u: $($_.Exception.Message)"
}

try {
    $principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description $Description `
        -Force | Out-Null
    Write-RegisterResult -Mode "interactive" -WakeToRun $true -Note "registered with interactive token"
    exit 0
} catch {
    $errors += "interactive: $($_.Exception.Message)"
}

try {
    $timeText = $startTime.ToString("HH:mm")
    $dateText = $startTime.ToString("MM/dd/yyyy")
    $taskRun = '"' + $Command + '" ' + $Arguments
    $schtasksOutput = schtasks.exe /Create /TN $TaskName /TR $taskRun /SC ONCE /ST $timeText /SD $dateText /RL LIMITED /F 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ($schtasksOutput | Out-String)
    }
    Write-RegisterResult -Mode "schtasks" -WakeToRun $false -Note "registered without wake-to-run fallback"
    exit 0
} catch {
    $errors += "schtasks: $($_.Exception.Message)"
}

throw ("failed_to_register_task`n" + ($errors -join "`n"))
