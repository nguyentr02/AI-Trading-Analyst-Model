# Registers the daily learning job with Windows Task Scheduler.
# Binance's daily candle closes at 00:00 UTC; the job runs 10 minutes later in local time.
# If the PC is off at that time, it runs as soon as possible afterwards.
#
#   powershell -ExecutionPolicy Bypass -File schedule_daily.ps1            # register or update
#   powershell -ExecutionPolicy Bypass -File schedule_daily.ps1 -Remove    # unregister
param([switch]$Remove)

$name = "Crypto AI daily learning"
if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false
    "Removed '$name'."
    return
}

$bat = Join-Path $PSScriptRoot "daily.bat"
$runAt = [DateTime]::UtcNow.Date.AddMinutes(10).ToLocalTime()  # 00:10 UTC in local time

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$bat`"" -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Daily -At $runAt
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings `
    -Description "Fetch new Binance candles, retrain the crypto AI, log signals. See logs\daily.log." -Force | Out-Null
"Registered '$name' to run daily at $($runAt.ToString('HH:mm')) local time."
