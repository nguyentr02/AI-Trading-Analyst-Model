# Starts the Crypto AI automatically and keeps it running.
#
# Registers two Windows scheduled tasks:
#   "Crypto AI live service"  live.bat: catches up on missed data, then learns and signals at every candle close
#   "Crypto AI dashboard"     the Streamlit dashboard on port 8501 (open from other devices at http://<this-PC-IP>:8501)
# Both run hidden (no console window to close by accident) and restart themselves 30 seconds after a crash.
# Replaces the older "Crypto AI daily learning" task.
#
#   powershell -ExecutionPolicy Bypass -File setup_autostart.ps1              # start when you log in (no admin)
#   powershell -ExecutionPolicy Bypass -File setup_autostart.ps1 -AtStartup   # start at boot, even with nobody
#                                                                             # logged in (24/7 PC; run as admin)
#   powershell -ExecutionPolicy Bypass -File setup_autostart.ps1 -Remove      # remove both tasks
param([switch]$AtStartup, [switch]$Remove)

$tasks = @{
    "Crypto AI live service" = @{
        Bat  = "live.bat"
        Info = "Learns from every closed Binance candle and logs signals. See logs\live.log."
    }
    "Crypto AI dashboard" = @{
        Bat  = "dashboard_server.bat"
        Info = "Crypto AI dashboard on http://localhost:8501."
    }
}

foreach ($old in @("Crypto AI daily learning") + $tasks.Keys) {
    if (Get-ScheduledTask -TaskName $old -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $old -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $old -Confirm:$false
    }
}

# Ending a task only stops the hidden launcher, not the batch loop and Python it started,
# so stop those too. Otherwise a re-run would leave the old copy running beside the new one.
$running = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'run_hidden\.vbs"? (live|dashboard_server)\.bat' -or
    $_.CommandLine -match '/c "?(live|dashboard_server)\.bat' -or
    $_.CommandLine -match '-m cryptoai live' -or
    $_.CommandLine -match 'streamlit(\.exe)?"? run app\.py'
}
foreach ($p in $running) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
if ($running) { "Stopped $(@($running).Count) running Crypto AI processes." }

if ($Remove) { "Removed the Crypto AI tasks. Nothing is running now."; return }

if ($AtStartup) {
    $trigger = New-ScheduledTaskTrigger -AtStartup
    # S4U: runs without anyone logged in and without storing a password. Registering it needs admin.
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
} else {
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
}
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

foreach ($name in $tasks.Keys) {
    $t = $tasks[$name]
    $action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$PSScriptRoot\run_hidden.vbs`" $($t.Bat)" `
        -WorkingDirectory $PSScriptRoot
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Principal $principal `
        -Settings $settings -Description $t.Info -Force | Out-Null
    Start-ScheduledTask -TaskName $name
    "Registered and started '$name'."
}
$when = if ($AtStartup) { "at boot" } else { "when you log in" }
"Both start $when, run hidden, and restart themselves if they crash. Logs are in the logs folder."
if ($AtStartup) {
    "For a 24/7 PC, also turn off sleep: Settings > System > Power > Sleep = Never (screen off is fine)."
}
