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
#   powershell -ExecutionPolicy Bypass -File setup_autostart.ps1 -Remove      # remove all tasks
#   add -Online to also put the dashboard on the internet through a Cloudflare tunnel ("Crypto AI online
#   access", tunnel.bat); it needs a login first: .venv\Scripts\python -m cryptoai set-login
param([switch]$AtStartup, [switch]$Remove, [switch]$Online)

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
if ($Online) {
    if (-not (Test-Path (Join-Path $PSScriptRoot "dashboard_auth.json"))) {
        "No login set. Run: .venv\Scripts\python -m cryptoai set-login   (then run this script again)"
        return
    }
    $tasks["Crypto AI online access"] = @{
        Bat  = "tunnel.bat"
        Info = "Puts the dashboard online via a Cloudflare tunnel and sends you the link. See logs\tunnel.log."
    }
}

foreach ($old in @("Crypto AI daily learning", "Crypto AI online access") + $tasks.Keys) {
    if (Get-ScheduledTask -TaskName $old -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $old -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $old -Confirm:$false
    }
}

# Ending a task only stops the hidden launcher, not the batch loop and Python it started,
# so stop those too. Otherwise a re-run would leave the old copy running beside the new one.
$running = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'run_hidden\.vbs"? (live|dashboard_server|tunnel)\.bat' -or
    $_.CommandLine -match '/c "?(live|dashboard_server|tunnel)\.bat' -or
    $_.CommandLine -match '-m cryptoai (live|tunnel)' -or
    $_.CommandLine -match 'cloudflared(\.exe)?"? tunnel' -or
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
