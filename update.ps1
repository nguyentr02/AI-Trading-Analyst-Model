# Keeps the 24/7 PC on the latest code from GitHub. Registered by setup_autostart.ps1 -AutoUpdate (every 15 minutes).
#
# Pulls new commits, checks that the new code compiles and imports, then restarts the live service and the
# dashboard so they run it (their own loops start them again within 30 seconds). If the check fails, or can't run,
# it goes back to the version that was running and leaves everything as it was. Every update is written to
# logs\update.log.
#
#   powershell -ExecutionPolicy Bypass -File update.ps1              # run once by hand
#   powershell -ExecutionPolicy Bypass -File update.ps1 -NoRestart   # update the code, restart nothing
param([switch]$NoRestart)
Set-Location $PSScriptRoot
if (-not (Test-Path logs)) { New-Item -ItemType Directory logs | Out-Null }
$log = Join-Path $PSScriptRoot "logs\update.log"
function Log($m) { "$((Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss')) UTC  $m" | Add-Content $log }
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Log "no Python environment at $py; nothing updated (see README, Part 1)"; return }

git fetch --quiet origin
if ($LASTEXITCODE -ne 0) { Log "could not reach GitHub; will try again"; return }
$old = (git rev-parse HEAD).Trim()
$new = (git rev-parse "@{u}").Trim()
if ($old -eq $new) { return }

$changed = @(git diff --name-only $old $new)
git merge --ff-only --quiet $new
if ($LASTEXITCODE -ne 0) { Log "update to $($new.Substring(0,7)) skipped: files changed on this PC block it (git status)"; return }

if ($changed -contains "requirements.txt") {
    & $py -m pip install --quiet -r requirements.txt
}
# The new code must compile and its main modules import. Anything else (an error, no exit code) counts as failed.
$ok = $false
try {
    & $py -c "import compileall, sys; ok = compileall.compile_dir('cryptoai', quiet=1) and compileall.compile_file('app.py', quiet=1); import cryptoai.live, cryptoai.stocks, cryptoai.paper; sys.exit(0 if ok else 1)"
    $ok = ($LASTEXITCODE -eq 0)
} catch {
    $ok = $false
}
if (-not $ok) {
    git reset --hard --quiet $old
    Log "update to $($new.Substring(0,7)) failed its check (does not compile or import); stayed on $($old.Substring(0,7))"
    return
}
Log "updated $($old.Substring(0,7)) -> $($new.Substring(0,7)): $($changed -join ', ')"

$code = $changed | Where-Object { $_ -match '^(cryptoai/|app\.py$|requirements\.txt$|.*\.bat$)' }
if ($code -and -not $NoRestart) {
    $running = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match '-m cryptoai live' -or $_.CommandLine -match 'streamlit(\.exe)?"? run app\.py'
    }
    foreach ($p in $running) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Log "restarted the live service and the dashboard to run the new code"
}
