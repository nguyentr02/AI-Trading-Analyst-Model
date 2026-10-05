' Runs a .bat file with no console window, so it can't be closed by accident.
' Used by setup_autostart.ps1:  wscript.exe run_hidden.vbs live.bat
Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
shell.Run "cmd.exe /c """ & WScript.Arguments(0) & """", 0, True
