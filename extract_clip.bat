@echo off
rem Cut a 30 s test clip from each video in to_convert\ into clips\.
rem Extra arguments are passed through, e.g.:  extract_clip.bat --start 1:30
"%~dp0.venv\Scripts\python.exe" "%~dp0src\extract_clip.py" %*
