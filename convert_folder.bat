@echo off
rem Convert all videos under a folder to VR Full SBS (iw3 + Video Depth Anything Small),
rem recreating the folder tree in the output folder. Already converted videos are skipped.
rem Usage: convert_folder.bat <input_folder> <output_folder> [--model VDA_L]
rem If it runs out of VRAM or gets very slow:  set IW3_EXTRA_ARGS=--low-vram
rem torch.compile:  set IW3_EXTRA_ARGS=--compile   (measured no faster here, see README.md)
rem   needs "setup.bat --compile" and the Visual Studio C++ tools; vcvars64 is loaded here.
setlocal
if "%~2"=="" (echo Usage: %~nx0 input_folder output_folder [--model VDA_L] & exit /b 1)
if not exist "%~dp0.venv\Scripts\python.exe" (echo Run setup.bat first. & exit /b 1)

echo %IW3_EXTRA_ARGS% | findstr /c:"--compile" >nul && call :load_msvc

"%~dp0.venv\Scripts\python.exe" "%~dp0src\convert_folder.py" %*
exit /b %ERRORLEVEL%

:load_msvc
where cl >nul 2>&1 && exit /b 0
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo warning: vswhere not found, --compile may fail without cl.exe & exit /b 0)
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" (echo warning: Visual Studio C++ tools not found, --compile may fail & exit /b 0)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
exit /b 0
