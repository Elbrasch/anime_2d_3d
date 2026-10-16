@echo off
rem One-time setup (safe to re-run):
rem   - Python 3.12 + all libraries into .venv with uv, as listed in pyproject.toml / uv.lock
rem   - nunif (iw3) into vendor\nunif, at the commit this code was tested with
rem   - every model a conversion uses (src\download_models.py): VDA_S, warp, scene detection
rem   - optionally VDA_L, after asking you to accept its CC-BY-NC-4.0 license (get_vda_l.bat)
rem Usage: setup.bat [--compile]     --compile also installs triton-windows for torch.compile
setlocal
cd /d "%~dp0"

set "NUNIF_URL=https://github.com/nagadomi/nunif.git"
set "NUNIF_COMMIT=9fd1812477a2a7f372c87ac7ac14fdfab0a7c3ed"

where uv >nul 2>&1 || (echo uv not found. Install it with: winget install astral-sh.uv & exit /b 1)
where git >nul 2>&1 || (echo git not found. & exit /b 1)
where ffmpeg >nul 2>&1 || (echo ffmpeg not found on PATH. & exit /b 1)

rem Keep uv's Python and download cache inside the project unless set already
rem (the default AppData location failed on the original machine).
if not defined UV_PYTHON_INSTALL_DIR set "UV_PYTHON_INSTALL_DIR=%~dp0.uv\python"
if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%~dp0.uv\cache"

set "EXTRAS="
if /i "%~1"=="--compile" set "EXTRAS=--extra compile"
uv sync --python 3.12 %EXTRAS% || exit /b 1

if not exist vendor\nunif\.git (
    git clone "%NUNIF_URL%" vendor\nunif || exit /b 1
)
git -C vendor\nunif fetch --quiet origin || echo warning: could not fetch nunif, using the local checkout
git -C vendor\nunif -c advice.detachedHead=false checkout --quiet %NUNIF_COMMIT% || exit /b 1

set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" "%~dp0src\download_models.py" || exit /b 1

echo.
echo Optional: Video Depth Anything Large (better depth, about half the speed, non-commercial license).
call "%~dp0get_vda_l.bat" || echo VDA_L skipped; run get_vda_l.bat later to add it.

"%PY%" -c "import torch, PyNvVideoCodec; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo Setup done.
