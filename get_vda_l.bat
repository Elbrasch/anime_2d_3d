@echo off
rem Download the Video Depth Anything Large weights (~1.5 GB) for convert_vda_l.bat.
setlocal
set "DIR=%~dp0vendor\nunif\iw3\pretrained_models\hub\checkpoints"
set "CKPT=%DIR%\video_depth_anything_vitl.pth"
set "URL=https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth"

if exist "%CKPT%" (echo Already present: %CKPT% & exit /b 0)
echo Video-Depth-Anything-Large is licensed CC-BY-NC-4.0 (non-commercial use only).
echo https://huggingface.co/depth-anything/Video-Depth-Anything-Large
set /p "OK=Type YES to accept the license and download: "
if /i not "%OK%"=="YES" (echo Cancelled. & exit /b 1)

if not exist "%DIR%" mkdir "%DIR%"
curl -L --fail -o "%CKPT%.part" "%URL%" || (del "%CKPT%.part" 2>nul & exit /b 1)
move /y "%CKPT%.part" "%CKPT%" >nul
echo Saved %CKPT%
