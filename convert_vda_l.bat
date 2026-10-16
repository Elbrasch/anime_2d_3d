@echo off
rem 2D -> VR Full SBS with iw3 + Video Depth Anything Large (best depth, about half the speed of VDA_S,
rem ~8 GB VRAM). The weights are CC-BY-NC-4.0 (non-commercial) and are not downloaded by setup.bat:
rem run get_vda_l.bat once.
rem Usage: convert_vda_l.bat [input file or folder]   default: clips\   output: output\VDA_L\
rem If it runs out of VRAM or gets very slow (spills into shared memory), try:
rem   set IW3_EXTRA_ARGS=--low-vram
setlocal
set "CKPT=%~dp0vendor\nunif\iw3\pretrained_models\hub\checkpoints\video_depth_anything_vitl.pth"
if not exist "%CKPT%" (
    echo VDA_L weights missing: %CKPT%
    echo Run get_vda_l.bat first.
    exit /b 1
)
set "INPUT=%~1"
if "%INPUT%"=="" set "INPUT=%~dp0clips"
call "%~dp0convert_folder.bat" "%INPUT%" "%~dp0output\VDA_L" --model VDA_L
