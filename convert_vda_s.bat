@echo off
rem 2D -> VR Full SBS with iw3 + Video Depth Anything Small (fast, Apache-2.0, ~3 GB VRAM).
rem Usage: convert_vda_s.bat [input file or folder]   default: clips\   output: output\VDA_S\
rem For any output folder use convert_folder.bat.
setlocal
set "INPUT=%~1"
if "%INPUT%"=="" set "INPUT=%~dp0clips"
call "%~dp0convert_folder.bat" "%INPUT%" "%~dp0output\VDA_S" --model VDA_S
