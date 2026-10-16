# anime_2d_3d

Batch conversion of 2D videos (anime and live action) to **VR Full SBS** 3D videos, using
[iw3](https://github.com/nagadomi/nunif) with the Video Depth Anything models and a GPU-only
decode → depth → warp → encode pipeline.

- Converts a whole folder tree in one go; the output folder mirrors the input folder structure.
- Re-runnable: videos whose output already exists are skipped, so an interrupted run continues
  where it stopped.
- Keeps every audio track, subtitle and chapter of the source (stream copy).
- Output is named `<name>_LRF_Full_SBS.mkv`, which VR players (Skybox, DeoVR, Bigscreen, ...)
  recognise as side-by-side 3D.

## What you need

- Windows 10 (21H1 or newer) or Windows 11
- An NVIDIA graphics card, RTX 20 series or newer. Developed on an RTX 4070 SUPER (12 GB).
- About 10 GB of free disk space for the installation, plus room for the converted videos

## 1. Install the prerequisites (once per PC)

Open **Windows Terminal** or **PowerShell** (Start menu → type "Terminal") and run the commands
below one at a time. `winget` is the package installer built into Windows; confirm with `Y` if it
asks to accept source agreements.

1. **NVIDIA driver**: install or update it with the [NVIDIA App](https://www.nvidia.com/en-us/software/nvidia-app/)
   or from [nvidia.com/drivers](https://www.nvidia.com/drivers). Version 570 or newer is needed.

   **You do not need the CUDA Toolkit.** PyTorch and PyNvVideoCodec bring their own CUDA
   libraries, and the video encoder/decoder come with the driver.

2. **git**, to download this project and iw3:

   ```powershell
   winget install --id Git.Git -e
   ```

3. **uv**, the Python package manager:

   ```powershell
   winget install --id astral-sh.uv -e
   ```

   **You do not need to install Python.** uv downloads Python 3.12 into the project folder by
   itself, and an existing Python installation does not interfere.

4. **ffmpeg** (full build, with NVIDIA support):

   ```powershell
   winget install --id Gyan.FFmpeg -e
   ```

5. **Optional, only for torch.compile** (it did not make conversions faster here, so most people
   can skip this): the Microsoft C++ compiler, about 6 GB:

   ```powershell
   winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```

**Close the terminal and open a new one** so it finds the new programs. Then check them; each
command should print a version or a list, not "not recognized":

```powershell
nvidia-smi
git --version
uv --version
ffmpeg -hide_banner -filters | findstr scale_cuda
```

`nvidia-smi` shows the driver version in its top line (570 or higher), and the last command
should print a line with `scale_cuda`.

## 2. Download the project and run the setup (once)

```powershell
git clone https://github.com/Elbrasch/anime_2d_3d.git E:\anime_2d_3d
cd E:\anime_2d_3d
.\setup.bat
```

(Any folder works instead of `E:\anime_2d_3d`; a path without spaces is safest.)

The setup takes a while and downloads about 4 GB. It:

- installs Python 3.12 and all libraries into `.venv` from `pyproject.toml` / `uv.lock`
  (uv's Python and download cache go to `.uv\`),
- clones nunif (iw3) into `vendor\nunif` at the commit this code was tested with,
- downloads every model a conversion uses (`download_models.py`): Video Depth Anything Small, the
  stereo warp model and the scene detection model,
- offers to download **Video Depth Anything Large** (~1.5 GB, better depth at about half the
  speed). Its weights are licensed **CC-BY-NC-4.0 (non-commercial only)**: type `YES` to accept,
  or anything else to skip. `get_vda_l.bat` adds it later.

Re-running `setup.bat` is safe; it brings everything back in line with `pyproject.toml`.
`setup.bat --compile` also installs triton-windows for torch.compile (needs step 5 above).
Afterwards the folder `.uv\cache` (a few GB) can be deleted to free space.

## 3. Convert videos

Example: convert everything in `D:\Anime` (including all subfolders) and write the 3D versions to
`E:\Anime 3D`:

```powershell
cd E:\anime_2d_3d
.\convert_folder.bat "D:\Anime" "E:\Anime 3D"
```

- `D:\Anime\Show\Season 1\Episode 01.mkv` becomes
  `E:\Anime 3D\Show\Season 1\Episode 01_LRF_Full_SBS.mkv`; the folders are created as needed.
- The originals are not changed.
- Videos that are already converted are skipped. If the run is interrupted (Ctrl+C, reboot),
  start the same command again and it continues where it stopped. The same works after adding new
  videos to the input folder.
- Paths with spaces need the quotes.

With the larger model: `.\convert_folder.bat "D:\Anime" "E:\Anime 3D VDA_L" --model VDA_L`.
A single video file also works as input. All options:
`.venv\Scripts\python.exe convert_folder.py -h`.

Shortcuts for quick tests:

| Script | What it does |
|---|---|
| `extract_clip.bat [--start 1:30] [--duration 10]` | cuts test clips from the videos in `to_convert\` into `clips\` |
| `convert_vda_s.bat [input]` | converts `clips\` (or the given file/folder) with VDA_S into `output\VDA_S\` |
| `convert_vda_l.bat [input]` | the same with VDA_L into `output\VDA_L\` |

### Models

| Model | Speed | VRAM | License |
|---|---|---|---|
| VDA_S (default) | fast | ~3 GB at 1080p, ~8 GB at 4K | Apache-2.0 |
| VDA_L | about half the speed of VDA_S | ~8 GB at 1080p | CC-BY-NC-4.0 (non-commercial) |

### Stereo settings

Set in `IW3_ARGS` in `convert_folder.py` (used by both engines): `row_flow_v3` warp, divergence 2,
convergence 0.5, synthetic view on both eyes, edge dilation 2 (fewer halos around moving objects),
EMA depth normalisation reset at scene cuts, source frame rate kept, HEVC Main 10 at constant QP 20.

### Environment variables

| Variable | Effect |
|---|---|
| `IW3_EXTRA_ARGS` | extra iw3 options, e.g. `--low-vram` if VRAM runs out |
| `IW3_EXTRA_ARGS=--compile` | `torch.compile`; needs `setup.bat --compile` and the Visual Studio C++ build tools (loaded automatically). Measured no faster here, so off by default |
| `FAST_SBS_STREAMS=0` | run the stereo stage serially instead of overlapped with the depth model (for comparisons; overlapped is the default) |

Set them in the terminal before starting the conversion; they apply to that terminal window only:

```powershell
$env:IW3_EXTRA_ARGS = "--low-vram"     # PowerShell
set IW3_EXTRA_ARGS=--low-vram          # cmd.exe
```

## How it works

`convert_folder.py` finds the videos, runs `fast_sbs.py` for each one (falling back to plain iw3
if it fails; `--engine iw3|fast|auto`), then remuxes the 3D video with all audio, subtitle and
chapter tracks of the source and checks that the timing matches before renaming the result into
place (`.partial` files are unfinished).

`fast_sbs.py` runs iw3's own stereo code, but feeds and drains it without the CPU in the way:

```
ffmpeg (NVDEC) --raw YUV--> Python: YUV->RGB, depth (VDA), warp, RGB->YUV on the GPU --VRAM--> NVENC --> .mkv
```

- **Scene detection** (TransNetV2) runs first, on frames downscaled on the GPU.
- **Decoding** runs on the GPU's video decoder in its own ffmpeg process (H.264, HEVC, AV1, VP9,
  MPEG-2, ...); formats it cannot decode fall back to ffmpeg's CPU decoder automatically.
- **Depth and stereo warp** overlap on two CUDA streams: the warp of one window runs while the
  depth model computes the next.
- **Encoding**: NVENC reads the finished frames straight from GPU memory (PyNvVideoCodec), and
  PyAV writes the compressed stream into the Matroska file. Raw 4K SBS frames never cross to the CPU.

Measured on an RTX 4070 SUPER with VDA_S:

| Source | Plain iw3 | fast_sbs.py |
|---|---|---|
| 1080p → 3840×1080 SBS | 15–19 fps | ~63 fps |
| 4K → 7680×2160 SBS | – | ~31–37 fps |

The output matches iw3's own (PSNR ~49 dB, the difference being encoder rounding).

## Tests

```powershell
.venv\Scripts\python.exe -m pytest                 # everything (~2 min with a GPU)
.venv\Scripts\python.exe -m pytest -m "not gpu"    # only the CPU tests (seconds, no GPU needed)
```

The tests generate tiny synthetic videos (320×180, 3 s, a hard cut after frame 35, audio,
subtitles, chapters) with ffmpeg, so no test footage is stored in the repository.

- **CPU tests** (`tests/test_convert_folder.py`, `tests/test_fast_sbs_cpu.py`): folder mirroring
  and skipping, the remux keeping every track, the timing check, colour conversion round trips
  and P010 bit layout, probing, the decoder pipe, the encoder's QP settings.
- **GPU tests** (`tests/test_gpu_pipeline.py`, marker `gpu`, skipped without CUDA or models):
  the scene cut is found at the right frame with GPU and CPU downscaling, the output is HEVC
  Main 10 with bt709 tags, correct timestamps and every frame, serial and overlapped stereo give a
  byte-identical video, a source NVDEC cannot decode still converts, and `convert_folder.py` end
  to end (including the skip on a second run).

Speed is not tested automatically; it depends on the machine.

## Licenses

This project's code is MIT licensed (see `LICENSE`). What it downloads and uses:

| Component | License | Use |
|---|---|---|
| nunif / iw3, its warp and anti-aliasing models | MIT | any |
| TransNetV2 (scene detection) | MIT | any |
| Video Depth Anything Small (code and weights) | Apache-2.0 | any |
| Video Depth Anything Large (weights, optional) | CC-BY-NC-4.0 | non-commercial only |
| PyTorch, torchvision | BSD-3-Clause | any |
| PyNvVideoCodec | MIT | any |
| PyAV | BSD-3-Clause | any |
| other Python libraries (numpy, scipy, timm, ...) | MIT / BSD / Apache-2.0 | any |
| ffmpeg (gyan.dev full build, installed separately) | GPL-3.0 | any; conditions only apply when redistributing it |

All of these allow private use. The only restriction is VDA_L's **non-commercial** clause: don't use
it (or videos made with it) for paid work or anything you sell. The permissive licenses (MIT, BSD,
Apache) only ask for their notices to be kept when you redistribute the software itself.

These licenses cover the tools, not your videos: whether you may convert and keep a 3D copy of a
video depends on the video's own copyright and your country's private-copy rules (for example,
copy protection may not be circumvented).
