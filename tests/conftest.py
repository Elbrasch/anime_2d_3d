"""Shared fixtures: tiny synthetic test videos made with ffmpeg (CPU encoders), and the gpu marker.

The clips are generated once per test session into a temp folder, so the repository carries no
video files. They are small (320x180, 3 s) so even the GPU pipeline tests take seconds.
"""
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FPS = Fraction(24000, 1001)
W, H = 320, 180
SHOT_FRAMES = 36            # frames in the first shot; the hard cut is between frame 35 and 36
TOTAL_FRAMES = 72


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                   check=True)


def ffprobe_json(path, *args):
    out = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "json", str(path)],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return json.loads(out)


def two_shot_filter():
    """Two moving test patterns, SHOT_FRAMES each, joined with a hard cut."""
    rate = f"{FPS.numerator}/{FPS.denominator}"
    return (f"testsrc2=s={W}x{H}:r={rate},trim=end_frame={SHOT_FRAMES},setpts=PTS-STARTPTS[a];"
            f"life=s={W}x{H}:r={rate}:mold=10:ratio=0.5:death_color=#203060:life_color=#e0c040,"
            f"format=yuv420p,trim=end_frame={TOTAL_FRAMES - SHOT_FRAMES},setpts=PTS-STARTPTS[b];"
            f"[a][b]concat=n=2:v=1:a=0[v]")


def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def models_present():
    return (ROOT / "vendor" / "nunif" / "iw3").is_dir()


def pytest_collection_modifyitems(config, items):
    skip = None
    if not has_cuda():
        skip = pytest.mark.skip(reason="no CUDA GPU")
    elif not models_present():
        skip = pytest.mark.skip(reason="vendor/nunif missing: run setup.bat")
    if skip is not None:
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _needs_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not on PATH", allow_module_level=True)


@pytest.fixture(autouse=True)
def _restore_cwd():
    # fast_sbs.load_iw3 changes into vendor/nunif; keep tests independent of that
    cwd = os.getcwd()
    yield
    os.chdir(cwd)


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    """Paths of the generated test clips (all 320x180, 72 frames at 24000/1001 fps, cut after frame 35).

    hevc10  HEVC Main 10, tagged bt709, with AAC audio, an SRT subtitle and a chapter
    h264    H.264 8-bit, untagged colours
    hi10    H.264 High 10 (NVDEC cannot decode it: exercises the CPU decoder fallbacks)
    """
    d = tmp_path_factory.mktemp("clips")
    seconds = TOTAL_FRAMES / FPS
    srt = d / "subs.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:02,000\nHello\n", encoding="utf-8")
    chapters = d / "chapters.txt"
    chapters.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1500\ntitle=One\n"
                        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=1500\nEND=3000\ntitle=Two\n", encoding="utf-8")
    paths = {"hevc10": d / "hevc10.mkv", "h264": d / "h264.mp4", "hi10": d / "hi10.mkv"}
    ffmpeg("-filter_complex", two_shot_filter(),
           "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={float(seconds):.3f}",
           "-i", srt, "-i", chapters,
           "-map", "[v]", "-map", "0:a", "-map", "1:s", "-map_chapters", "2",
           "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error",
           "-pix_fmt", "yuv420p10le", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-c:a", "aac", "-c:s", "srt", paths["hevc10"])
    ffmpeg("-filter_complex", two_shot_filter(), "-map", "[v]",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", paths["h264"])
    ffmpeg("-filter_complex", two_shot_filter(), "-map", "[v]",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le", paths["hi10"])
    return paths
