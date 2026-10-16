"""The real GPU pipeline on tiny clips: scene cuts, output format, serial == overlapped, fallbacks, end to end.

Needs an NVIDIA GPU and the models from setup.bat; skipped otherwise. Each conversion runs in its
own process, as convert_folder.py runs it.
"""
import os
import shutil
import subprocess
import sys

import pytest

from conftest import FPS, H, ROOT, SHOT_FRAMES, TOTAL_FRAMES, W, ffprobe_json

pytestmark = pytest.mark.gpu

CUT = {SHOT_FRAMES - 1}     # detect_scenes returns the last frame of each shot that ends in a cut


def run_fast_sbs(src, out, **env):
    result = subprocess.run([sys.executable, str(ROOT / "fast_sbs.py"), str(src), str(out)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env={**os.environ, **env})
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return result.stdout + result.stderr


def video_md5(path):
    """MD5 of the video bitstream (container-independent)."""
    return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy", "-f", "md5", "-"],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(scope="module")
def scene_model():
    import torch
    import fast_sbs as fs
    from nunif.utils.transnetv2 import TransNetV2
    from nunif.utils.ui import TorchHubDir
    from iw3.hub_dir import HUB_MODEL_DIR
    with TorchHubDir(HUB_MODEL_DIR):
        model = TransNetV2().load().eval().to("cuda")
    yield fs, model, torch.device("cuda")
    del model
    torch.cuda.empty_cache()


@pytest.mark.parametrize("gpu_scale", [True, False], ids=["gpu-downscale", "cpu-downscale"])
@pytest.mark.parametrize("clip", ["hevc10", "h264"])
def test_scene_cut_found(scene_model, clips, clip, gpu_scale):
    import torch
    fs, model, device = scene_model
    with torch.inference_mode():
        cuts = fs._scene_pass(model, clips[clip], fs.probe(clips[clip]), device, gpu_scale=gpu_scale)
    assert cuts == CUT


def test_scene_detection_falls_back_to_cpu_downscale(clips, capsys):
    import torch
    import fast_sbs as fs
    cuts = fs.detect_scenes(clips["hi10"], fs.probe(clips["hi10"]), torch.device("cuda"))
    assert cuts == CUT
    assert "CPU path" in capsys.readouterr().out


def test_output_format(clips, tmp_path):
    out = tmp_path / "sbs.mkv"
    log = run_fast_sbs(clips["hevc10"], out)
    assert f"{TOTAL_FRAMES} frames in" in log

    info = ffprobe_json(out, "-show_streams", "-show_format", "-count_frames")
    (v,) = info["streams"]
    assert (v["codec_name"], v["profile"], v["pix_fmt"]) == ("hevc", "Main 10", "yuv420p10le")
    assert (v["width"], v["height"]) == (2 * W, H)
    assert (v["color_space"], v["color_primaries"], v["color_transfer"], v["color_range"]) == \
        ("bt709", "bt709", "bt709", "tv")
    assert v["r_frame_rate"] == f"{FPS.numerator}/{FPS.denominator}"
    assert int(v["nb_read_frames"]) == TOTAL_FRAMES
    assert float(info["format"]["start_time"]) == 0

    packets = ffprobe_json(out, "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags")["packets"]
    assert len(packets) == TOTAL_FRAMES
    assert packets[0]["flags"].startswith("K")
    pts = sorted(float(p["pts_time"]) for p in packets)
    assert pts == pytest.approx([i / float(FPS) for i in range(TOTAL_FRAMES)], abs=1.5e-3)


def test_serial_and_overlapped_stereo_give_identical_video(clips, tmp_path):
    run_fast_sbs(clips["h264"], tmp_path / "serial.mkv", FAST_SBS_STREAMS="0")
    run_fast_sbs(clips["h264"], tmp_path / "overlapped.mkv", FAST_SBS_STREAMS="1")
    assert video_md5(tmp_path / "serial.mkv") == video_md5(tmp_path / "overlapped.mkv")


def test_source_nvdec_cannot_decode_still_converts(clips, tmp_path):
    out = tmp_path / "hi10.mkv"
    log = run_fast_sbs(clips["hi10"], out)
    assert f"{TOTAL_FRAMES} frames in" in log
    assert "CPU path" in log                       # scene detection fell back as well


def test_convert_folder_end_to_end(clips, tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    (inp / "Show").mkdir(parents=True)
    shutil.copy(clips["hevc10"], inp / "Show" / "ep 01.mkv")
    shutil.copy(clips["h264"], inp / "clip.mp4")
    cmd = [sys.executable, str(ROOT / "convert_folder.py"), str(inp), str(out), "--engine", "fast"]

    first = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    assert "2 converted, 0 failed" in first.stdout

    ep = out / "Show" / "ep 01_LRF_Full_SBS.mkv"
    types = [s["codec_type"] for s in ffprobe_json(ep, "-show_streams")["streams"]]
    assert types == ["video", "audio", "subtitle"]
    assert (out / "clip_LRF_Full_SBS.mkv").exists()
    assert not (out / ".sbs_tmp").exists()

    second = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert second.returncode == 0
    assert "2 already converted, 0 to do" in second.stdout
