"""convert_folder.py: finding videos, skipping finished ones, remux with all tracks, timing check (no GPU)."""
import sys

import pytest

import convert_folder as cf
from conftest import H, TOTAL_FRAMES, W, FPS, ffmpeg, ffprobe_json


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def rel_strings(videos):
    return [r.as_posix() for _, r in videos]


def test_find_videos_mirrors_tree_and_skips_non_videos(tmp_path):
    inp = tmp_path / "in"
    touch(inp / "a.mkv")
    touch(inp / "Show" / "Season 1" / "ep 01.MP4")
    touch(inp / "notes.txt")
    touch(inp / "old_LRF_Full_SBS.mkv")          # an earlier result lying in the input
    out = inp / "out"                              # output folder inside the input folder
    touch(out / "a_LRF_Full_SBS.mkv")
    touch(out / "stray.mkv")

    videos = cf.find_videos(inp, out.resolve())

    assert rel_strings(videos) == ["a.mkv", "Show/Season 1/ep 01.MP4"]
    assert videos[0][0] == inp / "a.mkv"


def test_find_videos_single_file(tmp_path):
    f = tmp_path / "movie.mkv"
    touch(f)
    assert rel_strings(cf.find_videos(f, (tmp_path / "out").resolve())) == ["movie.mkv"]


def run_main(monkeypatch, inp, out, convert):
    monkeypatch.setattr(cf, "convert", convert)
    monkeypatch.setattr(sys, "argv", ["convert_folder.py", str(inp), str(out)])
    return cf.main()


def test_main_converts_missing_and_skips_existing(tmp_path, monkeypatch):
    inp, out = tmp_path / "in", tmp_path / "out"
    touch(inp / "a.mkv")
    touch(inp / "sub" / "b.mkv")
    touch(out / "sub" / "b_LRF_Full_SBS.mkv")      # already converted
    calls = []

    rc = run_main(monkeypatch, inp, out, lambda src, out_path, *rest: calls.append((src, out_path)))

    assert rc == 0
    assert calls == [(inp / "a.mkv", out.resolve() / "a_LRF_Full_SBS.mkv")]


def test_main_continues_after_a_failure(tmp_path, monkeypatch):
    inp, out = tmp_path / "in", tmp_path / "out"
    touch(inp / "a.mkv")
    touch(inp / "b.mkv")
    calls = []

    def convert(src, out_path, *rest):
        calls.append(src.name)
        if src.name == "a.mkv":
            raise RuntimeError("boom")

    assert run_main(monkeypatch, inp, out, convert) == 1
    assert calls == ["a.mkv", "b.mkv"]


def fake_sbs(frames=None):
    """make_sbs stand-in: a double-width video with the source's timing, encoded on the CPU."""
    def make_sbs(src, dst, model, engine):
        limit = ["-frames:v", frames] if frames else []
        ffmpeg("-i", src, "-map", "0:v", "-vf", f"scale={2 * W}:{H}", *limit,
               "-c:v", "libx264", "-preset", "ultrafast", dst)
    return make_sbs


def test_convert_keeps_audio_subtitles_and_chapters(clips, tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "make_sbs", fake_sbs())
    out = tmp_path / "out" / "show" / "ep_LRF_Full_SBS.mkv"
    tmp = tmp_path / "out" / cf.TMP_DIR_NAME

    cf.convert(clips["hevc10"], out, tmp, "VDA_S", "auto")

    assert out.exists()
    assert not out.with_name(out.name + ".partial").exists()
    assert not any(tmp.iterdir())                  # the intermediate SBS video is removed
    info = ffprobe_json(out, "-show_streams", "-show_chapters")
    streams = {s["codec_type"]: s for s in info["streams"]}
    assert set(streams) == {"video", "audio", "subtitle"}
    assert (streams["video"]["width"], streams["video"]["height"]) == (2 * W, H)
    assert streams["video"]["tags"]["title"] == "3D Full SBS"
    assert streams["audio"]["codec_name"] == "aac"
    assert streams["subtitle"]["codec_name"] == "subrip"
    assert len(info["chapters"]) == 2


def test_convert_rejects_a_video_that_ends_early(clips, tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "make_sbs", fake_sbs(frames=TOTAL_FRAMES - 24))
    out = tmp_path / "ep_LRF_Full_SBS.mkv"

    with pytest.raises(RuntimeError, match="timing mismatch"):
        cf.convert(clips["h264"], out, tmp_path / cf.TMP_DIR_NAME, "VDA_S", "auto")

    assert not out.exists()
    assert out.with_name(out.name + ".partial").exists()   # kept for inspection


def test_video_info_counts_frames_from_packets(clips):
    frames, start, end = cf.video_info(clips["h264"])
    assert frames == TOTAL_FRAMES
    assert start == pytest.approx(0, abs=1e-3)
    assert end == pytest.approx(float(TOTAL_FRAMES / FPS), abs=2e-3)
