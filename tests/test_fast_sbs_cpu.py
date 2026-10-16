"""fast_sbs.py parts that run without a GPU: colour conversion, probing, the decoder pipe, encoder QPs."""
import os
import subprocess
import sys

import pytest
import torch

import fast_sbs as fs
from conftest import FPS, H, W


def p010_codes(raw, n_luma):
    """10-bit codes (Y plane, UV plane) from P010 bytes."""
    v = (raw.view(torch.int16).to(torch.int32) & 0xFFFF) >> 6
    return v[:n_luma], v[n_luma:]


@pytest.mark.parametrize("matrix", ["bt709", "smpte170m", "bt2020nc"])
@pytest.mark.parametrize("full_range", [False, True])
def test_rgb_to_p010_and_back(matrix, full_range):
    kr, kb = fs.MATRICES[matrix]
    for colour in [(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.25, 0.5, 0.75), (0.9, 0.1, 0.6)]:
        rgb = torch.tensor(colour, dtype=torch.float32).view(3, 1, 1).expand(3, 8, 16).contiguous()
        raw = fs.rgb_to_p010(rgb, kr, kb, full_range)
        assert raw.dtype == torch.uint8 and raw.numel() == 16 * 8 * 3   # 2 bytes x 1.5 samples
        back = fs.yuv_to_rgb(raw, 16, 8, True, kr, kb, full_range)
        assert torch.allclose(back, rgb, atol=4e-3), (colour, back[:, 0, 0])


@pytest.mark.parametrize("full_range, black, white", [(False, 64, 940), (True, 0, 1023)])
def test_p010_codes_and_layout(full_range, black, white):
    kr, kb = fs.MATRICES["bt709"]
    for value, y_code in [(0.0, black), (1.0, white)]:
        raw = fs.rgb_to_p010(torch.full((3, 4, 6), value), kr, kb, full_range)
        y, uv = p010_codes(raw, 4 * 6)
        assert y.unique().tolist() == [y_code]
        assert uv.numel() == 2 * 2 * 3 and uv.unique().tolist() == [512]   # interleaved U/V, 4:2:0


def test_nv12_8bit_input():
    kr, kb = fs.MATRICES["bt709"]
    w, h = 4, 2
    for y_code, expected in [(16, 0.0), (235, 1.0)]:
        raw = torch.tensor([y_code] * (w * h) + [128] * (w * h // 2), dtype=torch.uint8)
        rgb = fs.yuv_to_rgb(raw, w, h, False, kr, kb, False)
        assert torch.allclose(rgb, torch.full((3, h, w), expected), atol=1e-5)


def test_probe_hevc10_tagged(clips):
    info = fs.probe(clips["hevc10"])
    assert (info["codec"], info["width"], info["height"]) == ("hevc", W, H)
    assert info["ten_bit"] and info["fps"] == FPS
    assert (info["matrix"], info["full_range"]) == ("bt709", False)
    assert info["duration"] == pytest.approx(3.0, abs=0.1)


def test_probe_guesses_bt601_for_untagged_sd(clips):
    info = fs.probe(clips["h264"])
    assert not info["ten_bit"]
    assert info["matrix"] == "smpte170m"           # untagged and below 720p, as iw3 guesses


def test_big_pipe_and_read_exact():
    payload = bytes(range(256)) * 4096             # 1 MiB, larger than a default pipe buffer
    read_fd, write_fd = fs.big_pipe(4 << 20)
    try:
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import sys; sys.stdout.buffer.write(bytes(range(256)) * 4096)"],
                                stdout=write_fd)
    finally:
        os.close(write_fd)
    with open(read_fd, "rb", buffering=0) as stream:
        buf = bytearray(len(payload) // 4)
        got = b""
        for _ in range(4):
            assert fs.read_exact(stream, memoryview(buf))
            got += bytes(buf)
        assert not fs.read_exact(stream, memoryview(buf))   # clean EOF
    assert proc.wait() == 0
    assert got == payload


def test_read_exact_reports_truncated_frame():
    read_fd, write_fd = fs.big_pipe(1 << 20)
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 10)"],
                            stdout=write_fd)
    os.close(write_fd)
    with open(read_fd, "rb", buffering=0) as stream, pytest.raises(EOFError):
        fs.read_exact(stream, memoryview(bytearray(16)))
    proc.wait()


@pytest.mark.parametrize("qp, expected", [(20, (20, 26, 16)), (18, (18, 24, 14)), (0, (0, 1, 0))])
def test_nvenc_qps_match_ffmpeg(qp, expected):
    # ffmpeg hevc_nvenc -qp N: P = N, B = N * 1.25 + 1.25, I = N * 0.8 (rounded)
    assert fs.nvenc_qps(qp) == expected
