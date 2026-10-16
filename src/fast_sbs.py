"""Fast 2D -> VR Full SBS for one video: iw3 + Video Depth Anything, with decode and encode outside Python.

    python fast_sbs.py <input video> <output.mkv> [--model VDA_S] [--batch-size 8]

iw3's own video loop runs on one Python thread, one step after another: CPU decode, CPU colour
conversion to RGB, GPU depth + stereo warp, CPU conversion of the 3840-wide result back to YUV,
then the encoder. The GPU sits idle while the CPU works and the other way round. Here the
stages run at the same time:

    ffmpeg (NVDEC decode, CPU fallback) --raw YUV pipe--> Python (GPU only) --VRAM--> NVENC --HEVC--> PyAV (.mkv)

* Decoding runs on the GPU's video decoder (H.264, HEVC, VP9, AV1, MPEG-2 ...). Formats NVDEC
  cannot handle (e.g. H.264 High 10) are decoded by ffmpeg on the CPU automatically, still in
  its own process.
* YUV <-> RGB conversion and chroma resampling run on the GPU (torch), not in swscale.
* NVENC reads the finished frames straight from GPU memory (PyNvVideoCodec); only the compressed
  stream reaches the CPU, where PyAV writes it into the Matroska file.
* The reader and encoder threads spend their time in blocking reads, NVENC calls and CUDA syncs,
  which release the GIL, so they do not compete with the main (GPU) thread.

The stereo part is iw3's own code (depth model, EMA normalisation, scene resets, row_flow warp,
post-processing), driven exactly like iw3's Video-Depth-Anything path. Scene cuts come from the
same TransNetV2 model in a fast pre-pass (frames downscaled on the GPU). The output is video
only; convert_folder.py adds the audio and subtitle tracks of the source.
"""
import argparse
import contextlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

from settings import IW3_ARGS, NUNIF_DIR

sys.path.insert(0, str(NUNIF_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Kr, Kb luma coefficients per matrix; ffprobe color_space names
MATRICES = {
    "bt709": (0.2126, 0.0722),
    "bt470bg": (0.299, 0.114),
    "smpte170m": (0.299, 0.114),
    "bt2020nc": (0.2627, 0.0593),
    "bt2020c": (0.2627, 0.0593),
}
# ITU-T H.273 codes (ffprobe names) for the HEVC VUI colour description
H273_MATRIX = {"bt709": 1, "bt470bg": 5, "smpte170m": 6, "bt2020nc": 9, "bt2020c": 10}
H273_PRIMARIES = {"bt709": 1, "bt470m": 4, "bt470bg": 5, "smpte170m": 6, "smpte240m": 7, "bt2020": 9}
H273_TRANSFER = {"bt709": 1, "bt470m": 4, "bt470bg": 5, "smpte170m": 6, "smpte240m": 7,
                 "bt2020-10": 14, "bt2020-12": 15, "smpte2084": 16, "arib-std-b67": 18}
SCENE_PAD = 25         # TransNetV2 settings, as in nunif.utils.shot_boundary_detection
SCENE_WINDOW = 100
SCENE_THRESHOLD = 0.5
IN_SLOTS = 8            # pinned host buffers between the decoder pipe and the GPU
DECODER_PIPE_BYTES = 64 << 20   # pipe buffer from the decoder: two 4K P010 frames
ENC_QUEUE = 4           # finished frames (in VRAM) waiting for the encoder thread
ENC_BFRAMES = 3         # as ffmpeg's hevc_nvenc defaults used before
ENC_GOP = 250
NV_ENC_PIC_TYPE_IDR = 3  # picture_type of encoded packets (0 P, 1 B, 2 I, 3 IDR)
STEREO_JOBS = 1         # depth windows handed to the stereo worker at a time (VRAM bound)


# ---------------------------------------------------------------- probing

def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,profile,width,height,pix_fmt,avg_frame_rate,r_frame_rate,"
         "color_space,color_range,color_primaries,color_transfer:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout
    info = json.loads(out)
    s = info["streams"][0]
    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = s.get(key, "0/0").partition("/")
        if den not in ("", "0") and num != "0":
            fps = Fraction(int(num), int(den))
            break
    if fps is None:
        raise ValueError("cannot determine the frame rate")
    pix_fmt = s.get("pix_fmt", "")
    ten_bit = any(tag in pix_fmt for tag in ("10", "12", "16"))
    height = s["height"]
    matrix = s.get("color_space")
    if matrix not in MATRICES:
        matrix = "bt709" if height >= 720 else "smpte170m"   # same guess as iw3
    full_range = s.get("color_range") == "pc" or pix_fmt.startswith("yuvj")
    duration = float(info.get("format", {}).get("duration") or 0)
    return {
        "codec": s.get("codec_name"), "profile": s.get("profile"),
        "width": s["width"], "height": height, "pix_fmt": pix_fmt, "ten_bit": ten_bit,
        "fps": fps, "matrix": matrix, "full_range": full_range,
        "primaries": s.get("color_primaries"), "transfer": s.get("color_transfer"),
        "duration": duration,
    }


def fps_str(fps):
    return f"{fps.numerator}/{fps.denominator}"


def decoder_cmd(src, fps, vf_tail, pix_fmt, gpu_frames=False):
    # -hwaccel cuda decodes on NVDEC and downloads the frames; unsupported formats fall back to
    # the CPU decoder. The fps filter makes the frame sequence constant-rate like iw3 does;
    # passthrough stops the CLI from adding its own CFR step on top, which would duplicate the
    # first frame when the video does not start at t=0 and shift every frame by one.
    # gpu_frames keeps the decoded frames in VRAM for CUDA filters in vf_tail (no CPU fallback).
    hw = ["-hwaccel", "cuda"] + (["-hwaccel_output_format", "cuda"] if gpu_frames else [])
    return ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            *hw, "-i", str(src), "-map", "0:v:0",
            "-vf", f"fps={fps_str(fps)}" + vf_tail, "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", pix_fmt, "-avioflags", "direct", "pipe:1"]


def big_pipe(size):
    """(read fd, write fd) of a pipe with a `size`-byte buffer on Windows.

    subprocess pipes use the small default buffer, so a 25 MB 4K frame crossed it in thousands
    of small reads and writes; with room for whole frames (and -avioflags direct, so ffmpeg
    writes a frame in one call) it takes a few.
    """
    if os.name != "nt":
        return os.pipe()
    import _winapi
    import msvcrt
    read_handle, write_handle = _winapi.CreatePipe(None, size)
    return msvcrt.open_osfhandle(read_handle, os.O_RDONLY), msvcrt.open_osfhandle(write_handle, 0)


def read_exact(stream, view):
    """Fill a memoryview from a pipe; returns False on a clean EOF before the first byte."""
    got = 0
    n = len(view)
    while got < n:
        k = stream.readinto(view[got:])
        if not k:
            if got == 0:
                return False
            raise EOFError(f"truncated frame from decoder ({got} of {n} bytes)")
        got += k
    return True


# ---------------------------------------------------------------- scene cuts

# TransNetV2 looks at 48x27 RGB frames. Downscaling on the GPU before the download keeps the full
# frames in VRAM: copying every 4K frame to system RAM and scaling it there halved the speed.
# The GPU goes to 384x216 first and swscale does the same last bilinear step as the CPU path,
# which keeps the pixels within ~1 level of it (a direct GPU scale to 48x27 aliases).
SCENE_SCALE_CPU = ",scale=48:27:flags=bilinear"
SCENE_SCALE_GPU = ",scale_cuda=384:216:interp_algo=bilinear:format=nv12,hwdownload,format=nv12" + SCENE_SCALE_CPU


@torch.inference_mode()
def detect_scenes(src, info, device):
    """Frame indices that end a shot (iw3's segment_pts semantics, with frame index as pts)."""
    from nunif.utils.transnetv2 import TransNetV2
    from nunif.utils.ui import TorchHubDir
    from iw3.hub_dir import HUB_MODEL_DIR

    with TorchHubDir(HUB_MODEL_DIR):
        model = TransNetV2().load().eval().to(device)
    try:
        try:
            return predict_cuts(model, scene_frames(src, info, device, gpu_scale=True))
        except RuntimeError:
            # formats NVDEC cannot decode come out as CPU frames, which scale_cuda cannot take
            print("note: no GPU downscaling for this video (not decodable on NVDEC?); "
                  "scene detection uses the CPU path", flush=True)
            return predict_cuts(model, scene_frames(src, info, device, gpu_scale=False))
    finally:
        del model
        torch.cuda.empty_cache()


def predict_cuts(model, chunks):
    """TransNetV2 over a stream of frame chunks -> indices of the frames that end a shot.

    `chunks` yields (n, 3, 27, 48) float tensors of SCENE_PAD frames (the last may be shorter);
    `model(x)` returns (per-frame logits, _). Windowing as nunif's shot_boundary_detection: the
    first frame is repeated SCENE_PAD times in front, each window of SCENE_WINDOW frames keeps the
    predictions of its middle part and advances by that much, and the last frame is repeated
    until every real frame has been predicted.
    """
    window, probs = deque(), []
    count = 0
    last = None

    def push(x):
        window.append(x)
        if len(window) == SCENE_WINDOW // SCENE_PAD:
            single, _ = model(torch.cat(list(window)))
            probs.append(torch.sigmoid(single).flatten()[SCENE_PAD:-SCENE_PAD].cpu())
            for _ in range((SCENE_WINDOW - SCENE_PAD * 2) // SCENE_PAD):
                window.popleft()

    for x in chunks:
        if count == 0:
            window.append(torch.cat((x[:1],) * SCENE_PAD))
        count += x.shape[0]
        if x.shape[0] < SCENE_PAD:
            x = torch.cat((x,) + (x[-1:],) * (SCENE_PAD - x.shape[0]))
        push(x)
        last = x[-1:]
    if count == 0:
        return set()
    while sum(p.numel() for p in probs) < count:
        push(torch.cat((last,) * SCENE_PAD))
    preds = torch.cat(probs)[:count]
    return set(torch.nonzero(preds > SCENE_THRESHOLD).flatten().tolist())


def scene_frames(src, info, device, gpu_scale):
    """The video as 48x27 RGB frames for TransNetV2: (n, 3, 27, 48) float tensors on `device`,
    SCENE_PAD frames each (the last may be shorter). Raises RuntimeError if the decoder fails."""
    frame_bytes = 48 * 27 * 3
    vf = SCENE_SCALE_GPU if gpu_scale else SCENE_SCALE_CPU
    # ffmpeg's messages go to a file: they explain a failure, and a pipe could fill up and stall it
    errlog = tempfile.TemporaryFile()
    proc = subprocess.Popen(decoder_cmd(src, info["fps"], vf, "rgb24", gpu_frames=gpu_scale),
                            stdout=subprocess.PIPE, stderr=errlog, bufsize=0)
    total = int(info["duration"] * info["fps"]) or None
    pbar = tqdm(total=total, desc="scene detection", unit="f", ncols=90)
    try:
        n = SCENE_PAD
        while n == SCENE_PAD:
            chunk = bytearray(frame_bytes * SCENE_PAD)
            view = memoryview(chunk)
            n = 0
            while n < SCENE_PAD and read_exact(proc.stdout, view[n * frame_bytes:(n + 1) * frame_bytes]):
                n += 1
            if n:
                x = torch.frombuffer(chunk, dtype=torch.uint8)[:n * frame_bytes].to(device)
                pbar.update(n)
                yield x.view(n, 27, 48, 3).permute(0, 3, 1, 2).float() / 255.0
    finally:
        pbar.close()
        proc.stdout.close()
        rc = proc.wait()
        errlog.seek(0)
        err = errlog.read().decode("utf-8", "replace").strip()
        errlog.close()
        if rc != 0:
            raise RuntimeError(f"scene detection decoder failed (exit code {rc}): {err[-300:]}")


# ---------------------------------------------------------------- colour conversion on the GPU

_MATRIX_CACHE = {}


def _matrices(kr, kb, device):
    """(YCbCr -> RGB, RGB -> YCbCr) 3x3 matrices for luma coefficients kr, kb, on `device`."""
    key = (kr, kb, str(device))
    if key not in _MATRIX_CACHE:
        kg = 1.0 - kr - kb
        to_rgb = torch.tensor([
            [1.0, 0.0, 2 * (1 - kr)],
            [1.0, -2 * (1 - kb) * kb / kg, -2 * (1 - kr) * kr / kg],
            [1.0, 2 * (1 - kb), 0.0],
        ], dtype=torch.float32, device=device)
        to_yuv = torch.tensor([
            [kr, kg, kb],
            [-kr / (2 * (1 - kb)), -kg / (2 * (1 - kb)), 0.5],
            [0.5, -kg / (2 * (1 - kr)), -kb / (2 * (1 - kr))],
        ], dtype=torch.float32, device=device)
        _MATRIX_CACHE[key] = (to_rgb, to_yuv)
    return _MATRIX_CACHE[key]


def yuv_to_rgb(raw, w, h, ten_bit, kr, kb, full_range):
    """NV12 / P010 bytes (on the GPU) -> float RGB CHW in [0, 1]."""
    if ten_bit:
        # P010: 10 bits in the top bits of each little-endian 16-bit sample
        v = raw.view(torch.int16).to(torch.int32).bitwise_and_(0xFFFF).float().mul_(1.0 / 64.0)
        y_off, y_rng, c_off, c_rng, vmax = 64, 876, 512, 896, 1023
    else:
        v = raw.float()
        y_off, y_rng, c_off, c_rng, vmax = 16, 219, 128, 224, 255
    if full_range:
        y_off, y_rng, c_rng = 0, vmax, vmax
    y = v[:h * w].view(1, h, w).sub(y_off).mul_(1.0 / y_rng)
    uv = v[h * w:].view(h // 2, w // 2, 2).permute(2, 0, 1).sub(c_off).mul_(1.0 / c_rng)
    uv = F.interpolate(uv.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]
    to_rgb, _ = _matrices(kr, kb, raw.device)
    return torch.mm(to_rgb, torch.cat((y, uv)).view(3, -1)).view(3, h, w).clamp_(0.0, 1.0)


def rgb_to_p010(rgb, kr, kb, full_range):
    """float RGB CHW in [0, 1] (on the GPU) -> P010 bytes (uint8, on the GPU)."""
    _, h, w = rgb.shape
    _, to_yuv = _matrices(kr, kb, rgb.device)
    yuv = torch.mm(to_yuv, rgb.reshape(3, -1)).view(3, h, w)
    ys, yo, cs = (1023.0, 0.0, 1023.0) if full_range else (876.0, 64.0, 896.0)
    c = F.avg_pool2d(yuv[1:3].unsqueeze(0), 2)[0]                          # 4:2:0
    y16 = yuv[0].mul(ys).add_(yo).round_().clamp_(0, 1023).mul_(64.0)
    c16 = c.mul(cs).add_(512.0).round_().clamp_(0, 1023).mul_(64.0).permute(1, 2, 0)  # interleaved UV
    # float -> int32 -> int16 keeps the 16-bit pattern of values >= 32768 (little-endian P010 samples)
    out = torch.empty(h * w * 3 // 2, dtype=torch.int16, device=rgb.device)
    out[:h * w].view(h, w).copy_(y16.to(torch.int32))
    out[h * w:].view(h // 2, w // 2, 2).copy_(c16.to(torch.int32))
    return out.view(torch.uint8)


# ---------------------------------------------------------------- iw3 setup

def load_iw3(model, batch_size, extra_args):
    os.chdir(NUNIF_DIR)    # iw3 resolves some resources relative to its checkout
    from iw3 import utils as IU
    from iw3 import models  # noqa: F401  (registers the stereo models)

    stereo_args = [a for a in IW3_ARGS if a != "--yes"]
    args = IU.create_parser().parse_args(
        ["-i", "dummy.mp4", "-o", "dummy_out.mp4", "--depth-model", model,
         *stereo_args, "--batch-size", str(batch_size), *extra_args])
    IU.set_state_args(args)
    if args.warp_steps is None:
        args.warp_steps = IU.calc_auto_warp_steps(method=args.method, divergence=args.divergence,
                                                  synthetic_view=args.synthetic_view)
    depth_model = args.state["depth_model"]
    if depth_model.get_name() != "VideoDepthAnything":
        raise ValueError(f"fast_sbs.py supports the Video Depth Anything models only (got {model})")
    depth_model.load(gpu=args.gpu, resolution=args.resolution, limit_resolution=args.limit_resolution)
    args.mapper = IU.resolve_mapper_name(mapper=args.mapper, foreground_scale=args.foreground_scale,
                                         metric_depth=depth_model.is_metric(), mapper_type=args.mapper_type)
    side_model = IU.create_stereo_model(
        args.method,
        divergence=args.divergence * (2.0 if args.synthetic_view in {"right", "left"} else 1.0),
        device_id=args.gpu[0], inpaint_model=args.inpaint_model)
    return IU, args, depth_model, side_model


# ---------------------------------------------------------------- pipeline

def nvenc_qps(qp):
    """(P, B, I) QPs for a constant-QP encode, as ffmpeg's hevc_nvenc derives them from -qp
    (i_quant_factor 0.8, b_quant_factor 1.25 + offset 1.25). PyNvVideoCodec takes "P,B,I"."""
    return qp, int(qp * 1.25 + 1.25 + 0.5), int(qp * 0.8 + 0.5)


def hevc_metadata_filter(info):
    """hevc_metadata bitstream filter that writes the source's colour description into the VUI
    (NVENC writes none). Unknown primaries/transfer of BT.709 and BT.601 video get the matrix's
    standard ones, as players assume anyway."""
    standard = info["matrix"] if info["matrix"] in ("bt709", "smpte170m") else None
    prim = info["primaries"] if info["primaries"] not in (None, "unknown") else standard
    trc = info["transfer"] if info["transfer"] not in (None, "unknown") else standard
    vui = [f"matrix_coefficients={H273_MATRIX[info['matrix']]}",
           f"video_full_range_flag={int(info['full_range'])}"]
    if prim in H273_PRIMARIES:
        vui.append(f"colour_primaries={H273_PRIMARIES[prim]}")
    if trc in H273_TRANSFER:
        vui.append(f"transfer_characteristics={H273_TRANSFER[trc]}")
    return "hevc_metadata=" + ":".join(vui)


class Encoder:
    """SBS frames (float RGB on the GPU) -> P010 -> NVENC straight from VRAM -> Matroska file.

        enc = Encoder(out_path, info, qp, preset)
        enc.put(rgb)          # any thread; blocks while ENC_QUEUE frames wait for NVENC
        enc.close()           # finishes the file; raises if encoding failed or no frame came

    NVENC (PyNvVideoCodec) starts with the first frame, whose size sets the output size. The
    frames never leave the GPU; only the compressed HEVC stream (a few MB/s) reaches the CPU, where
    PyAV writes it. (Piping raw 7680x2160 frames into an ffmpeg process cost several CPU cores.)
    Settings match the earlier ffmpeg hevc_nvenc command: constant QP with ffmpeg's I/B offsets,
    preset, high quality tuning, 3 B-frames, GOP 250, Main 10, source colour description.
    """

    def __init__(self, out_path, info, qp, preset):
        self.out_path, self.info, self.qp, self.preset = out_path, info, qp, preset
        self.kr, self.kb = MATRICES[info["matrix"]]
        self.vui_filter = hevc_metadata_filter(info)
        self.time_base = Fraction(info["fps"].denominator, info["fps"].numerator)
        self.stream_cuda = torch.cuda.Stream()
        self.enc = None                                     # NVENC, created by the first put()
        self.bsf = self.container = self.stream = None      # set up from the first packet
        self.frames = 0                                     # frames put
        self.frames_in = 0                                  # frames handed to NVENC
        self.packets_out = 0

        self.todo = queue.Queue(maxsize=ENC_QUEUE)
        self.error = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _start(self, width, height):
        import PyNvVideoCodec as nvc
        self.enc = nvc.CreateEncoder(
            width, height, "P010", False, codec="hevc", preset=self.preset.upper(),
            tuning_info="high_quality", rc="constqp", constqp=",".join(map(str, nvenc_qps(self.qp))),
            bf=ENC_BFRAMES, gop=ENC_GOP, fps=max(1, round(float(self.info["fps"]))),
            cudastream=self.stream_cuda.cuda_stream)
        self.shape = (height * 3 // 2, width)   # P010 as one (Y rows + UV rows) x width uint16 image
        self.pic_params = nvc.NV_ENC_PIC_PARAMS

    @staticmethod
    def _parse(data):
        """An HEVC stream as PyAV parses it from one packet (an IDR, which carries VPS/SPS/PPS)."""
        import io
        import av
        return av.open(io.BytesIO(bytes(data)), format="hevc").streams.video[0]

    def _open(self, first_filtered):
        # The output stream copies everything (profile, 10-bit format, colour tags, extradata)
        # from the first packet after hevc_metadata, parsed by ffmpeg. opaque=True keeps the
        # template's decoder as the stream codec: a new stream would otherwise open an encoder
        # (libx265) that PyAV never uses.
        import av
        template = self._parse(first_filtered)
        self.container = av.open(str(self.out_path), "w", format="matroska")
        self.stream = self.container.add_stream_from_template(template, opaque=True)
        self.stream.codec_context.thread_count = 1

    def _mux(self, packets):
        import av
        for p in packets:
            if self.bsf is None:
                self.bsf = av.BitStreamFilterContext(self.vui_filter, self._parse(p["data"]))
            pkt = av.Packet(p["data"])
            pkt.pts = p["timestamp"]           # input frame index, packets come in coded order
            pkt.dts = self.packets_out - ENC_BFRAMES  # monotonic, never above pts with <= 3 B-frames
            pkt.duration = 1
            pkt.is_keyframe = p["picture_type"] == NV_ENC_PIC_TYPE_IDR
            self.packets_out += 1
            for out in self.bsf.filter(pkt):
                if self.container is None:
                    self._open(out)
                self._write(out)

    def _write(self, pkt):
        # the filter keeps the timestamps but not their time base
        pkt.time_base = self.time_base
        pkt.stream = self.stream
        self.container.mux(pkt)

    def _worker(self):
        try:
            while True:
                item = self.todo.get()
                if item is None:
                    break
                frame, event = item
                event.synchronize()
                params = self.pic_params()
                params.inputTimeStamp = self.frames_in
                self.frames_in += 1
                self._mux(self.enc.Encode(frame.view(torch.uint16).view(self.shape), params))
                del frame, item
            if self.enc is not None:
                self._mux(self.enc.EndEncode())
            if self.bsf is not None:
                for out in self.bsf.filter(None):
                    self._write(out)
            if self.container is not None:
                self.container.close()
        except Exception as e:  # noqa: BLE001  (reported by the main thread)
            self.error = e
            # unblock a producer waiting for queue space
            while True:
                try:
                    self.todo.get_nowait()
                except queue.Empty:
                    break

    def put(self, rgb):
        if self.error is not None:
            raise RuntimeError(f"encoder failed: {self.error!r}")
        if self.enc is None:
            self._start(rgb.shape[2], rgb.shape[1])
        p010_gpu = rgb_to_p010(rgb, self.kr, self.kb, self.info["full_range"])
        self.frames += 1
        # the encoder reads the frame on its own stream: keep its memory reserved until then
        p010_gpu.record_stream(self.stream_cuda)
        event = torch.cuda.Event()
        event.record()
        while True:
            try:
                self.todo.put((p010_gpu, event), timeout=0.5)
                return
            except queue.Full:
                if self.error is not None:
                    raise RuntimeError(f"encoder failed: {self.error!r}") from self.error

    def close(self):
        if self.error is None:
            self.todo.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"encoder failed: {self.error!r}") from self.error
        if self.frames == 0:
            raise RuntimeError("no frames to encode")

    def kill(self):
        # the output is discarded after a failure; let the worker finish what it has, if it can
        try:
            self.todo.put_nowait(None)
        except queue.Full:
            pass


class Decoder:
    """ffmpeg NVDEC process -> reader thread -> pinned host buffers -> GPU."""

    def __init__(self, src, info, device):
        w, h = info["width"], info["height"]
        self.pix_fmt = "p010le" if info["ten_bit"] else "nv12"
        self.frame_bytes = w * h * 3 // 2 * (2 if info["ten_bit"] else 1)
        self.device = device
        read_fd, write_fd = big_pipe(DECODER_PIPE_BYTES)
        try:
            self.proc = subprocess.Popen(decoder_cmd(src, info["fps"], "", self.pix_fmt), stdout=write_fd)
        finally:
            os.close(write_fd)       # the child has its own copy; EOF arrives when it exits
        self.stdout = open(read_fd, "rb", buffering=0)
        self.slots = [torch.empty(self.frame_bytes, dtype=torch.uint8).pin_memory() for _ in range(IN_SLOTS)]
        self.free = queue.Queue()
        for i in range(IN_SLOTS):
            self.free.put((i, None))
        self.ready = queue.Queue()
        self.error = None
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        try:
            while True:
                slot, event = self.free.get()
                if event is not None:
                    event.synchronize()          # GPU copy of the previous frame in this slot done
                if not read_exact(self.stdout, memoryview(self.slots[slot].numpy())):
                    break
                self.ready.put(slot)
        except Exception as e:  # noqa: BLE001
            self.error = e
        self.ready.put(None)

    def __iter__(self):
        while True:
            slot = self.ready.get()
            if slot is None:
                if self.error is not None:
                    raise RuntimeError(f"decoder failed: {self.error}")
                return
            raw = self.slots[slot].to(self.device, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            self.free.put((slot, event))
            yield raw

    def close(self):
        self.stdout.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"decoder failed (exit code {rc})")

    def kill(self):
        self.proc.kill()


class StereoStage:
    """Runs `work(depths, frames, flush)` for each window of frames whose depth is ready.

        stage = StereoStage(work, device, overlapped)
        stage.submit(depths, frames)       # frames: (raw YUV tensor, frame index) pairs
        stage.submit([], [], flush=True)   # last call
        stage.close()                      # waits for the work; raises if it failed

    Serial: `work` runs inline. Overlapped: it runs on a worker thread with its own CUDA stream,
    so the GPU warps one window while the depth model computes the next. Only STEREO_JOBS windows
    are handed over at a time (each keeps its frames in VRAM until warped); submit() waits for a
    free slot. Both give byte-identical output. A failure in the worker surfaces in the next
    submit() or in close().
    """

    def __init__(self, work, device, overlapped):
        self.work = work
        self.error = None
        self.stream = torch.cuda.Stream(device) if overlapped else None
        if overlapped:
            self.jobs = queue.Queue()
            self.slots = threading.Semaphore(STEREO_JOBS)
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def submit(self, depths, frames, flush=False):
        if self.stream is None:
            self.work(depths, frames, flush)
            return
        if not depths and not flush:
            return
        # tensors made on the main stream and used on the stereo stream: keep their memory
        # reserved until the stereo stream is done with them
        for d in depths:
            d.record_stream(self.stream)
        for x, _ in frames:
            x.record_stream(self.stream)
        ready = torch.cuda.Event()
        ready.record()
        while not self.slots.acquire(timeout=0.5):
            self._raise_worker_error()
        self._raise_worker_error()
        self.jobs.put((depths, frames, flush, ready))

    def close(self):
        if self.stream is None:
            return
        self._raise_worker_error()
        self.jobs.put(None)
        self.thread.join()
        self._raise_worker_error()

    def _raise_worker_error(self):
        if self.error is not None:
            raise RuntimeError(f"stereo worker failed: {self.error!r}") from self.error

    def _worker(self):
        try:
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                while (job := self.jobs.get()) is not None:
                    depths, frames, flush, ready = job
                    self.stream.wait_event(ready)
                    self.work(depths, frames, flush)
                    del depths, frames, job        # free the window's VRAM before the next one
                    self.slots.release()
        except BaseException as e:  # noqa: BLE001  (re-raised in the main thread)
            self.error = e


def compile_models(args, IU, depth_model, side_model, device):
    """--compile (e.g. via IW3_EXTRA_ARGS), applied the way iw3's process_video_full does it.

    Returns (context to keep open while converting, side model to use)."""
    ctx = contextlib.ExitStack()
    if not args.compile:
        return ctx, side_model
    from nunif.models import compile_model
    from nunif.models.utils import check_compile_support
    if not check_compile_support(device):
        print("warning: --compile requested but torch.compile does not work here "
              "(on Windows it needs the triton-windows package); running uncompiled", flush=True)
        return ctx, side_model
    if side_model is not None and not hasattr(side_model, "compile_context"):
        side_model = compile_model(side_model)
    ctx.enter_context(depth_model.compile_context(enabled=True))
    ctx.enter_context(IU.try_compile_context(side_model, enabled=True))
    return ctx, side_model


@torch.inference_mode()
def convert(src, out_path, model="VDA_S", batch_size=2, qp=20, preset="p5", extra_args=()):
    src = Path(src).resolve()
    out_path = Path(out_path).resolve()
    info = probe(src)
    w, h = info["width"], info["height"]
    if w % 2 or h % 2:
        raise ValueError(f"odd frame size {w}x{h} is not supported by the fast path")
    kr, kb = MATRICES[info["matrix"]]

    IU, args, depth_model, side_model = load_iw3(model, batch_size, list(extra_args))
    device = args.state["device"]
    enable_amp = not args.disable_amp

    t0 = time.time()
    segment_pts = detect_scenes(src, info, device) if args.scene_detect else set()
    t_scene = time.time() - t0

    depth_model.reset()
    if args.ema_normalize and args.max_fps >= 15:
        depth_model.enable_ema(decay=args.ema_decay, buffer_size=args.ema_buffer)

    compile_ctx, side_model = compile_models(args, IU, depth_model, side_model, device)

    total = int(info["duration"] * info["fps"]) or None
    pbar = tqdm(total=total, desc=src.name[:40], unit="f", ncols=100)
    encoder = Encoder(out_path, info, qp, preset)

    def to_rgb(raw):
        return yuv_to_rgb(raw, w, h, info["ten_bit"], kr, kb, info["full_range"])

    def emit(left, right):
        encoder.put(IU.postprocess_image(left, right, args))
        pbar.update(1)

    def warp(depths, frames, flush):
        """Stereo warp + SBS assembly + encode for frames whose depth is ready."""
        for i in range(0, len(depths), args.batch_size):
            batch = frames[i:i + args.batch_size]
            reset_pts = [idx in segment_pts for _, idx in batch]
            # .half() as when the queue held fp16 RGB, so the output stays the same
            x_srcs = torch.stack([to_rgb(raw).half() for raw, _ in batch]).float()
            left, right = IU.apply_divergence(torch.stack(depths[i:i + args.batch_size]), x_srcs, args,
                                              side_model, reset_pts=reset_pts)
            if left is not None:
                for le, re in zip(left, right):
                    emit(le, re)
        if flush and hasattr(side_model, "flush"):
            left, right = side_model.flush(enable_amp=enable_amp)
            if left is not None:
                for le, re in zip(left, right):
                    emit(le, re)

    stereo = StereoStage(warp, device, overlapped=os.environ.get("FAST_SBS_STREAMS", "1") != "0")
    decoder = Decoder(src, info, device)
    # Frames waiting for their depth stay in the decoder's YUV 4:2:0 form (P010: 3 bytes per
    # pixel, half of RGB fp16) and are converted to RGB again for the warp; VDA keeps up to a
    # 32-frame window of them, which at 4K is the largest block of VRAM.
    waiting = deque()          # (raw YUV frame on GPU, frame index) waiting for their depth

    def depth(rgbs, idxs):
        """Depth for one batch; VDA returns the depths of earlier frames once its window allows."""
        return depth_model.infer_with_normalize(torch.stack(rgbs), idxs, segment_pts, enable_amp=enable_amp,
                                                edge_dilation=args.edge_dilation, depth_aa=args.depth_aa)

    def to_stereo(depths, flush=False):
        stereo.submit(depths, [waiting.popleft() for _ in depths], flush)

    ok = False
    try:
        rgbs, idxs = [], []
        for idx, raw in enumerate(decoder):
            waiting.append((raw, idx))
            rgbs.append(to_rgb(raw))
            idxs.append(idx)
            if len(rgbs) == args.batch_size:
                to_stereo(depth(rgbs, idxs))
                rgbs, idxs = [], []
        if rgbs:
            to_stereo(depth(rgbs, idxs))
        to_stereo(depth_model.flush_with_normalize(enable_amp=enable_amp, edge_dilation=args.edge_dilation,
                                                   depth_aa=args.depth_aa), flush=True)
        stereo.close()
        decoder.close()
        encoder.close()
        ok = True
    finally:
        pbar.close()
        compile_ctx.close()
        if not ok:
            decoder.kill()
            encoder.kill()

    elapsed = time.time() - t0
    frames_out = encoder.frames
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"{frames_out} frames in {elapsed:.1f} s ({frames_out / max(elapsed - t_scene, 1e-6):.1f} fps "
          f"after {t_scene:.1f} s scene detection), peak torch VRAM {peak:.1f} GB")
    return frames_out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="output video (.mkv), video track only")
    parser.add_argument("--model", default="VDA_S", help="VDA_S (default), VDA_B, VDA_L")
    parser.add_argument("--batch-size", type=int, default=2, help="frames per depth/warp batch (default 2, as iw3; larger was slower)")
    parser.add_argument("--qp", type=int, default=20, help="NVENC constant QP (default 20, as iw3 --crf 20)")
    parser.add_argument("--preset", default="p5", help="NVENC preset (default p5)")
    args, extra = parser.parse_known_args()   # unknown options go to iw3 (e.g. --low-vram)
    extra += os.environ.get("IW3_EXTRA_ARGS", "").split()
    convert(args.input, args.output, model=args.model, batch_size=args.batch_size,
            qp=args.qp, preset=args.preset, extra_args=extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
