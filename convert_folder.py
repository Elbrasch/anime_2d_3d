"""Convert every video under an input folder to VR Full SBS with iw3, mirroring the folder tree.

    python convert_folder.py <input_folder> <output_folder> [--model VDA_S]

For each video, input/a/b/movie.mkv becomes output/a/b/movie_LRF_Full_SBS.mkv (the suffix lets VR
players detect the layout). Videos whose output already exists are skipped, so the script can be
re-run after adding videos or after an interruption. A single file also works as input.

The 3D conversion runs through fast_sbs.py (iw3's stereo code with GPU decode/encode in separate
processes) and falls back to plain iw3 if that fails; --engine picks one explicitly.
iw3 keeps only the first audio track and no subtitles, so after the 3D conversion the SBS video is
remuxed with every audio, subtitle and chapter track of the source (stream copy, no re-encode).
Each result is written to a .partial file first and renamed only after its video covers the same
time span as the source video.
Extra iw3 options can be passed with the IW3_EXTRA_ARGS environment variable (e.g. --low-vram).
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUNIF_DIR = ROOT / "vendor" / "nunif"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".ts", ".m2ts", ".wmv", ".mpg", ".mpeg"}
SUFFIX = "_LRF_Full_SBS"
TMP_DIR_NAME = ".sbs_tmp"

# Stereo settings (1080p source -> 3840x1080 Full SBS):
#   row_flow_v3, divergence 2 / convergence 0.5   iw3 defaults for the stereo warp
#   synthetic-view both                          spreads warp artifacts over both eyes
#   edge-dilation 2                              fewer halos around people, less edge accuracy
#   ema-normalize + scene-detect                 steadier depth scale, reset at cuts
#   max-fps 128                                  keep the source frame rate (iw3 caps at 30 by default)
#   hevc_nvenc 10-bit, qp 20                     fast GPU encode, small banding
IW3_ARGS = [
    "--method", "row_flow_v3", "--divergence", "2.0", "--convergence", "0.5",
    "--synthetic-view", "both", "--edge-dilation", "2",
    "--ema-normalize", "--ema-decay", "0.9", "--scene-detect",
    "--max-fps", "128",
    "--video-codec", "hevc_nvenc", "--preset", "p5", "--crf", "20", "--pix-fmt", "yuv420p10le",
    "--yes",
]


def ffprobe(path, *args):
    out = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "json", str(path)],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return json.loads(out)


def video_info(path):
    """(frame count, start time, end time) of the first video stream, read from the packets (no decoding)."""
    info = ffprobe(path, "-select_streams", "v:0", "-show_entries", "packet=pts_time,duration_time")
    pts = [(float(p["pts_time"]), float(p.get("duration_time", 0) or 0))
           for p in info["packets"] if p.get("pts_time") not in (None, "N/A")]
    start = min(t for t, _ in pts)
    end = max(t + d for t, d in pts)
    return len(pts), start, end


def frame_duration(path):
    num, den = ffprobe(path, "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate")["streams"][0]["r_frame_rate"].split("/")
    return float(den) / float(num)


def find_videos(input_path, output_dir):
    if input_path.is_file():
        return [(input_path, Path(input_path.name))]
    videos = []
    for dirpath, dirnames, filenames in os.walk(input_path):
        here = Path(dirpath)
        # never descend into the output folder (it may sit inside the input folder)
        dirnames[:] = sorted(d for d in dirnames if (here / d).resolve() != output_dir)
        for name in sorted(filenames):
            path = here / name
            if path.suffix.lower() in VIDEO_EXTS and SUFFIX not in path.stem:
                videos.append((path, path.relative_to(input_path)))
    return videos


def run_iw3(src, dst, model):
    extra = shlex.split(os.environ.get("IW3_EXTRA_ARGS", ""), posix=False)
    cmd = [sys.executable, "-m", "iw3", "-i", str(src), "-o", str(dst), "--depth-model", model,
           *IW3_ARGS, *extra]
    subprocess.run(cmd, cwd=NUNIF_DIR, check=True)


def run_fast(src, dst, model):
    """fast_sbs.py: same stereo settings, NVDEC decode and NVENC encode in their own processes."""
    subprocess.run([sys.executable, str(ROOT / "fast_sbs.py"), str(src), str(dst), "--model", model],
                   check=True)


def make_sbs(src, dst, model, engine):
    if engine == "iw3" or not model.startswith("VDA"):
        run_iw3(src, dst, model)
        return
    try:
        run_fast(src, dst, model)
    except subprocess.CalledProcessError as e:
        if engine == "fast":
            raise
        print(f"fast path failed ({e}); retrying with plain iw3", flush=True)
        dst.unlink(missing_ok=True)
        run_iw3(src, dst, model)


def remux(src, sbs, out_partial):
    """SBS video from iw3 + all audio/subtitle/chapter streams of the source, aligned in time.

    iw3 starts its video at t=0; the source video may start a little later, so the SBS track is
    shifted by the difference to keep the original audio sync.
    """
    src_info = video_info(src)
    _, sbs_start, _ = video_info(sbs)
    offset = src_info[1] - sbs_start
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-itsoffset", f"{offset:.6f}", "-i", str(sbs), "-i", str(src),
        "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
        "-map_metadata", "1", "-map_chapters", "1",
        "-c", "copy", "-metadata:s:v:0", "title=3D Full SBS",
        "-f", "matroska", str(out_partial),
    ]
    subprocess.run(cmd, check=True)
    return src_info


def convert(src, out_path, tmp_dir, model, engine):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sbs = tmp_dir / f"{out_path.stem}.mkv"
    out_partial = out_path.with_name(out_path.name + ".partial")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    make_sbs(src, sbs, model, engine)
    src_frames, src_start, src_end = remux(src, sbs, out_partial)

    # The SBS track must cover the same time span as the source video, or the audio drifts.
    # iw3 writes constant frame rate, so a source with timestamp gaps or variable frame rate can
    # gain or lose frames while staying in sync; that is only a warning.
    out_frames, out_start, out_end = video_info(out_partial)
    tolerance = 2 * frame_duration(src)
    if abs(out_start - src_start) > tolerance or abs(out_end - src_end) > tolerance:
        raise RuntimeError(f"timing mismatch: source video {src_start:.3f}-{src_end:.3f} s, "
                           f"output {out_start:.3f}-{out_end:.3f} s (kept {out_partial.name} for inspection)")
    if out_frames != src_frames:
        print(f"note: {src_frames} source frames, {out_frames} output frames "
              f"(source has timestamp gaps or variable frame rate; timing matches)")
    out_partial.replace(out_path)
    sbs.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="input folder (searched recursively) or a single video")
    parser.add_argument("output", type=Path, help="output folder; the input folder tree is recreated here")
    parser.add_argument("--model", default="VDA_S", help="iw3 depth model (default VDA_S; VDA_L needs get_vda_l.bat)")
    parser.add_argument("--engine", choices=["auto", "fast", "iw3"], default="auto",
                        help="fast: fast_sbs.py (GPU decode/encode pipeline); iw3: plain iw3; "
                             "auto (default): fast, falling back to iw3 if it fails")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1
    tmp_dir = output_dir / TMP_DIR_NAME

    videos = find_videos(input_path, output_dir)
    todo = []
    for src, rel in videos:
        out_path = output_dir / rel.parent / f"{rel.stem}{SUFFIX}.mkv"
        if out_path.exists():
            print(f"skip (exists): {rel}")
        else:
            todo.append((src, rel, out_path))
    print(f"{len(videos)} videos found, {len(videos) - len(todo)} already converted, {len(todo)} to do.")

    failed = []
    for i, (src, rel, out_path) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {rel} -> {out_path.relative_to(output_dir)}", flush=True)
        t0 = time.time()
        try:
            convert(src, out_path, tmp_dir, args.model, args.engine)
        except KeyboardInterrupt:
            print("\nInterrupted. Run the same command again to continue.")
            return 130
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"FAILED: {rel}: {e}")
            failed.append(rel)
            continue
        print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)

    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()
    print(f"\nFinished: {len(todo) - len(failed)} converted, {len(failed)} failed.")
    for rel in failed:
        print(f"  failed: {rel}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
