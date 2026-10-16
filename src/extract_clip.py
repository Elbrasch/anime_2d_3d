"""Cut a short test clip from every video in to_convert/ into clips/.

The clip is re-encoded (not stream-copied) so it starts exactly at the requested time instead of
the nearest keyframe. Quality is near-lossless so the clip is a fair stand-in for the source.

    python extract_clip.py                 # 30 s from the middle of each video
    python extract_clip.py --start 1:30    # 30 s starting at 1:30
    python extract_clip.py --duration 10 --force
    python extract_clip.py movie.mkv --start 15:00 52:00 --duration 10   # several clips from one file
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from settings import PROJECT_DIR

SRC_DIR = PROJECT_DIR / "to_convert"
CLIP_DIR = PROJECT_DIR / "clips"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".ts", ".wmv"}


def parse_time(text):
    """'90', '1:30' or '0:01:30' -> seconds."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def probe_duration(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def pick_start(duration, clip_len, requested):
    if requested is not None:
        start = requested
    else:
        start = (duration - clip_len) / 2  # middle of the video is more typical than the intro
    return max(0.0, min(start, duration - clip_len))


def extract(video, start, clip_len, out_path):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{clip_len:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", type=Path, help="video files or folders (default: to_convert/)")
    parser.add_argument("--start", type=parse_time, nargs="+",
                        help="clip start(s) (s, mm:ss or hh:mm:ss); one clip per start; default: middle")
    parser.add_argument("--duration", type=float, default=30.0, help="clip length in seconds (default 30)")
    parser.add_argument("--force", action="store_true", help="overwrite existing clips")
    args = parser.parse_args()

    videos = []
    for item in args.inputs or [SRC_DIR]:
        if item.is_dir():
            videos += sorted(p for p in item.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        else:
            videos.append(item)
    if not videos:
        print(f"No videos in {', '.join(map(str, args.inputs or [SRC_DIR]))}")
        return 1
    CLIP_DIR.mkdir(exist_ok=True)

    for video in videos:
        duration = probe_duration(video)
        clip_len = min(args.duration, duration)
        for requested in args.start or [None]:
            start = pick_start(duration, clip_len, requested)
            tag = f"_{int(start) // 60:03d}m{int(start) % 60:02d}s" if args.start else ""
            out_path = CLIP_DIR / f"{video.stem}{tag}_clip{int(args.duration)}s.mp4"
            if out_path.exists() and not args.force:
                print(f"skip (exists): {out_path.name}")
                continue
            print(f"{video.name}: {start:.1f}s .. {start + clip_len:.1f}s of {duration:.1f}s -> {out_path.name}")
            extract(video, start, clip_len, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
