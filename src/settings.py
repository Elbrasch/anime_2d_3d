"""Project layout and the stereo settings shared by both conversion engines (fast_sbs.py and plain iw3)."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
NUNIF_DIR = PROJECT_DIR / "vendor" / "nunif"     # nunif (iw3) checkout and its models, from setup.bat

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
