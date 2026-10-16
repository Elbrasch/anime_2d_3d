"""Download every model a conversion uses, so the first conversion needs no network (run by setup.bat).

    python download_models.py

- Video Depth Anything code (torch hub) and the VDA_S weights, plus iw3's depth anti-aliasing model
- the stereo warp model set in convert_folder.IW3_ARGS (row_flow_v3)
- TransNetV2 for scene detection

Models load on the CPU, so this also works while the GPU is busy. VDA_L (CC-BY-NC-4.0) is not
downloaded here; get_vda_l.bat asks for the license first.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUNIF_DIR = ROOT / "vendor" / "nunif"
sys.path.insert(0, str(NUNIF_DIR))


def main():
    os.chdir(NUNIF_DIR)    # iw3 resolves some resources relative to its checkout
    from nunif.utils.transnetv2 import TransNetV2
    from nunif.utils.ui import TorchHubDir
    from iw3.hub_dir import HUB_MODEL_DIR
    from iw3.stereo_model_factory import create_stereo_model
    from iw3.video_depth_anything_model import VideoDepthAnythingModel
    from convert_folder import IW3_ARGS

    print("Video Depth Anything code (torch hub)", flush=True)
    VideoDepthAnythingModel.force_update()
    print("VDA_S weights + depth anti-aliasing model", flush=True)
    VideoDepthAnythingModel("VDA_S").load(gpu=-1)
    method = IW3_ARGS[IW3_ARGS.index("--method") + 1]
    print(f"stereo warp model ({method})", flush=True)
    create_stereo_model(method, divergence=2.0, device_id=-1)
    print("scene detection model (TransNetV2)", flush=True)
    with TorchHubDir(HUB_MODEL_DIR):
        TransNetV2().load()
    print("All models present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
