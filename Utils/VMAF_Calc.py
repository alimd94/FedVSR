
#for running this code you need to have ffmpeg and vmaf installed and accessible from your PATH

import os
import json
import subprocess
from pathlib import Path
from PIL import Image   # pip install Pillow


REF_ROOT   = "path to reference png folders" 
DIST_ROOT  = "path to distorted png folders"
YUV_DIR    = "path to create yuv_videos"
MODEL_PATH = "path to vmaf_v0.6.1.json"   # or vmaf_4k_v0.6.1.json
FPS        = 30
PIX_FMT    = "yuv420p"

os.makedirs(YUV_DIR, exist_ok=True)


def get_png_info(folder: Path):
    """Return (width, height, nframes, pattern) for a folder of PNGs."""
    pngs = sorted([f for f in folder.iterdir() if f.suffix.lower() == ".png"])
    if not pngs:
        raise ValueError("no PNG files")

    # ---- resolution from first frame ----
    w, h = Image.open(pngs[0]).size

    # ---- guess pattern (%03d.png, %08d.png, …) ----
    name = pngs[0].stem
    # try to find the first run of digits
    import re
    m = re.search(r"\d+", name)
    if m:
        digits = len(m.group())
        pattern = f"%0{digits}d.png"
    else:
        pattern = "%d.png"  

    return w, h, len(pngs), pattern


video_names = sorted([d for d in os.listdir(REF_ROOT)
                     if os.path.isdir(os.path.join(REF_ROOT, d))])

vmaf_scores = []

for video in video_names:
    print(f"\n=== Processing video: {video} ===")
    ref_dir  = Path(REF_ROOT) / video
    dist_dir = Path(DIST_ROOT) / video

    if not ref_dir.exists():
        print(f"  reference folder missing: {ref_dir}")
        continue
    if not dist_dir.exists():
        print(f"distorted folder missing: {dist_dir}")
        continue

    try:
        w_ref, h_ref, n_ref, pat_ref = get_png_info(ref_dir)
        w_dis, h_dis, n_dis, pat_dis = get_png_info(dist_dir)
    except Exception as e:
        print(f"cannot read PNG info: {e}")
        continue


    if w_ref != w_dis or h_ref != h_dis:
        print(f"resolution mismatch – ref {w_ref}×{h_ref}, dist {w_dis}×{h_dis}")
        continue

 
    if n_ref != n_dis:
        print(f"frame-count mismatch – ref {n_ref}, dist {n_dis}")
        continue

    width, height, nframes = w_ref, h_ref, n_ref
    pattern = pat_ref   # both folders use the same pattern

    print(f"  → {width}×{height}, {nframes} frames, pattern '{pattern}'")


    ref_yuv  = Path(YUV_DIR) / f"{video}_ref.yuv"
    dist_yuv = Path(YUV_DIR) / f"{video}_dist.yuv"

    for src, dst in [(ref_dir, ref_yuv), (dist_dir, dist_yuv)]:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(src / pattern),
            "-pix_fmt", PIX_FMT,
            "-vframes", str(nframes),
            str(dst)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"  [FFMPEG ERROR] {e}")
            continue


    out_json = Path(YUV_DIR) / f"{video}_vmaf.json"

    vmaf_cmd = [
        "vmaf",
        "-r", str(ref_yuv),
        "-d", str(dist_yuv),
        "--width",  str(width),
        "--height", str(height),
        "--pixel_format", "420",
        "--bitdepth", "8",
        "--frame_cnt", str(nframes),
        "--model", f"path={MODEL_PATH}",
        "--output", str(out_json),
        "--json"
    ]

    try:
        subprocess.run(vmaf_cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"  [VMAF ERROR] {e}")
        continue


    try:
        with open(out_json) as f:
            data = json.load(f)
        mean_vmaf = data["pooled_metrics"]["vmaf"]["mean"]
        print(f"  VMAF = {mean_vmaf:.3f}")
        vmaf_scores.append(mean_vmaf)
    except Exception as e:
        print(f"  JSON ERROR {e}")


if vmaf_scores:
    avg = sum(vmaf_scores) / len(vmaf_scores)
    print(f"\nOverall average VMAF across {len(vmaf_scores)} videos: {avg:.3f}")
else:
    print("\nNo videos were successfully processed.")