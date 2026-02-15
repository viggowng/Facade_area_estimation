# =========================================
# PIPELINE STAGE 2: DOWNLOAD + PROCESS PANOS
# =========================================

import os
import math
import pandas as pd
import requests
import numpy as np
import cv2
from PIL import Image

from Stage_0 import MAPILLARY_TOKEN, CSV_PATHS, IMAGE_PATHS

### ==== CONFIGURATION ==== ###

# Token
ACCESS_TOKEN = MAPILLARY_TOKEN

# Stage 1 output
INPUT_CSV = CSV_PATHS["road_network"] / "sample_points.csv"

# Output folders
RAW_DIR   = IMAGE_PATHS["raw_images"]
ROT_DIR   = IMAGE_PATHS["rotated_images"]
LEFT_DIR  = IMAGE_PATHS["left_facades"]
RIGHT_DIR = IMAGE_PATHS["right_facades"]

# Perspective crop settings
HFOV_DEG = 96.4
VFOV_DEG = 90     # field of view in degrees
OUT_W = 800      # output width in pixels
OUT_H = 600      # output height in pixels

# If False: do not re-download / re-process if outputs already exist
OVERWRITE = False

# Make sure folders exist
for d in [RAW_DIR, ROT_DIR, LEFT_DIR, RIGHT_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# SMALL BUT IMPORTANT HELPERS
# ============================================================

def get_thumb_url(image_id, access_token):
    """
    Asks Mapillary Graph API for the pano thumbnail URL.
    parameters:
      - thumb_2048_url (the actual image download URL)
      - is_pano (to ensure it is a pano)
    """
    r = requests.get(
        f"https://graph.mapillary.com/{image_id}",
        params={"fields": "thumb_2048_url,is_pano", "access_token": access_token},
        timeout=30
    )
    r.raise_for_status()
    meta = r.json()

    if not meta.get("is_pano", False):
        raise RuntimeError(f"Image {image_id} is not panoramic (is_pano=false).")

    url = meta.get("thumb_2048_url")
    if not url:
        raise RuntimeError(f"No thumb_2048_url returned for image {image_id}. Response: {meta}")

    return url


def download_pano(image_id, access_token, out_dir, overwrite=OVERWRITE):
    """
    Download pano JPG to raw_pano_images/<image_id>.jpg
    """
    out_path = os.path.join(out_dir, f"{image_id}.jpg")

    # Cache behaviour: if file exists, reuse it
    if (not overwrite) and os.path.exists(out_path):
        return out_path

    url = get_thumb_url(image_id, access_token)

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


# ============================================================
# PANORAMA -> ROTATED + LEFT/RIGHT PERSPECTIVE CROPS
# ============================================================

def equi_to_perspective(equi_img,
                        hfov_deg=HFOV_DEG, vfov_deg=VFOV_DEG,
                        yaw_deg=0, pitch_deg=0,
                        out_w=OUT_W, out_h=OUT_H):
    """
    Equirectangular pano -> rectilinear perspective.
    hfov_deg: horizontal FOV of the output camera
    vfov_deg: vertical FOV (optional). If None, derived from hfov + aspect.
    yaw_deg:   left/right (deg)
    pitch_deg: up/down (deg)
    """
    H, W = equi_img.shape[:2]

    hfov = np.deg2rad(hfov_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    if vfov_deg is None:
        # derive VFOV from HFOV and aspect ratio
        vfov = 2 * np.arctan((out_h / out_w) * np.tan(hfov / 2))
    else:
        vfov = np.deg2rad(vfov_deg)

    # focal lengths for rectilinear camera
    fx = (out_w / 2) / np.tan(hfov / 2)
    fy = (out_h / 2) / np.tan(vfov / 2)

    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x = (u - out_w / 2) / fx
    y = -(v - out_h / 2) / fy
    z = np.ones_like(x)

    # Normalize ray directions
    norm = np.sqrt(x*x + y*y + z*z)
    x /= norm; y /= norm; z /= norm

    # Pitch around X
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    y2 = y * cos_p - z * sin_p
    z2 = y * sin_p + z * cos_p
    x2 = x

    # Yaw around Y
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    x3 = x2 * cos_y + z2 * sin_y
    z3 = -x2 * sin_y + z2 * cos_y
    y3 = y2

    # Spherical angles
    theta = np.arctan2(x3, z3)                         # [-pi, pi]
    phi = np.arctan2(y3, np.sqrt(x3*x3 + z3*z3))       # [-pi/2, pi/2]

    # Map to equirectangular pixel coords
    map_x = (theta + np.pi) / (2 * np.pi) * W
    map_y = (np.pi/2 - phi) / np.pi * H

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    persp = cv2.remap(equi_img, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT)
    return persp



def process_image(raw_path, road_angle,
                  rot_dir, left_dir, right_dir,
                  do_rotate=True,
                  vfov_deg=VFOV_DEG, hfov_deg=HFOV_DEG, out_w=OUT_W, out_h=OUT_H,
                  overwrite=OVERWRITE):
    """
    This is your "data processing stage" after downloading:
      1) read pano (raw_path)
      2) rotate pano so that road direction is centered
      3) generate left/right rectilinear crops
      4) write outputs

    Output filenames are stable and keyed by the pano image_id (base name).
    """
    base = os.path.splitext(os.path.basename(raw_path))[0]  # image_id (string)

    rotated_path = os.path.join(rot_dir, f"rotated_{base}.jpg")
    left_path = os.path.join(left_dir, f"left_{base}.jpg")
    right_path = os.path.join(right_dir, f"right_{base}.jpg")

    # Cache behaviour: if all outputs exist, skip processing
    if (not overwrite) and all(os.path.exists(p) for p in [rotated_path, left_path, right_path]):
        return left_path, right_path, rotated_path

    pano = cv2.imread(raw_path)
    H, W = pano.shape[:2]
    print(f"[PANO] {raw_path} size = {W}x{H}  ratio W/H={W/H:.3f}")

    if pano is None:
        raise FileNotFoundError(f"Could not read pano: {raw_path}")

    H, W = pano.shape[:2]

    # -------------------------
    # 1) ROTATE TO ROAD CENTER
    # -------------------------
    # Mapillary panos are "wrapped" horizontally.
    # Rolling pixels horizontally is a cheap & stable way to rotate the pano.
    if do_rotate:
        shift_px = int((road_angle / 360.0) * W)
        pano_rot = np.roll(pano, -shift_px, axis=1)
    else:
        pano_rot = pano

    # Save rotated pano
    if overwrite or (not os.path.exists(rotated_path)):
        cv2.imwrite(rotated_path, pano_rot)

    # -------------------------
    # 2) LEFT + RIGHT VIEWS
    # -------------------------
    # LEFT façade (yaw -90)
    left_np = equi_to_perspective(
    pano_rot,
    hfov_deg=HFOV_DEG,
    vfov_deg=VFOV_DEG, 
    yaw_deg=-90,
    pitch_deg=0,
    out_w=OUT_W,
    out_h=OUT_H
)


    left_img = Image.fromarray(cv2.cvtColor(left_np, cv2.COLOR_BGR2RGB))
    if overwrite or (not os.path.exists(left_path)):
        left_img.save(left_path)

    # RIGHT façade (yaw +90)
    right_np = equi_to_perspective(
    pano_rot,
    hfov_deg=96.4,
    vfov_deg=90,   # keep the old VFOV
    yaw_deg=90,
    pitch_deg=0,
    out_w=OUT_W,
    out_h=OUT_H
)

    right_img = Image.fromarray(cv2.cvtColor(right_np, cv2.COLOR_BGR2RGB))
    if overwrite or (not os.path.exists(right_path)):
        right_img.save(right_path)

    return left_path, right_path, rotated_path


# ============================================================
# MAIN: Reads CSV -> RAW + ROTATED + CROPS
# ============================================================

def download_images(input_csv, image_id_col="image_id"):
    """
    Pipeline logic:
      - read Stage1 CSV
      - for each row:
          * read image_id + road_angle
          * download pano
          * rotate + produce left/right crops

    Safe to re-run due to caching/overwrite logic in download_pano/process_image.
    """
    df = pd.read_csv(input_csv, dtype={"image_id": str})
    df["image_id"] = df["image_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    # Minimal requirements for this stage
    required = {image_id_col, "road_angle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    for i, row in df.iterrows():
        image_id = row[image_id_col]
        road_angle = row["road_angle"]

        if pd.isna(image_id) or str(image_id).strip() == "":
            print(f"[{i+1}/{len(df)}] SKIP: missing {image_id_col}")
            continue

        if pd.isna(road_angle):
            print(f"[{i+1}/{len(df)}] SKIP: image_id={image_id} has no road_angle")
            continue

        try:
            # 1) Download pano
            raw_path = download_pano(image_id, ACCESS_TOKEN, RAW_DIR, overwrite=OVERWRITE)

            # 2) Rotate + crops
            left_path, right_path, rotated_path = process_image(
                raw_path,
                road_angle=road_angle,
                rot_dir=ROT_DIR,
                left_dir=LEFT_DIR,
                right_dir=RIGHT_DIR,
                do_rotate=True,
                vfov_deg=VFOV_DEG,
                hfov_deg=HFOV_DEG,
                out_w=OUT_W,
                out_h=OUT_H,
            )

            print(f"[{i+1}/{len(df)}] OK: image_id={image_id}")
        except Exception as e:
            print(f"[{i+1}/{len(df)}] FAIL: image_id={image_id} -> {e}")



# Runs the pipeline
if __name__ == "__main__":
    download_images(INPUT_CSV)
    print("\nDone, ready for Stage 3: Segmentation.")
