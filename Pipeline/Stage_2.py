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

# ============================================================
# CONFIG 
# ============================================================

ACCESS_TOKEN = "MLY|25078958115060460|63887214bce846039a87cac96f71e44a"

# Stage 1 output
INPUT_CSV = r"C:\Users\viggo\OneDrive - Universiteit Utrecht\Year 2\Thesis\Python_thesis\Facade_area_estimation\Results_csv\road_network\Frankendael_sample_points.csv"

# Output folders (match your pipeline naming)
BASE_RESULTS = r"C:\Users\viggo\OneDrive - Universiteit Utrecht\Year 2\Thesis\Python_thesis\Facade_area_estimation\Results_images"

RAW_DIR   = os.path.join(BASE_RESULTS, "raw_images")
ROT_DIR   = os.path.join(BASE_RESULTS, "rotated_images")
LEFT_DIR  = os.path.join(BASE_RESULTS, "left_facades")
RIGHT_DIR = os.path.join(BASE_RESULTS, "right_facades")

# Perspective crop settings
FOV_DEG = 90     # field of view in degrees
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


def download_pano(image_id, access_token, out_dir, overwrite=False):
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

def cyl_to_perspective(cyl_img, fov_deg=60, yaw_deg=0, pitch_deg=0,
                       out_w=800, out_h=600):
    """
    Convert a cylindrical pano image into a rectilinear perspective view.

    Notes:
    - yaw_deg: horizontal view direction
      * -90 = looking left
      * +90 = looking right
    - pitch_deg: vertical tilt (you keep it 0 for now)
    """
    H, W = cyl_img.shape[:2]

    fov = np.deg2rad(fov_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # Effective focal length in pixel units for the output rectilinear image
    fx = (out_w / 2) / math.tan(fov / 2)
    fy = fx

    # Cylindrical "focal" in pixels (maps vertical angle to pixel displacement)
    f_cyl = W / (2 * math.pi)

    # Pixel grid in the output view
    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))

    # Normalized camera rays (rectilinear camera model)
    x = (u - out_w / 2) / fx
    y = (v - out_h / 2) / fy
    z = np.ones_like(x)

    # --- Apply pitch rotation (around X axis) ---
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    y2 = y * cos_p - z * sin_p
    z2 = y * sin_p + z * cos_p
    x2 = x

    # --- Apply yaw rotation (around Y axis) ---
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    x3 = x2 * cos_y + z2 * sin_y
    z3 = -x2 * sin_y + z2 * cos_y
    y3 = y2

    # Map 3D rays to cylindrical coordinates:
    theta = np.arctan2(x3, z3)
    h = y3 / np.sqrt(x3**2 + z3**2)

    # Convert cylindrical coords to pixel coords in the pano
    map_x = ((theta + np.pi) / (2 * np.pi) * W).astype(np.float32)
    map_y = (h * f_cyl + H / 2).astype(np.float32)

    # Sample pixels from pano into perspective image
    persp = cv2.remap(
        cyl_img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    return persp


def process_image(raw_path, road_angle,
                  rot_dir, left_dir, right_dir,
                  do_rotate=True,
                  fov_deg=90, out_w=800, out_h=600,
                  overwrite=False):
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
    left_np = cyl_to_perspective(
        pano_rot, fov_deg=fov_deg, yaw_deg=-90, pitch_deg=0,
        out_w=out_w, out_h=out_h
    )
    left_img = Image.fromarray(cv2.cvtColor(left_np, cv2.COLOR_BGR2RGB))
    if overwrite or (not os.path.exists(left_path)):
        left_img.save(left_path)

    # RIGHT façade (yaw +90)
    right_np = cyl_to_perspective(
        pano_rot, fov_deg=fov_deg, yaw_deg=90, pitch_deg=0,
        out_w=out_w, out_h=out_h
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
    df = pd.read_csv(input_csv)

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
                fov_deg=FOV_DEG,
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
