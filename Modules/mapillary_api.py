# ---Requirements---
import os
import math
import csv
import requests
import pandas as pd
import numpy as np
from PIL import Image
from samgeo.text_sam import LangSAM

from process_data import process_image


# ============================
# CONFIG
# ============================

# --- Mapillary API token ---
ACCESS_TOKEN = "MAPILLARY_ACCESS_TOKEN_HERE"



# Base results folder and subfolders

# ---Foldername---
BASE_RESULTS = r"path_to_results_folder"

# ---Subfolders---
RAW_DIR = os.path.join(BASE_RESULTS, "raw_pano_images")
ROT_DIR = os.path.join(BASE_RESULTS, "rotated_pano")
LEFT_DIR = os.path.join(BASE_RESULTS, "facade_crops_left")
RIGHT_DIR = os.path.join(BASE_RESULTS, "facade_crops_right")
SEG1_DIR = os.path.join(BASE_RESULTS, "temp_seg_facades")   # temporary mask storage
SEG2_DIR = os.path.join(BASE_RESULTS, "seg_facades") 
CSV_DIR = os.path.join(BASE_RESULTS, "CSV")                 # outputs CSV containing results

# Input samples and output CSV paths
INPUT_CSV = r"path_to_input_csv_file"
OUT_CSV = os.path.join(CSV_DIR, "Height_Area_results.csv")


# Required parameters need later on
# Height model parameters
VFOV_DEG = 90.0   # VFOV of perspective crop
IMG_H_PX = 600    # pixel height of perspective crop
BAND_FRAC = 0.6   # central band for extents

# Mapillary search
MAX_TRIES = 6
START_DELTA = 0.00015  # ~16m in lat, will expand if needed


# ============================
# FOLDER PREP
# ============================

#Creates results subfolders if they don't exist
for d in [RAW_DIR, ROT_DIR, LEFT_DIR, RIGHT_DIR, SEG1_DIR, SEG2_DIR, CSV_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================
# GEO + MAPILLARY HELPERS
# ============================

# Formula for computing the haversine distance in meters between input point and pano image point
def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

# Formula for retrieving closest pano image_id from Mapillary API
def find_closest_pano_image_id(lon, lat, access_token, max_tries=MAX_TRIES, start_delta=START_DELTA):
    """
    BBOX expansion search that only returns panoramas (is_pano=true),
    then selects the closest by haversine distance.
    """
    endpoint = "https://graph.mapillary.com/images"
    delta = float(start_delta)
    last_error = None

    for _ in range(max_tries):
        min_lon = lon - delta
        min_lat = lat - delta
        max_lon = lon + delta
        max_lat = lat + delta

        params = {
            "fields": "id,computed_geometry,is_pano",
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "is_pano": "true",
            "limit": 2000,
            "access_token": access_token,
        }

        try:
            resp = requests.get(endpoint, params=params, timeout=30)
            data = resp.json()
        except Exception as e:
            last_error = e
            delta *= 2
            continue

        images = data.get("data", [])
        if not images:
            delta *= 2
            continue

        best_id = None
        best_dist = float("inf")

        for img in images:
            if not img.get("is_pano", False):
                continue

            geom = img.get("computed_geometry")
            if not geom or "coordinates" not in geom:
                continue

            img_lon, img_lat = geom["coordinates"][0], geom["coordinates"][1]
            d = haversine_m(lon, lat, img_lon, img_lat)

            if d < best_dist:
                best_dist = d
                best_id = img.get("id")

        if best_id:
            return best_id, best_dist

        delta *= 2

    raise RuntimeError(f"No pano images found near lon/lat after {max_tries} expansions. Last error: {last_error}")

# Function to get the thumbnail URL for a given image_id
def get_thumb_url(image_id, access_token):
    meta = requests.get(
        f"https://graph.mapillary.com/{image_id}",
        params={"fields": "thumb_2048_url,is_pano", "access_token": access_token},
        timeout=30
    ).json()

    if not meta.get("is_pano", False):
        raise RuntimeError(f"Selected image {image_id} is not panoramic (is_pano=false).")

    if "thumb_2048_url" not in meta:
        raise RuntimeError(f"No thumb_2048_url returned for image {image_id}. Response: {meta}")

    return meta["thumb_2048_url"]


# ============================
# SEGMENTATION + METRICS HELPERS
# ============================

# center band mask is required to account for distortions on the sides of crops
def keep_center_band(mask: np.ndarray, frac: float = BAND_FRAC) -> np.ndarray:
    h, w = mask.shape
    x0 = int(w * (0.5 - frac / 2))
    x1 = int(w * (0.5 + frac / 2))
    out = mask.copy()
    out[:, :x0] = 0
    out[:, x1:] = 0
    return out

# Count pixels above threshold in mask, required for area calculation
def count_mask_pixels(mask: np.ndarray, thresh: int = 128) -> int:
    return int((mask > thresh).sum())

# Detect vertical extents (top/bottom parts) from center-band mask required for VFOV calculation
def detect_vertical_extents(mask: np.ndarray,
                                   thresh: int = 128,           # RGB threshold for façade pixel
                                   min_row_frac: float = 0.06): # Threshold of minimal 6% pixels per row
    """
    Determine top/bottom only from rows that have enough façade pixels.
    Thin clutter like branches/poles will be ignored.
    """
    h, w = mask.shape
    binm = (mask > thresh)

    row_counts = binm.sum(axis=1)
    min_pixels = min_row_frac * w

    good_rows = np.where(row_counts >= min_pixels)[0]
    if good_rows.size == 0:
        raise ValueError("No façade rows found after row-coverage filtering")

    top_row = int(good_rows.min())
    bottom_row = int(good_rows.max())

    midrow = h // 2
    top_part = midrow - top_row
    bottom_part = bottom_row - midrow
    return top_part, bottom_part


# Height computation for top / bottom part
def height_from_parts(top_part_px: float, bottom_part_px: float, dist_m: float,
                      vfov_deg: float = VFOV_DEG, img_h_px: int = IMG_H_PX) -> float:
    """
    mathematical formula:
      fy = (H/2) / tan(vfov/2)
      height = dist * ((top_part/fy) + (bottom_part/fy))
    """
    vfov = math.radians(float(vfov_deg))
    fy = (img_h_px / 2.0) / math.tan(vfov / 2.0)
    return float(dist_m) * ((float(top_part_px) / fy) + (float(bottom_part_px) / fy))

# Prevent confusion by always getting the newest tif in a folder
def newest_tif_in(folder: str) -> str:
    tif_files = [fn for fn in os.listdir(folder) if fn.lower().endswith(".tif")]
    if not tif_files:
        return ""
    tif_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return os.path.join(folder, tif_files[-1])

def clear_temp_tifs(folder: str) -> None:
    """Prevents mixing masks between prompts/images."""
    for fn in os.listdir(folder):
        if fn.lower().endswith(".tif"):
            try:
                os.remove(os.path.join(folder, fn))
            except OSError:
                pass

def save_overlay(input_path: str, mask: np.ndarray, out_path: str, rgba=(0, 255, 0, 110)):
    img = Image.open(input_path).convert("RGBA")
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask > 128] = rgba
    Image.alpha_composite(img, Image.fromarray(overlay)).save(out_path)

# Main segmentation + measurement function
def segment_and_measure(image_path: str, sam: LangSAM, dist_m: float):
    """
    Runs 4 prompts per image -> builds usable facade mask -> counts pixels -> detects
    center-band extents -> saves the mask overlay, and returns metrics.
    """
    prompts = {
        "facades": ("facade, building facade", 0.20), 
        "windows": ("windows", 0.25), 
        "doors": ("doors", 0.25), 
        "roofs": ("roof", 0.30),
    }

    masks = {}

    clear_temp_tifs(SEG1_DIR) # clears previous masks to avoid confusion

    for label, (prompt, thr) in prompts.items():       #Loops over prompts for every image
        sam.predict_batch(
            images=[image_path],                #Fetches image to SAM
            out_dir=SEG1_DIR,                   #Output directory for masks
            text_prompt=prompt,
            box_threshold=thr,                 #Confidence thresholds for proposing coherent regions
            text_threshold=thr,                #Confidence thresholds for matching the region to the prompt
            merge=False                         #No merging of masks, we want separate ones
        )
        # Fetches the newest mask produced in the temporary folder
        mask_path = newest_tif_in(SEG1_DIR)
        if not mask_path:
            raise RuntimeError(f"No .tif mask produced for {label} on {os.path.basename(image_path)}")

        masks[label] = np.array(Image.open(mask_path).convert("L"))

    # --- Combine masks ---
    usable_facade = masks["facades"].copy()
    usable_facade[masks["windows"] > 128] = 0   #Removes windows from facade mask, >128 is RGB value assigned to detected pixels
    usable_facade[masks["doors"] > 128] = 0     #Removes doors from facade mask
    usable_facade[masks["roofs"] > 128] = 0     #Removes roofs from facade mask

    # --- Count pixels on FULL usable mask -> area calculation ---
    total_facade_pixels = count_mask_pixels(usable_facade)

    # --- Computes vertical pixel extents of center band -> VFOV calculation ---
    usable_facade_center = keep_center_band(usable_facade, frac=BAND_FRAC)

    # --- Overlays mask on image and saves it in the seg_facades folder---
    base = os.path.splitext(os.path.basename(image_path))[0]
    save_overlay(image_path, usable_facade,
                 os.path.join(SEG2_DIR, f"{base}_FULL.png"))
    save_overlay(image_path, usable_facade_center,
                 os.path.join(SEG2_DIR, f"{base}_BAND.png"))

    # --- HEIGHT CALCULATION ---
    top_part, bottom_part = detect_vertical_extents(usable_facade_center)

    height_p = int(top_part + bottom_part)                      #Total height in px
    height_m = height_from_parts(top_part, bottom_part, dist_m) #Total height in meters

    # --- AREA CALCULATION ---
    pixel_size_m = height_m / height_p                          #Size of one pixel in meters
    pixel_area_m2 = pixel_size_m ** 2                           #Area of one pixel in m^2
    facade_area_m2 = total_facade_pixels * pixel_area_m2        #Total usable facade area in m^2

    return top_part, bottom_part, height_m, height_p, total_facade_pixels, facade_area_m2


# ============================
# Pipeline configuration
# ============================

# Load input samples CSV
samples = pd.read_csv(INPUT_CSV)

# Basic column validation
required_cols = {"FID", "xcoord", "ycoord", "dist", "road_angle"}
missing = required_cols - set(samples.columns)
if missing:
    raise ValueError(f"Missing required columns in input CSV: {missing}")

sam = LangSAM()

# ============================
# MAIN PIPELINE
# ============================

# Create output CSV and write header
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out)
    writer.writerow([
        "FID", "image_id", "side",
        "top_part", "bottom_part",
        "height_m", "height_p",
        "total_facade_pixels",
        "facade_area_m2"
    ])

    # Process each sample point
    for idx, row in samples.iterrows():
        fid = row["FID"]

        lon = float(row["xcoord"])
        lat = float(row["ycoord"])

        dist_m = float(row["dist"])
        road_angle = float(row["road_angle"])

        print(f"\n[{idx+1}/{len(samples)}] FID={fid} lon={lon} lat={lat} dist={dist_m} road_angle={road_angle}")

        # 1) Find pano image_id
        try:
            image_id, dist_to_point = find_closest_pano_image_id(lon, lat, ACCESS_TOKEN)
            print(f"   Mapillary pano: {image_id} (≈ {dist_to_point:.1f} m from point)")
        except Exception as e:
            print(f"   Could not find pano for FID={fid}: {e}")
            writer.writerow([fid, "", "left", "", "", "", "", "", ""])
            writer.writerow([fid, "", "right", "", "", "", "", "", ""])
            continue

        # 2) Download pano
        raw_path = os.path.join(RAW_DIR, f"{image_id}.jpg")
        try:
            if not os.path.exists(raw_path):
                url = get_thumb_url(image_id, ACCESS_TOKEN)
                img_bytes = requests.get(url, timeout=60).content
                with open(raw_path, "wb") as f_img:
                    f_img.write(img_bytes)
                print(f"   Downloaded pano: {raw_path}")
            else:
                print(f"   Pano exists: {raw_path}")
        except Exception as e:
            print(f"   Download failed for image_id={image_id}: {e}")
            writer.writerow([fid, image_id, "left", "", "", "", "", "", ""])
            writer.writerow([fid, image_id, "right", "", "", "", "", "", ""])
            continue

        # 3) Create left/right crops
        try:
            result = process_image(raw_path, road_angle)
            if result is None:
                raise RuntimeError("process_image returned None (should return left_path, right_path, rotated_path).")
            left_path, right_path, rotated_path = result
        except Exception as e:
            print(f"   process_image failed for image_id={image_id}: {e}")
            writer.writerow([fid, image_id, "left", "", "", "", "", "", ""])
            writer.writerow([fid, image_id, "right", "", "", "", "", "", ""])
            continue

        # 4) Segment + measure each side
        for side, img_path in [("left", left_path), ("right", right_path)]:
            try:
                top_part, bottom_part, height_m, height_p, total_pixels, facade_area_m2 = segment_and_measure(
                    img_path, sam, dist_m
                )

                writer.writerow([
                    fid, image_id, side,
                    int(top_part), int(bottom_part),
                    f"{height_m:.3f}", int(height_p),
                    int(total_pixels),
                    f"{facade_area_m2:.3f}"
                ])

                print(
                    f"   ✓ {side}: top={top_part} bottom={bottom_part} "
                    f"height_m={height_m:.2f} px={height_p} total_px={total_pixels} area_m2={facade_area_m2:.2f}"
                )

            except Exception as e:
                print(f"   Segmentation/metrics failed ({side}) for image_id={image_id}: {e}")
                writer.writerow([fid, image_id, side, "", "", "", "", "", ""])




print(f"\nDone. Output CSV written to:\n{OUT_CSV}")
