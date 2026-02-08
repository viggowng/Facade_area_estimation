# ============================================================
# STAGE 4: FAÇADE HEIGHT + AREA ESTIMATION
# ============================================================
import os
import math
import numpy as np
import pandas as pd
from PIL import Image
from Stage_0 import CSV_PATHS, IMAGE_PATHS

# ============================================================
# CONFIGURATION
# ============================================================

MIN_ROW_FRAC = 0.06     # minimum fraction of row pixels to consider row as facade (default 6%)

# IMPORTANT:
# VFOV_DEG must match the VFOV used in Stage 2 when generating the perspective crops.
VFOV_DEG = 90           # vertical field of view of perspective crops (degrees)

# If True: write rows even when missing masks / missing distance, but fill zeros + error message
# If False: skip those rows entirely
WRITE_FAIL_ROWS = True

# Input from Stage 1
STAGE1_CSV = CSV_PATHS["road_network"] / "sample_points.csv"

# Output of Stage 4
STAGE4_CSV = CSV_PATHS["Main_results"] / "Facade_height_area_estimations.csv"

# Segmentation inputs
LEFT_SEGS_DIR = IMAGE_PATHS["left_segs"]
RIGHT_SEGS_DIR = IMAGE_PATHS["right_segs"]

# ============================================================
# MASK IO HELPERS
# ============================================================

def load_mask(mask_path):
    """
    Loads a mask image (e.g. *_mask.png) as a uint8 grayscale numpy array.
    Expected values: 0..255 (8 bit grayscale)
    """
    return np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)


def save_mask(mask, out_path):
    """
    Saves a uint8 grayscale mask (0..255) to PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(out_path)


def save_overlay(image_path, mask, out_path, rgba=(0, 255, 0, 110), thresh=128):
    """
    Saves an RGBA overlay for quick QC.
    mask pixels > thresh become a translucent color overlay.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = Image.open(image_path).convert("RGBA")
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[mask > thresh] = rgba

    Image.alpha_composite(img, Image.fromarray(overlay)).save(out_path)

# ============================================================
# CORE PROCESSING
# ============================================================
def count_mask_pixels(mask, thresh=128):
    """
    Counts number of pixels classified as facade (mask pixels with value > 128 are seen as facade).
    """
    return int((mask > thresh).sum())


def detect_vertical_extents(mask, thresh=128, min_row_frac=MIN_ROW_FRAC):
    """
    Determine top/bottom extents of facade from a mask.
    Steps:
      - Scans pixel rows downwards to find the first and last rows with sufficient facade pixels
      (top row is 1, bottom row is 600)
      - Locates the middle of the image
      - Computes offsets from midline to top/bottom rows

    Returns:
      top_part_px, bottom_part_px
    """
    h, w = mask.shape
    binm = (mask > thresh)                                  # Just to make sure the loaded mask is binary, though it should be already.

    row_counts = binm.sum(axis=1)                           # count facade pixels per row
    min_pixels = min_row_frac * w                           # minimum amount of pixels to be considered as facade (6% of the row).
    # A minimum row coverage filter is applied (default 6% of image width)
    # This prevents loose pixels at the top or bottom of image from affecting the extents

    good_rows = np.where(row_counts >= min_pixels)[0]       # good rows are the pixel rows with enough facade pixels to be considered facade
    if good_rows.size == 0:
        raise ValueError("No facade rows found after row-coverage filtering")

    top_row = good_rows.min()                                # first good row (from top)
    bottom_row = good_rows.max()                             # last good row (from top)

    midrow = h // 2                                          # middle row of the image (assumed horizon line)
    top_part = midrow - top_row                              # offset from midrow to top_row
    bottom_part = bottom_row - midrow                        # offset from midrow to bottom_row

    return int(top_part), int(bottom_part)


def height_from_parts_pinhole(top_part_px, bottom_part_px, dist_m, vfov_deg, img_h_px):
    """
    Correct rectilinear (pinhole) mapping:
    - Convert pixel offsets from image center to elevation angles via atan(offset / fy)
    - Then use: height = d * (tan(alpha_top) + tan(alpha_bottom))

    Assumptions:
    - Camera pitch = 0 (horizon is at midrow)
    - VFOV is known and matches the Stage 2 perspective crop VFOV
    - dist_m is the perpendicular distance to the façade plane (or a good proxy)
    """
    half_h = img_h_px / 2.0
    vfov = math.radians(vfov_deg)

    # focal length in pixels (vertical)
    fy = half_h / math.tan(vfov / 2.0)

    # pixel offsets from center (positive up for top_part, positive down for bottom_part)
    alpha_top = math.atan(top_part_px / fy)
    alpha_bottom = math.atan(bottom_part_px / fy)

    height_m = dist_m * (math.tan(alpha_top) + math.tan(alpha_bottom))
    return height_m

# ============================================================
# METRICS EXTRACTION
# ============================================================

def metrics_from_mask(mask: np.ndarray, dist_m,
                      vfov_deg=VFOV_DEG,
                      thresh=128,
                      min_row_frac=MIN_ROW_FRAC):
    """
    One-call metric extraction:
      - total facade pixels (full mask)
      - vertical extents
      - height_px and height_m

    IMPORTANT CHANGE:
      - Uses the REAL mask height (mask.shape[0]) instead of a fixed IMG_H_PX constant.

    Returns:
      top_part_px, bottom_part_px, height_px, height_m, total_facade_pixels
    """
    # Grab REAL image dimensions from the mask itself
    img_h_px, img_w_px = mask.shape

    total_facade_pixels = count_mask_pixels(mask, thresh=thresh)

    top_part_px, bottom_part_px = detect_vertical_extents(mask, thresh=thresh, min_row_frac=min_row_frac)

    height_px = top_part_px + bottom_part_px
    height_m = height_from_parts_pinhole(
        top_part_px=top_part_px,
        bottom_part_px=bottom_part_px,
        dist_m=dist_m,
        vfov_deg=vfov_deg,
        img_h_px=img_h_px
    )

    return top_part_px, bottom_part_px, height_px, height_m, total_facade_pixels


def facade_area_from_pixels(total_facade_pixels, height_m, height_px):
    """
    Convert pixel count to m² using pixel size derived from height calibration.

    Assumption:
      - pixel height in meters = height_m / height_px
      - pixel area ~ (pixel_height_m)^2  (square-ish pixels in rectilinear crop)

    Returns facade_area_m2
    """
    if height_px <= 0:
        return 0.0

    pixel_size_m = height_m / height_px
    pixel_area_m2 = pixel_size_m ** 2
    return float(total_facade_pixels) * pixel_area_m2

# ============================================================
# “DATABASE” BUILD: one row per (image_id, side)
# ============================================================

def build_result_table(stage1_csv) -> pd.DataFrame:
    """
    Reads Stage 1 CSV and outputs a long-format table:
      one record per side per sample point

    Columns you get:
      - FID (from Stage 1)
      - image_id
      - side: "left" or "right"
      - dist_m (dist_left_m or dist_right_m)
      - height_m, height_px, top_part_px, bottom_part_px, facade_pixels, facade_area_m2
      - status / error text
    """

    # Force image_id to be treated as text to avoid pandas numeric inference
    df = pd.read_csv(stage1_csv, dtype={"image_id": "string"})

    required = {"FID", "image_id", "dist_left_m", "dist_right_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stage 1 CSV missing columns: {missing}")

    rows = []

    for r in df.itertuples(index=False):
        image_id = r.image_id
        fid = r.FID

        # Create two “candidate” records: left + right
        for side in ("left", "right"):
            if side == "left":
                dist_m = getattr(r, "dist_left_m")
                mask_path = os.path.join(LEFT_SEGS_DIR, f"left_{image_id}.png")
            else:
                dist_m = getattr(r, "dist_right_m")
                mask_path = os.path.join(RIGHT_SEGS_DIR, f"right_{image_id}.png")

            rec = {
                "FID": fid,
                "image_id": image_id,
                "side": side,
                "dist_m": dist_m,
                # metrics placeholders
                "top_part_px": 0,
                "bottom_part_px": 0,
                "height_px": 0,
                "height_m": 0.0,
                "facade_pixels": 0,
                "facade_area_m2": 0.0,
                "status": "ok",
                "error": "",
            }

            # 1) Skip if no distance (Stage 1 sets “invalid/too far” to 0.0)
            if dist_m is None or dist_m <= 0:
                rec["status"] = "no visible façade"
                if not WRITE_FAIL_ROWS:
                    continue
                rows.append(rec)
                continue

            # 2) Skip if image_id missing
            if image_id is None or str(image_id).strip() == "":
                rec["status"] = "skip_no_image_id"
                if not WRITE_FAIL_ROWS:
                    continue
                rows.append(rec)
                continue

            # 3) Compute metrics from mask
            try:
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask not found: {mask_path}")

                mask = load_mask(mask_path)

                top_part, bottom_part, height_px, height_m, facade_pixels = metrics_from_mask(
                    mask=mask,
                    dist_m=float(dist_m),
                    vfov_deg=VFOV_DEG,
                    thresh=128,
                    min_row_frac=MIN_ROW_FRAC
                )

                area_m2 = facade_area_from_pixels(facade_pixels, height_m, height_px)

                rec.update({
                    "top_part_px": top_part,
                    "bottom_part_px": bottom_part,
                    "height_px": height_px,
                    "height_m": float(height_m),
                    "facade_pixels": int(facade_pixels),
                    "facade_area_m2": float(area_m2),
                    "status": "ok",
                    "error": "",
                })

            except Exception as e:
                rec["status"] = "fail"
                rec["error"] = str(e)
                if not WRITE_FAIL_ROWS:
                    continue

            rows.append(rec)

    return pd.DataFrame(rows)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    out = build_result_table(STAGE1_CSV)
    out.to_csv(STAGE4_CSV, index=False)
    print("Wrote Stage 4 table:", STAGE4_CSV)
    print("Rows:", len(out))
    print(out["status"].value_counts(dropna=False))
