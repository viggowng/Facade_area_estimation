# ============================================================
# STAGE 3: SAM segmentation
# ============================================================

import os
import numpy as np
import pandas as pd
from PIL import Image
from samgeo.text_sam import LangSAM
from Stage_0 import IMAGE_PATHS, CSV_PATHS

### ==== CONFIGURATION ==== ###
TEMP_DIR = IMAGE_PATHS["temp_segs"]

INPUT_DIR_LEFT = IMAGE_PATHS["left_facades"]
INPUT_DIR_RIGHT = IMAGE_PATHS["right_facades"]

OUTPUT_DIR_LEFT = IMAGE_PATHS["left_segs"]
OUTPUT_DIR_RIGHT = IMAGE_PATHS["right_segs"]

SAVE_OVERLAYS = False # If True, saves RGBA overlay images for visual purposes

STAGE1_CSV = CSV_PATHS["road_network"] / "sample_points.csv"

PROMPTS = {
    "facades": ("facade, building facade", 0.20),
    "windows": ("windows", 0.25),
    "doors":   ("doors", 0.25),
    "roofs": ("slanted roof, roof tiles", 0.40)
}
# Lower thresholds  -> more detections (but more false positives)
# Higher thresholds -> fewer detections (but more false negatives)

# These thresholds were determined via trial and error:
# -> higher roof threshold resulted in no roof detection
# -> lower resulted in too much roof detection (included facade parts).
# ============================================================

# -------------------------------------------

### ==== HELPERS ==== ###
def clear_temp_tifs(folder):
    """Delete old .tif masks so we don't accidentally load the previous image/prompt output."""
    for fn in os.listdir(folder):
        if fn.lower().endswith(".tif"):
            try:
                os.remove(os.path.join(folder, fn))
            except OSError:
                pass

def newest_tif_in(folder):
    """
    LangSAM writes .tif files into TEMP_DIR.
    We assume the most recently modified .tif is the one from the last predict_batch call.
    """
    tifs = [fn for fn in os.listdir(folder) if fn.lower().endswith(".tif")]
    if not tifs:
        return ""
    tifs.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    return os.path.join(folder, tifs[-1])

def mask_to_png(mask, out_path):
    """
    Convert a grayscale mask (0..255) to a strict binary PNG:
        pixels > thresh -> 255 (white / keep)
        pixels <= thresh -> 0 (black / discard)

    This is purely for *output standardization*.
    It ensures later stages always read the same kind of mask.
    """
    out = (mask > 128).astype(np.uint8) * 255
    Image.fromarray(out, mode="L").save(out_path)

def save_overlay(image_path, mask, out_path, rgba=(0, 255, 0, 110)):
    """
    Overlay the mask on the RGB image for visual QC.
    Only used if SAVE_OVERLAYS = True.
    """
    img = Image.open(image_path).convert("RGBA")
    m = (mask > 128)

    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[m] = rgba

    Image.alpha_composite(img, Image.fromarray(overlay)).save(out_path)
# ============================================================

# -------------------------------------------------

### ==== Facade visibility clause ==== ###
def visible_facade_by_distance(row, side) -> bool:
    """
    Uses Stage 1 distances to decide if a façade is visible.
    If distance is 0.0 the code considers there is no visible façade on that side.
    To save computation time, segmentation will be skipped for that facade and return a black mask instead
    (If the distance is 0.0, the estimated facade area will be 0 due to the used formula is stage 4).
    """
    if side == "left":
        dist = row.get("dist_left_m", 0)
    elif side == "right":
        dist = row.get("dist_right_m", 0)
    else:
        raise ValueError("side must be 'left' or 'right'")

    if pd.isna(dist):
        return False

    try:
        return (dist) > 0
    except (TypeError, ValueError):
        return False

def write_black_mask_like(image_path, out_path):
    """
    Writes a completely black binary mask (all zeros) with the same size as the input image.
    This means: "no visible façade" but keeps the pipeline going without the need to handle missing values.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with Image.open(image_path) as im:
        w, h = im.size
    black = np.zeros((h, w), dtype=np.uint8)
    Image.fromarray(black, mode="L").save(out_path)
# ============================================================

# ------------------------------------------------------------

### ==== CORE SEGMENTATION ==== ###
def segment_image(image_path, sam):
    """
    Runs LangSAM with multiple prompts, then combines results.

    Output:
      usable_facade (uint8 0..255):
        - Start with facade mask
        - Remove windows, doors, roofs by zeroing those pixels out

    Why "usable facade"?
      Because your area/height work should NOT count:
        - windows (holes in facade)
        - doors
        - roof pixels (sloped roof misclassified as facade is a known issue)
    """
    masks = {}

    # Prevent mixing prompt outputs between images
    clear_temp_tifs(TEMP_DIR)

    # Run each prompt separately and load the latest .tif output each time
    for label, (prompt, thr) in PROMPTS.items():

        sam.predict_batch(
            images=[image_path],
            out_dir=TEMP_DIR,
            text_prompt=prompt,
            box_threshold=thr,
            text_threshold=thr,
            merge=False
        )

        mask_path = newest_tif_in(TEMP_DIR)
        if not mask_path:
            raise RuntimeError(f"No .tif produced for prompt '{label}' on {os.path.basename(image_path)}")

        masks[label] = np.array(Image.open(mask_path).convert("L"))


    # Combine masks into final "usable facade"
    usable = masks["facades"].copy()

    # Removes unneeded item from masks
    usable[masks["windows"] > 128] = 0
    usable[masks["doors"] > 128] = 0
        # Why 128?
        #   The produced mask is grayscale: 0 (black) -> 255 (white).
        #   The selected pixels in the mask (facades, windows, etc.) are in the white range (> 128).
        #   128 is a safe threshold to make sure all pixels are properly selected.
        #   It could be changed to i.e. usable[masks["windows"] == 255] = 0,
        #   but that might miss some pixels that are slightly less than 255.    

    # Roof safety check (prevents "entire building becoming roof")
    fac_area  = (masks["facades"] > 128).sum()  # total mask pixels
    roof_area = (masks["roofs"] > 128).sum()    # total roof pixels 

    # If roof mask is suspiciously large relative to the facade mask,
    # it is likely a false positive (e.g., dark/brown facade interpreted as roof).
    
    # fraction of tolerated roof area relative to facade area
    MAX_ROOF_FACADE_RATIO = 0.6  # <-- found through iterative process
   
    if fac_area > 0 and (roof_area / fac_area) <= MAX_ROOF_FACADE_RATIO:
        usable[masks["roofs"] > 128] = 0
    else:
        # Skip roof subtraction for this image
        # (optional debug)
        # print(f"Skip roof subtraction: roof/facade={roof_area/fac_area:.2f} for {os.path.basename(image_path)}")
        pass

    return usable


def run_folder(in_dir, out_dir, sam, stage1_df=None, side=None):
    """
    Batch process a directory:
      - for each image, compute usable facade mask
      - save binary PNG
      - optionally save overlay PNG (if set to true in the config)

    Integration:
      If stage1_df + side are provided:
        - looks up dist_left_m / dist_right_m for this image_id (parsed from filename)
        - if distance is 0/NaN -> writes a black mask and prints "no visible façade"
    """
    exts = (".jpg", ".jpeg", ".png")    # can only see images with these extensions
    files = sorted([fn for fn in os.listdir(in_dir) if fn.lower().endswith(exts)])

    if not files:
        print(f"No images found in: {in_dir}")
        return

    # Build quick lookup: image_id -> row-dict (only if Stage 1 provided)
    lookup = None
    if stage1_df is not None:
        
        stage1_df = stage1_df.copy()        # makes a copy to avoid modifying the original
        stage1_df["image_id"] = stage1_df["image_id"].astype(str)           # Forces image_id to be a string (lookup fails if it is no string)
        lookup = stage1_df.set_index("image_id").to_dict(orient="index")    # creates an index to speed up search process

    for i, fn in enumerate(files, start=1):                 # starts loop over images
        image_path = os.path.join(in_dir, fn)
        base = os.path.splitext(fn)[0]                      # e.g. "left_12345"
        mask_out = os.path.join(out_dir, f"{base}.png")     # Saves image in output_dir as.png
        
        # Skips segmentation if output mask already exists
        if os.path.exists(mask_out):
            print(f"[{i}/{len(files)}] SKIP: {fn} -> output exists ({os.path.basename(mask_out)})")
            continue

        # ---------------------------------------------
        # Segmentation under special circumstances
        # ---------------------------------------------
        
        # if rows has values and side is correct-> moves on to segmentation
        if lookup is not None and side in ("left", "right"): 
            # splits filename at 1st "_" and grabs the part after it-> "left_123445" becomes "123445"
            if "_" in base:
                image_id = base.split("_", 1)[1] 
            else:
                image_id = base  # fallback for safety

            row = lookup.get(str(image_id))
            
            # If no Stage 1 row exists, the code treats it as "not segmentable" to be safe.
            if row is None:
                write_black_mask_like(image_path, mask_out)
                print(f"[{i}/{len(files)}] SKIP: {fn} -> no Stage 1 row for image_id={image_id} (wrote black mask)")
                continue

            # If visible_facade_by distance == 0 -> write black mask and skips segmentation 
            if not visible_facade_by_distance(row, side):
                write_black_mask_like(image_path, mask_out)
                print(f"[{i}/{len(files)}] SKIP: {fn} -> no visible façade (dist_{side}_m={row.get(f'dist_{side}_m')}) (wrote black mask)")
                continue

        # ---------------------------------------------
        # Normal segmentation
        # ---------------------------------------------
        try:
            usable_mask = segment_image(image_path, sam)

            mask_to_png(usable_mask, mask_out)

            if SAVE_OVERLAYS:
                overlay_out = os.path.join(out_dir, f"{base}_overlay.png")
                save_overlay(image_path, usable_mask, overlay_out)

            print(f"[{i}/{len(files)}] OK: {fn}")

        except Exception as e:
            print(f"[{i}/{len(files)}] FAIL: {fn} -> {e}")
# =========================================================

# ---------------------------------------------------------

### ==== MAIN ==== ###
if __name__ == "__main__":
    # Read Stage 1 once, then use it to decide if a façade is visible per side
    stage1_df = pd.read_csv(STAGE1_CSV, dtype={"image_id": "string"})

    sam = LangSAM()

    run_folder(INPUT_DIR_LEFT, OUTPUT_DIR_LEFT, sam, stage1_df=stage1_df, side="left")
    print("Left folder done, starting right folder...")
    run_folder(INPUT_DIR_RIGHT, OUTPUT_DIR_RIGHT, sam, stage1_df=stage1_df, side="right")
    print("Segmentation completed, move on to stage 4: Façade height and area calculation")
