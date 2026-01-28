# =========================================
# PIPELINE STAGE 0: FOLDER PREPARATION
# =========================================

import os

PROJECT_ROOT = r"C:\Users\viggo\OneDrive - Universiteit Utrecht\Year 2\Thesis\Python_thesis\Facade_area_estimation"
RESULTS_IMAGES = os.path.join(PROJECT_ROOT, "Results_images")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "Results_csv")

# -------------------------------------------------
# Folder presets
# -------------------------------------------------

IMAGE_FOLDERS = [
    "raw_images",          # downloaded raw panoramas
    "rotated_images",      # rotated panoramas
    "left_facades",        # left crops
    "right_facades",       # right crops
    "left_segs",           # left segmentations
    "right_segs",          # right segmentations
    "temp_segs",           # temp tif storage during segmentation
    "temp_center_bands",   # temp center-band masks
]

CSV_FOLDERS = [
    "road_network",        # CSV with sample points + attributes
    "Main_results",        # final results CSV
]

# -------------------------------------------------
# Core helpers
# -------------------------------------------------

def prepare_base_results():
    """
    Creates both base folders and returns them.
    """
    os.makedirs(RESULTS_IMAGES, exist_ok=True)
    os.makedirs(RESULTS_CSV, exist_ok=True)
    return RESULTS_IMAGES, RESULTS_CSV


def prepare_image_folders():
    """
    Creates image folders under Results_images and returns dict of paths.
    """
    base_images, _ = prepare_base_results()
    paths = {}

    for name in IMAGE_FOLDERS:
        path = os.path.join(base_images, name)
        os.makedirs(path, exist_ok=True)
        paths[name] = path

    return paths


def prepare_csv_folders():
    """
    Creates csv folders under Results_csv and returns dict of paths.
    """
    _, base_csv = prepare_base_results()
    paths = {}

    for name in CSV_FOLDERS:
        path = os.path.join(base_csv, name)
        os.makedirs(path, exist_ok=True)
        paths[name] = path

    return paths


def prepare_folders():
    """
    One-call convenience: creates all pipeline folders.
    """
    prepare_image_folders()
    prepare_csv_folders()


if __name__ == "__main__":
    prepare_folders()
    print("Folders ready:")
    print("  Images:", RESULTS_IMAGES)
    print("  CSV:", RESULTS_CSV)
