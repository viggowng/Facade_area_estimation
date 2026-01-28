# =========================================
# PIPELINE STAGE 0: CONFIG + FOLDER SETUP
# =========================================

from pathlib import Path
import os
from dotenv import load_dotenv

# Load local secrets from .env (not committed)
load_dotenv()

# Project root (auto, no personal path)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Token (private, comes from .env)
MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN")
if not MAPILLARY_TOKEN:
    raise ValueError("MAPILLARY_TOKEN not set. Create a .env file in project root.")

# Base output dirs
RESULTS_IMAGES = PROJECT_ROOT / "Results_images"
RESULTS_CSV = PROJECT_ROOT / "Results_csv"

# Subfolders
IMAGE_FOLDERS = [
    "raw_images",
    "rotated_images",
    "left_facades",
    "right_facades",
    "left_segs",
    "right_segs",
    "temp_segs",
    "temp_center_bands",
]

CSV_FOLDERS = [
    "road_network",
    "Main_results",
]

# Computed paths (exportable)
IMAGE_PATHS = {name: RESULTS_IMAGES / name for name in IMAGE_FOLDERS}
CSV_PATHS = {name: RESULTS_CSV / name for name in CSV_FOLDERS}

ALL_FOLDERS = [RESULTS_IMAGES, RESULTS_CSV] + list(IMAGE_PATHS.values()) + list(CSV_PATHS.values())


def prepare_folders() -> None:
    """Create all required output folders."""
    for folder in ALL_FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    prepare_folders()
    print("Folders ready:")
    print(" Project root:", PROJECT_ROOT)
    print(" Images base:", RESULTS_IMAGES)
    print(" CSV base:", RESULTS_CSV)
