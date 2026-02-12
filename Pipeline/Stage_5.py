# ==========================================================
# Stage 5: Finishing touches and export ready CSV / GPKG
# ==========================================================

import pandas as pd
from Stage_0 import CSV_PATHS

STAGE1_CSV = CSV_PATHS["road_network"] / "sample_points.csv"
STAGE4_CSV = CSV_PATHS["Main_results"] / "Facade_height_area_estimations.csv"
STAGE5_CSV = CSV_PATHS["Main_results"] / "Thesis_results.csv"

Stage1 = pd.read_csv(STAGE1_CSV)
Stage4= pd.read_csv(STAGE4_CSV)

# (optional but smart) clean + align key types
Stage1["FID"] = Stage1["FID"].astype(str)
Stage4["FID"] = Stage4["FID"].astype(str)

# keep only what you need from sample points layers
sp_key = Stage1[["FID", "xcoord", "ycoord", "dist_left_m", "dist_right_m"]]

# one-to-many join: every facade row gets x/y
joined = Stage4.merge(sp_key, on="FID", how="left", validate="many_to_one")
joined.drop(
    columns=['dist_m','top_part_px','bottom_part_px','height_px','status','error'],
    inplace=True
)

# sanity check
missing_xy = joined["xcoord"].isna().sum()
print("rows with missing x/y:", missing_xy)

joined.to_csv(STAGE5_CSV, index=False)
print("wrote:", STAGE5_CSV)
