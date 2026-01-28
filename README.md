STAGE 0: Setup

Creates the required folder structure that will host imagery, temporary and CSV outputs
- Images
    "raw_images",          # downloaded raw panoramas
    "rotated_images",      # rotated panoramas
    "left_facades",        # left crops
    "right_facades",       # right crops
    "left_segs",           # left segmentations
    "right_segs",          # right segmentations
    "temp_segs",           # temp tif storage during segmentation
    "temp_center_bands",   # temp center-band masks

- CSV files
    "road_network",        # CSV with sample points + attributes
    "Main_results",        # final results CSV containing height and area


STAGE 1: ROAD NETWORK + SVI location (including distance, road angle and closest mapillary image id)

Input:
- BAG data (obtainable from PDOK)
- Boundaries of study area (Neighbourhood scale in this study)
- Road network map (OpenStreetMap)

Steps:
- Road network clipped on input boundaries
- Creates streetview sample locations along roadnetwork, 30 meters spacing
- Searches closest mapillary panoramic image_id via KDtree
- Computes distance to nearest building on the left and right side of the street

Output:
- A .csv file containing these columns:
    - xcoord, ycoord (EPSG:4326)
    - road_angle (bearing of road at sample point)
    - nearest Mapillary pano image_id (via vector tiles)
    - distance to that pano (meters)
    - dist_left_m / dist_right_m (meters), using a "wide strip" perpendicular search
      and a max cutoff (e.g., >15m => 0).

Notes:
- Distances above 15 meters are set to 0


Stage 2: Download and process panos

Input:
- Stage 1 output .csv file

Steps:
- reads stage 1 output CSV
    - for each row:
          * read image_id + road_angle
          * download pano
          * rotate + produce left/right crops
      - Images are now ready for segmentation

Output:
- Left and right facade cropped images perpendicular from the road

Note:
- The reporjection from cylindrical to rectilingilar facade crops still leaves little distortion to the side. 
- This is difficult to correct without zooming in more on the image (zooming may result in parts of the facade leaving - the image frame -> area cannot be calulated).


Stage 3: SAM Segmentation

Input:
   - Perspective facade crops produced in Stage2 (left/right images)

Steps:  
- Run LangSAM text-prompt segmentation
- 

Output:
- A binary PNG mask for each input image:
    255 = facade pixel (white)
    0 = background pixel (black)

 Notes:
- LangSAM writes intermediate outputs as GeoTIFF (.tif). To avoid confusion between .tif outputs,the code deletes every .tif after conversion to mask.
- The usable mask = facade - (roof + windows + doors)
- SAM often confuses roof and facade with each other, sometimes including the building facade as part of the roof.
When doing the usable mask generation, it may remove parts of the facade from that mask. A "Max_roof_area" clause has been added to tackle this issue (in the code is a more in-depth explanation).


STAGE 4: FAÇADE HEIGHT AREA ESTIMATION

Input:
- Stage 3 output masks
- Stage 1 output .csv

Steps:
- Calculating facade pixel height
- Conversion to real height following pinhole camera model and basic geometry
- Calculating pixel count
- Conversion to pixel area and facade area

Output:
- A new .csv file containing all the computed attributes (height, distance and area)

Notes:
- Height calculations follow a geometric camera model including focal length calculations
These are more in-depth explained using notes within the code