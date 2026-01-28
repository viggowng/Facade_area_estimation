# ============================================================
# PIPELINE STAGE 1: Prepping road network + SVI locations
# ============================================================

from vt2geojson.tools import vt_bytes_to_geojson
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely.ops import unary_union, linemerge, nearest_points

from scipy.spatial import cKDTree

import geopandas as gpd
import osmnx as ox
import mercantile

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import requests
import math
import os



### ==== CONFIGURATION ==== ###
ACCESS_TOKEN = "MLY|25078958115060460|63887214bce846039a87cac96f71e44a"

NEIGH_FILE = r"C:\Users\viggo\OneDrive - Universiteit Utrecht\Year 2\Thesis\Python_thesis\Facade_area_estimation\Input_data\Frankendael_geom.gpkg"
NEIGH_LAYER = None

BUILDINGS_FILE = r"C:\Users\viggo\OneDrive - Universiteit Utrecht\Year 2\Thesis\Python_thesis\Facade_area_estimation\Input_data\Buildings.gpkg"
BUILDINGS_LAYER = None

SAMPLES_OUT = r"C:\Users\viggo\OneDrive - Universiteit Utrecht\Year 2\Thesis\Python_thesis\Facade_area_estimation\Results_csv\road_network\Frankendael_sample_points.csv"
os.makedirs(os.path.dirname(SAMPLES_OUT), exist_ok=True)

# CRS
CRS_WGS84 = "EPSG:4326"    # WSG84 (mapillary crs)
CRS_METRIC = "EPSG:28992"  # RD New crs

# Road sampling
POINT_SPACING_M = 30            # Distance between sample points along roads
BEARING_EPS_M = 5               # how far to look forward/back along a road line to derive road angle

# Mapillary tile lookup
TILE_ZOOM = 14
MAX_PANO_DIST_M = 50            # max tolerated distance between sample coordinates and nearest Mapillary pano
TILESET = "mly1_public"
TILE_WORKERS = 10

# Building distance search parameters
STRIP_LENGTH_M = 80             # max distance to search for facades perpendicular to road
STRIP_HALF_WIDTH_M = 4          # half-width of the perpendicular search strip -> prevents the ray from missing slightly offset buildings
BUILDING_SEARCH_DIST_M = 20     # candidate building search radius (bbox prefilter)
MAX_FACADE_DIST_M = 15.0        # cutoff: if nearest facade further than this -> set to 0.0
# ============================================================

#-------------------------------------------------------------

### ==== Geometry helpers ==== ###
def iter_lines(geom):
    """
    Yield LineString parts for either LineString or MultiLineString.
    This lets downstream code treat everything as an iterator of LineStrings.
    """
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            yield g


def bearing_from_two_points(p0, p1):
    """
    Bearing in degrees clockwise from North, assuming coordinates are metric (x=east, y=north).
    Returns angle between 0 - 360 for every road segment.
    """
    ang = math.degrees(math.atan2(p1.x - p0.x, p1.y - p0.y))    #Mathematical formula for deriving bearing
    return ang + 360 if ang < 0 else ang


def load_boundaries(path, layer=None):
    """
    Loads a neighbourhood polygon and returns it in EPSG:4326 (required by OSMnx).
    """
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)    #gdf = GeoDataFrama which is used by Geopandas to store geospatial data
    if len(gdf) == 0:                   # checks if there are no features in the gdf
        raise ValueError("Neighbourhood polygon file has no features.")
    if gdf.crs is None:
        raise ValueError("Neighbourhood polygon has no CRS.")

    poly = gdf.geometry.iloc[0]
    poly_4326 = gpd.GeoSeries([poly], crs=gdf.crs).to_crs(CRS_WGS84).iloc[0]

    # fix minor validity issues
    if not poly_4326.is_valid:
        poly_4326 = poly_4326.buffer(0)

    return poly_4326
# ============================================================

# ------------------------------------------------------------

### ==== Road network creation -> sample points + attributes ==== ###
def get_roads_in_neighbourhood(poly_4326):
    """
    Downloads OSM roads in the neighbourhood polygon (WGS84), then projects to EPSG:28992.
    Merges all road edges into one multi/linestring geometry (for sampling).
    """
    # Restrict to residential roads (other roads are less relevant for facade segmentation)
    cf = '["highway"="residential"]'

    # OSMnx graph from polygon (must be EPSG:4326)
    G = ox.graph_from_polygon(poly_4326, simplify=True, truncate_by_edge=True, custom_filter=cf)

    # Project the graph to metric CRS for distance/spacing operations
    G_proj = ox.project_graph(G, to_crs=CRS_METRIC)
    _, edges = ox.graph_to_gdfs(G_proj)

    # Clip to neighbourhood (also in metric CRS)
    poly_28992 = gpd.GeoSeries([poly_4326], crs=CRS_WGS84).to_crs(CRS_METRIC).iloc[0]
    edges = edges.clip(poly_28992)

    # Keep only geometries
    edges = edges[["geometry"]].copy()
    edges = edges.reset_index(drop=True)
    edges.set_crs(CRS_METRIC, inplace=True)

    # Merge to one geometry -> required for evenly spread samples
    merged = linemerge(unary_union(edges.geometry))
    return gpd.GeoDataFrame(geometry=[merged], crs=CRS_METRIC)


def sample_points_with_local_bearing(roads_28992, spacing_m=30, eps_m=5):
    """
    Samples points every spacing_m meters along the road network,
    and computes local bearing using a small +/- eps_m window.
    Output is in EPSG:4326 (for Mapillary lookup/export).

    Why eps_m:
    - Use two nearby points along the road line and compute bearing between them.
    - This is more stable than using raw segment direction in a merged geometry.
    """
    merged_geom = roads_28992.geometry.iloc[0]

    rows = []
    for line in iter_lines(merged_geom):
        if line.is_empty or line.length < spacing_m:
            continue

        d = 0.0
        while d < line.length:
            pt = line.interpolate(d)

            d0 = max(d - eps_m, 0.0)
            d1 = min(d + eps_m, line.length)

            if d1 == d0:
                bearing = None
            else:
                p0 = line.interpolate(d0)
                p1 = line.interpolate(d1)
                bearing = bearing_from_two_points(p0, p1)

            rows.append((pt, bearing))
            d += spacing_m

    gdf = gpd.GeoDataFrame(rows, columns=["geometry", "road_angle"], geometry="geometry", crs=CRS_METRIC)

    # Export in WGS84 for Mapillary and downstream CSV convenience
    gdf = gdf.to_crs(CRS_WGS84)
    gdf["xcoord"] = gdf.geometry.x
    gdf["ycoord"] = gdf.geometry.y
    gdf["FID"] = np.arange(1, len(gdf) + 1)

    return gdf
# ============================================================

# ------------------------------------------------------------

### ===== Mapillary pano lookup (vector tiles) ==== ###
def get_features_for_tile(tile, access_token, tileset=TILESET):
    """
    Download a Mapillary vector tile and decode it to GeoJSON features.
    """
    tile_url = (
        f"https://tiles.mapillary.com/maps/vtp/{tileset}/2/"
        f"{tile.z}/{tile.x}/{tile.y}?access_token={access_token}"
    )
    r = requests.get(tile_url, timeout=30)
    r.raise_for_status()
    gj = vt_bytes_to_geojson(r.content, tile.x, tile.y, tile.z, layer="image")
    feats = gj.get("features", []) if isinstance(gj, dict) else []
    return tile, feats


def attach_nearest_panorama_from_tiles(points_gdf_metric, access_token,
                                       max_distance=50, zoom=14, tileset=TILESET, workers=10):
    """
    For each point:
    - determine which Mapillary vector tile it falls in
    - download all unique tiles (parallel)
    - keep only pano features (is_pano=True)
    - build KDTree of pano positions
    - assign nearest pano within max_distance

    Input points must be in a metric CRS (e.g., EPSG:28992) because max_distance is in meters.
    Output returns EPSG:4326 because mapillary uses this CRS.
    """
    if points_gdf_metric.crs is None:
        raise ValueError("points_gdf_metric has no CRS set.")

    local_crs = points_gdf_metric.crs
    pts = points_gdf_metric.copy()

    # Convert to EPSG:4326 for tile addressing
    pts_4326 = pts.to_crs(CRS_WGS84)
    pts_4326["tile"] = [mercantile.tile(x, y, zoom) for x, y in zip(pts_4326.geometry.x, pts_4326.geometry.y)]
    unique_tiles = list(pts_4326["tile"].unique())

    # Download tiles in parallel
    tile_results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(get_features_for_tile, t, access_token, tileset): t for t in unique_tiles}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading Mapillary tiles"):
            tile, feats = fut.result()
            tile_results.append((tile, feats))

    # Flatten and keep only pano features
    pano_features = []
    for _, feats in tile_results:
        for f in feats or []:
            props = f.get("properties", {})
            if props.get("is_pano", False) is True and f.get("geometry", {}).get("type") == "Point":
                pano_features.append(f)

    # If no pano features at all, return empty columns
    if len(pano_features) == 0:
        out = pts.to_crs(CRS_WGS84)
        out["image_id"] = ""
        out["is_panoramic"] = None
        out["distance_m"] = None
        return out

    # Make GeoDataFrame of pano points in EPSG:4326 then project to local CRS for KDTree
    pano_4326 = gpd.GeoDataFrame(
        [
            (
                Point(f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1]),
                str(f["properties"]["id"]),
                True,
            )
            for f in pano_features
        ],
        columns=["geometry", "image_id", "is_panoramic"],
        geometry="geometry",
        crs=CRS_WGS84,
    )

    pano_local = pano_4326.to_crs(local_crs)
    pts_local = pts_4326.to_crs(local_crs)

    # KDTree nearest neighbour in meters
    pano_xy = np.array([[p.x, p.y] for p in pano_local.geometry])
    tree = cKDTree(pano_xy)

    query_xy = np.array([[p.x, p.y] for p in pts_local.geometry])
    distances, indices = tree.query(query_xy, k=1, distance_upper_bound=float(max_distance))

    out = pts_local.copy()
    out["image_id"] = ""
    out["is_panoramic"] = None
    out["distance_m"] = None

    # Vectorised assignment is awkward with upper_bound; loop is fine for your scale (30–few 100s pts)
    for idx_pt, (d, i) in enumerate(zip(distances, indices)):
        if np.isfinite(d) and i < len(pano_local):
            out.iat[idx_pt, out.columns.get_loc("image_id")] = str(pano_local.iloc[int(i)]["image_id"])
            out.iat[idx_pt, out.columns.get_loc("is_panoramic")] = True
            out.iat[idx_pt, out.columns.get_loc("distance_m")] = float(d)

    return out.to_crs(CRS_WGS84)
# ============================================================

### ====  Building distances computations ==== ###
def build_road_segments_gdf(roads_geom):
    """
    Convert merged road geometry into a GeoDataFrame of individual LineStrings,
    so we can use a spatial index to snap points efficiently.

    This avoids: for each point -> loop over ALL road lines to find nearest.
    Instead: use sindex bbox query -> evaluate only nearby candidates.
    """
    lines = [line for line in iter_lines(roads_geom) if (line is not None and not line.is_empty)]
    return gpd.GeoDataFrame({"geometry": lines}, geometry="geometry", crs=CRS_METRIC)


def snap_point_to_nearest_segment(pt, roads_segments_gdf, roads_sindex):
    """
    Snap pt to the nearest road segment (LineString) using spatial index prefilter.

    Steps:
    1) Query index with a small bbox around pt (expanded iteratively if needed)
    2) For candidate segments, compute exact nearest point using project/interpolate
    3) Return snapped point, segment geometry, and along-segment position 's'
    """
    # Start with a small search radius and expand if we find nothing (rare, but safe)
    search_r = 20.0  # meters
    best = None  # (dist, snapped_point, line, s)

    for _ in range(4):  # expand a few times if needed
        bbox = pt.buffer(search_r).bounds
        cand_idx = list(roads_sindex.intersection(bbox))
        if len(cand_idx) == 0:
            search_r *= 2
            continue

        candidates = roads_segments_gdf.iloc[cand_idx]

        for line in candidates.geometry:
            if line.is_empty:
                continue
            s = line.project(pt)
            p = line.interpolate(s)
            d = pt.distance(p)
            if best is None or d < best[0]:
                best = (d, p, line, s)

        if best is not None:
            break  # found something good in this radius

        search_r *= 2

    if best is None:
        raise ValueError("Could not snap point to any road segment.")
    return best[1], best[2], best[3]


def local_bearing_on_line(line, s, eps=5.0):
    """
    Compute local bearing at position s along a line using +/- eps meters.
    """
    s0 = max(s - eps, 0.0)
    s1 = min(s + eps, line.length)
    if s1 == s0:
        return None
    p0 = line.interpolate(s0)
    p1 = line.interpolate(s1)
    return bearing_from_two_points(p0, p1)


def build_perpendicular_strip(pt, tx, ty, nx, ny, length=80, half_width=4):
    """
    Create a rectangle (Polygon) anchored at pt and extending along the normal direction.

    Coordinate frame:
    - (tx, ty): tangent unit vector along the road
    - (nx, ny): normal unit vector perpendicular to the road
    - half_width: half of the strip width along the tangent
    - length: how far the strip extends along the normal

    This polygon represents a "thick ray" that is robust to slight misalignment.
    """
    ox_, oy_ = pt.x, pt.y

    # half-width offsets along road direction
    wx = tx * half_width
    wy = ty * half_width

    # length offsets along normal direction
    lx = nx * length
    ly = ny * length

    p1 = (ox_ - wx, oy_ - wy)
    p2 = (ox_ + wx, oy_ + wy)
    p3 = (ox_ + wx + lx, oy_ + wy + ly)
    p4 = (ox_ - wx + lx, oy_ - wy + ly)

    return Polygon([p1, p2, p3, p4])


def add_left_right_facade_distances_wide_strip(
    points_wgs84,
    buildings_gdf,
    roads_28992,
    strip_length=80,
    strip_half_width=4,
    building_search_dist=60,
    eps_bearing=5,
    max_facade_dist=15.0,
):
    """
    Compute dist_left_m and dist_right_m for each point.

    Key design choices:
    - Snap the point to the nearest road segment first (reduces geometric noise).
    - Derive local road bearing at snap location.
    - Build two strips: left normal and right normal.
    - Intersect strips with nearby building polygons (prefilter via buildings sindex).
    - Distance is computed as *projection* onto the strip normal (not Euclidean),
      so you measure "perpendicular distance" in the intended direction.

    Cutoff:
    - If no building is found within strip OR nearest is farther than max_facade_dist => 0.0
    """
    # Work in metric CRS for distances
    pts = points_wgs84.to_crs(CRS_METRIC).copy()
    bld = buildings_gdf.to_crs(CRS_METRIC).copy()

    # Defensive geometry fix (common with polygon datasets)
    bld["geometry"] = bld.geometry.buffer(0)
    bld = bld[~bld.is_empty].copy()

    # Buildings spatial index (fast bbox prefilter)
    bld_sindex = bld.sindex

    # Build road segments index once
    roads_geom = roads_28992.to_crs(CRS_METRIC).geometry.iloc[0]
    road_segments = build_road_segments_gdf(roads_geom)
    road_sindex = road_segments.sindex

    left_vals = []
    right_vals = []

    for row in pts.itertuples():
        pt = row.geometry
        if pt is None or pt.is_empty:
            left_vals.append(0.0)
            right_vals.append(0.0)
            continue

        # 1) Snap to nearest road segment (efficient via sindex)
        try:
            snapped, seg_line, s = snap_point_to_nearest_segment(pt, road_segments, road_sindex)
        except Exception:
            left_vals.append(0.0)
            right_vals.append(0.0)
            continue

        # 2) Local road bearing
        bearing = local_bearing_on_line(seg_line, s, eps=eps_bearing)
        if bearing is None or not np.isfinite(bearing):
            left_vals.append(0.0)
            right_vals.append(0.0)
            continue

        # Convert bearing to tangent and normals (unit vectors)
        th = np.deg2rad(bearing)
        tx, ty = np.sin(th), np.cos(th)  # tangent (along road)
        nLx, nLy = -ty, tx               # left normal
        nRx, nRy = ty, -tx               # right normal

        # 3) Build wide strips
        strip_L = build_perpendicular_strip(
            snapped, tx, ty, nLx, nLy, length=strip_length, half_width=strip_half_width
        )
        strip_R = build_perpendicular_strip(
            snapped, tx, ty, nRx, nRy, length=strip_length, half_width=strip_half_width
        )

        # 4) Candidate buildings near the snapped point (bbox prefilter)
        cand_idx = list(bld_sindex.intersection(snapped.buffer(building_search_dist).bounds))
        if len(cand_idx) == 0:
            left_vals.append(0.0)
            right_vals.append(0.0)
            continue
        candidates = bld.iloc[cand_idx]

        # 5) Evaluate candidates and pick smallest positive projection distance
        best_L = None
        best_R = None

        for poly in candidates.geometry:
            if poly is None or poly.is_empty:
                continue

            # Left side
            if poly.intersects(strip_L):
                p = nearest_points(snapped, poly)[1]
                v = np.array([p.x - snapped.x, p.y - snapped.y])
                proj = float(v.dot([nLx, nLy]))  # signed distance along left normal
                if proj > 0 and (best_L is None or proj < best_L):
                    best_L = proj

            # Right side
            if poly.intersects(strip_R):
                p = nearest_points(snapped, poly)[1]
                v = np.array([p.x - snapped.x, p.y - snapped.y])
                proj = float(v.dot([nRx, nRy]))  # signed distance along right normal
                if proj > 0 and (best_R is None or proj < best_R):
                    best_R = proj

        # 6) Apply cutoff rule (no hit OR too far => 0.0)
        if best_L is None or best_L > max_facade_dist:
            best_L = 0.0
        if best_R is None or best_R > max_facade_dist:
            best_R = 0.0

        left_vals.append(best_L)
        right_vals.append(best_R)

    pts["dist_left_m"] = left_vals
    pts["dist_right_m"] = right_vals

    return pts.to_crs(CRS_WGS84)
# ============================================================

# ------------------------------------------------------------

### ==== MAIN RUN ==== ###
if __name__ == "__main__":

    # 1) Load neighbourhood polygon (WGS84)
    poly_4326 = load_boundaries(NEIGH_FILE, layer=NEIGH_LAYER)

    # 2) Download and merge residential roads (metric CRS)
    roads_28992 = get_roads_in_neighbourhood(poly_4326)

    # 3) Sample points along roads + compute local bearing
    points_wgs84 = sample_points_with_local_bearing(
        roads_28992, spacing_m=POINT_SPACING_M, eps_m=BEARING_EPS_M
    )

    # 4) Attach nearest Mapillary pano using vector tiles (needs metric for distance threshold)
    points_metric = points_wgs84.to_crs(CRS_METRIC)
    points_with_panos = attach_nearest_panorama_from_tiles(
        points_metric,
        access_token=ACCESS_TOKEN,
        max_distance=MAX_PANO_DIST_M,
        zoom=TILE_ZOOM,
        tileset=TILESET,
        workers=TILE_WORKERS,
    )

    # 5) Load buildings once
    buildings = gpd.read_file(BUILDINGS_FILE, layer=BUILDINGS_LAYER) if BUILDINGS_LAYER else gpd.read_file(BUILDINGS_FILE)

    # 6) Compute left/right facade distances using wide strips + cutoff
    points_final = add_left_right_facade_distances_wide_strip(
        points_with_panos,         # contains points + pano info
        buildings_gdf=buildings,
        roads_28992=roads_28992,
        strip_length=STRIP_LENGTH_M,
        strip_half_width=STRIP_HALF_WIDTH_M,
        building_search_dist=BUILDING_SEARCH_DIST_M,
        eps_bearing=BEARING_EPS_M,
        max_facade_dist=MAX_FACADE_DIST_M,
    )

    # 7) Export
    export_cols = [
        "FID", "xcoord", "ycoord",
        "road_angle",
        "image_id", "is_panoramic", "distance_m",
        "dist_left_m", "dist_right_m",
    ]
    export_cols = [c for c in export_cols if c in points_final.columns]

    points_final[export_cols].to_csv(SAMPLES_OUT, index=False)
    print("Saved CSV to:", SAMPLES_OUT)
    print("Columns exported:", export_cols)
    print("Ready for stage 2: Image downloading and processing.")
