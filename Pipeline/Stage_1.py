# ============================================================
# PIPELINE STAGE 1: Prepping road network + SVI locations
# ============================================================

from vt2geojson.tools import vt_bytes_to_geojson
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely.ops import unary_union, linemerge, nearest_points
from scipy.spatial import cKDTree

import geopandas as gpd
import pandas as pd
import osmnx as ox
import mercantile
import matplotlib
matplotlib.use("Agg")  # <- IMPORTANT: no Tkinter
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import requests
import math
import os

from pathlib import Path
from Stage_0 import PROJECT_ROOT, CSV_PATHS, MAPILLARY_TOKEN


### ==== CONFIGURATION ==== ###

# Unique mapillary token
ACCESS_TOKEN = MAPILLARY_TOKEN

# Case study area
NEIGH_FILE = PROJECT_ROOT / "Input_data" / "Neighbourhood_bounds.gpkg"
NEIGH_LAYER = None

# BAG datafile 
BUILDINGS_FILE = PROJECT_ROOT / "Input_data" / "Buildings.gpkg"
BUILDINGS_LAYER = None

# If chosen for manual point input
POINTS_MANUAL = PROJECT_ROOT / "Input_data" / "Manual_points.gpkg"
POINTS_MANUAL_LAYER = None

# Output .csv 
SAMPLES_OUT = CSV_PATHS["road_network"] / "sample_points.csv"

# CRS
CRS_WGS84 = "EPSG:4326"    # WSG84 (mapillary crs)
CRS_METRIC = "EPSG:28992"  # RD New crs

# Crossroad deletion parameters
PUNCH_INTERSECTION_HOLES = True
INTERSECTION_DEGREE_MIN = 4      # 4 = crossroads; set to 3 to also remove T-junctions
INTERSECTION_BUFFER_M = 7        # <-- iterative testing

# Road sampling
POINT_SPACING_M = 30            # Distance between sample points along roads
BEARING_EPS_M = 5               # how far to look forward/back along a road line to derive road angle

# Mapillary tile lookup
TILE_ZOOM = 14
MAX_PANO_DIST_M = 5            # max tolerated distance between sample coordinates and nearest Mapillary pano
TILESET = "mly1_public"
TILE_WORKERS = 10

# Building distance search parameters
STRIP_LENGTH_M = 80             # max distance to search for facades perpendicular to road
STRIP_HALF_WIDTH_M = 4          # half-width of the perpendicular search strip -> prevents the ray from missing slightly offset buildings
BUILDING_SEARCH_DIST_M = 20     # candidate building search radius (bbox prefilter)
MAX_FACADE_DIST_M = 20          # if distance is farther, it is set to 0.0 (parameter found through iterative testing)

# "auto" or "manual" to toggle between manual point insertion or automatic point creation
SAMPLING_MODE = "manual"  

# Creates a map of the output road network + sampling points
PLOT_MAP = True
PLOT_OUT = CSV_PATHS["road_network"] / "road_sampling_map.png"
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
    Loads a neighbourhood polygon layer and returns a SINGLE (multi)polygon in EPSG:4326.
    If the layer contains multiple features, they are dissolved/unioned into one geometry.
    """
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)

    if len(gdf) == 0:
        raise ValueError("Neighbourhood polygon file has no features.")
    if gdf.crs is None:
        raise ValueError("Neighbourhood polygon has no CRS.")

    # Union/dissolve all features into one geometry (instead of taking iloc[0])
    poly = gdf.geometry.union_all()

    poly_4326 = gpd.GeoSeries([poly], crs=gdf.crs).to_crs(CRS_WGS84).iloc[0]

    # fix minor validity issues
    if not poly_4326.is_valid:
        poly_4326 = poly_4326.buffer(0)

    return poly_4326

def remove_intersections(edges_gdf, G_proj, buffer_m=10.0, degree_min=4):
    """
    Removes road parts around intersection nodes by buffering nodes and subtracting the buffer.
    """
    # Nodes in metric CRS
    nodes = ox.graph_to_gdfs(G_proj, nodes=True, edges=False)

    # Graph topological degree
    deg = dict(G_proj.degree())
    nodes["degree"] = nodes.index.map(deg)

    # Select intersections
    inter = nodes[nodes["degree"] >= degree_min].copy()
    if len(inter) == 0:
        return edges_gdf, None, nodes

    # Buffer intersection nodes and union
    inter_buf = inter.geometry.buffer(float(buffer_m))
    holes = inter_buf.union_all()

    out = edges_gdf.copy()
    out["geometry"] = out.geometry.apply(
        lambda g: g.difference(holes) if g is not None and not g.is_empty else g
    )

    # explode multipart leftovers + drop empties
    out = out.explode(index_parts=False).reset_index(drop=True)
    out = out[~out.geometry.is_empty].copy()
    out = out[out.geometry.length > 0.5].copy()  # remove tiny artifacts

    return out, inter_buf, nodes

def plot_roads_and_points(edges_metric, points_wgs84, boundary_4326=None, inter_buf=None, out_png=None, title=None):

    pts_m = points_wgs84.to_crs(CRS_METRIC)

    fig, ax = plt.subplots(figsize=(12, 12))

    # --------------------
    # NEW: plot boundary
    # --------------------
    if boundary_4326 is not None:
        boundary_m = gpd.GeoSeries([boundary_4326], crs=CRS_WGS84).to_crs(CRS_METRIC)
        boundary_m.boundary.plot(
            ax=ax,
            linewidth=2.0,
            linestyle="--"
        )

    # roads
    edges_metric.plot(ax=ax, linewidth=0.8)

    # intersection buffers
    if inter_buf is not None and len(inter_buf) > 0:
        gpd.GeoSeries(inter_buf, crs=CRS_METRIC).plot(ax=ax, alpha=0.25)

    # points
    pts_m.plot(ax=ax, markersize=10)

    ax.set_aspect("equal", "box")
    ax.set_axis_off()

    if title:
        ax.set_title(title)

    if out_png:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print("Saved debug plot to:", out_png)

    plt.close(fig)


# ============================================================

# ------------------------------------------------------------

### ==== Road network creation -> sample points + attributes ==== ###
def get_roads_in_neighbourhood(poly_4326):
    """
    Downloads OSM roads in the neighbourhood polygon (WGS84), projects to EPSG:28992.
    Optionally punches holes around intersections before merging, to avoid sampling on crossroads.

    Returns:
      roads_merged_gdf (GeoDataFrame): one merged Line/MultiLine geometry in CRS_METRIC
      edges_clean (GeoDataFrame): individual road geometries after clipping and (optional) punching
      inter_buf (GeoSeries or None): intersection buffers used (for plotting)
    """

    cf = '["highway"~"residential|living_street|unclassified|service|tertiary|secondary"]'

    G = ox.graph_from_polygon(
    poly_4326,
    simplify=True,
    custom_filter=cf
)

    G_proj = ox.project_graph(G, to_crs=CRS_METRIC)
    _, edges = ox.graph_to_gdfs(G_proj)

    # Now clip in metric CRS (your existing clip)
    poly_28992 = gpd.GeoSeries([poly_4326], crs=CRS_WGS84).to_crs(CRS_METRIC).iloc[0]
    edges = edges.clip(poly_28992)


    # Keep only geometries
    edges = edges[["geometry"]].copy()
    edges = edges.reset_index(drop=True)
    edges.set_crs(CRS_METRIC, inplace=True)

    inter_buf = None

    # ---- Removes intersections ----
    if PUNCH_INTERSECTION_HOLES:
        edges_clean, inter_buf, _nodes = remove_intersections(
            edges_gdf=edges,
            G_proj=G_proj,
            buffer_m=INTERSECTION_BUFFER_M,
            degree_min=INTERSECTION_DEGREE_MIN,
        )
        print(f"[roads] punched holes: degree>={INTERSECTION_DEGREE_MIN}, buffer={INTERSECTION_BUFFER_M}m, "
              f"edges {len(edges)} -> {len(edges_clean)}")
    else:
        edges_clean = edges

    # Merge to one geometry for evenly spread samples
    merged = linemerge(unary_union(edges_clean.geometry))
    roads_merged = gpd.GeoDataFrame(geometry=[merged], crs=CRS_METRIC)

    return roads_merged, edges_clean, inter_buf

# Code when doing manual point creation (.gpkg input file)
def manual_points_with_attributes(
    roads_28992,
    points_gpkg,
    points_layer=None,
    input_crs=CRS_WGS84,
    bearing_eps_m=BEARING_EPS_M,
):
    """
    Create a test/evaluation sample set from manually defined points (GeoPackage)
    and compute road_angle by snapping points to the nearest road segment.

    Input:
      - points_gpkg: path to a vector file (GPKG/GeoJSON/Shapefile etc.) with point geometry
      - points_layer: optional layer name (only needed if gpkg has multiple layers)

    Output:
      GeoDataFrame in EPSG:4326 with columns:
        - geometry (Point)
        - road_angle (bearing at snapped road segment)
        - xcoord, ycoord
        - FID

    Notes:
      - Points are snapped to the nearest road segment before bearing is computed,
        to make bearings consistent with your distance + strip method.
      - If bearing cannot be computed for a point, road_angle is set to None.
    """
    # -----------------------------
    # 1) Load points
    # -----------------------------
    gdf = gpd.read_file(points_gpkg, layer=points_layer) if points_layer else gpd.read_file(points_gpkg)

    if gdf.crs is None:
        # assume WGS84 if not set, but better to set in QGIS/export
        gdf = gdf.set_crs(input_crs)

    if not all(gdf.geometry.geom_type.isin(["Point", "MultiPoint"])):
        raise ValueError("Input vector must contain Point geometries.")

    # explode multipoints if any
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # -----------------------------
    # 2) Work in metric CRS
    # -----------------------------
    pts_28992 = gdf.to_crs(CRS_METRIC).copy()

    # Build road segments + sindex once
    roads_geom = roads_28992.to_crs(CRS_METRIC).geometry.iloc[0]
    road_segments = build_road_segments_gdf(roads_geom)
    road_sindex = road_segments.sindex

    bearings = []
    snapped_points = []

    # -----------------------------
    # 3) Snap + bearing per point
    # -----------------------------
    snap_fail = 0
    bear_fail = 0

    for row in pts_28992.itertuples():
        pt = row.geometry
        if pt is None or pt.is_empty:
            bearings.append(None)
            snapped_points.append(pt)
            continue

        try:
            snapped, seg_line, s = snap_point_to_nearest_segment(pt, road_segments, road_sindex)
        except Exception as e:
            snap_fail += 1
            snapped_points.append(pt)
            bearings.append(None)
            continue

        bearing = local_bearing_on_line(seg_line, s, eps=bearing_eps_m)
        if bearing is None or not np.isfinite(bearing):
            bear_fail += 1

        snapped_points.append(snapped)
        bearings.append(bearing)

    print(f"[manual] snap_fail={snap_fail}, bear_fail={bear_fail}, n={len(pts_28992)}")

    pts_28992["geometry"] = snapped_points
    pts_28992["road_angle"] = bearings

    # -----------------------------
    # 4) Export in EPSG:4326 + standard columns
    # -----------------------------
    out = pts_28992.to_crs(CRS_WGS84)
    out["xcoord"] = out.geometry.x
    out["ycoord"] = out.geometry.y

    # If user already has an ID column you can keep it; otherwise create FID
    if "FID" not in out.columns:
        out["FID"] = np.arange(1, len(out) + 1)

    # Match output contract of sample_points_with_local_bearing
    keep_cols = [c for c in ["FID", "xcoord", "ycoord", "road_angle", "geometry"] if c in out.columns]
    return out[keep_cols].copy()

# code for automatic sampling (points every X meters along road network)
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
                                       max_distance=MAX_PANO_DIST_M, zoom=14, tileset=TILESET, workers=10):
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
    max_facade_dist=MAX_FACADE_DIST_M,
):
    """
    Compute dist_left_m and dist_right_m for each point.

    Key design choices:
    - Snap the point to the nearest road segment first (reduces geometric noise).
    - Derive local road bearing at snap location.
    - Build two strips: left normal and right normal.
    - Intersect strips with nearby building polygons (prefilter via buildings sindex).
    - Distance is computed as *projection* onto the strip normal,
      so it measures "perpendicular distance" in the intended direction, matching the direction of the camera

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

    # 2) Download roads + punch holes + get edges for plotting
    roads_28992, edges_clean_28992, inter_buf = get_roads_in_neighbourhood(poly_4326)

    # 3) Sample points (auto) OR uses manual evaluation points (toggle in config)
    if SAMPLING_MODE.lower() == "auto":
        points_wgs84 = sample_points_with_local_bearing(
            roads_28992, spacing_m=POINT_SPACING_M, eps_m=BEARING_EPS_M
        )
        print(f"Sampling mode = AUTO, points = {len(points_wgs84)}")

    elif SAMPLING_MODE.lower() == "manual":
        points_wgs84 = manual_points_with_attributes(
            roads_28992,
            points_gpkg=POINTS_MANUAL,
            points_layer=POINTS_MANUAL_LAYER,
            input_crs=CRS_WGS84,
            bearing_eps_m=BEARING_EPS_M,
        )
        print(f"Sampling mode = MANUAL, points = {len(points_wgs84)}")

    else:
        raise ValueError("SAMPLING_MODE must be 'auto' or 'manual'")

    # ---- Plots a map of roads + sample points ----
    if PLOT_MAP:
        plot_roads_and_points(
            edges_metric=edges_clean_28992,
            points_wgs84=points_wgs84,
            boundary_4326=poly_4326,   # <<< ADD THIS
            inter_buf=inter_buf,
            out_png=PLOT_OUT,
            title=f"Roads after intersection holes + sample points (spacing={POINT_SPACING_M}m)",
        )
    # --------------------------------------------------------

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
        points_with_panos,
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

