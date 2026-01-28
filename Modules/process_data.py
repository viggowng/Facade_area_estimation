import os
import cv2
import numpy as np
import math
from PIL import Image


# Function to convert cylindrical panorama to perspective view
def cyl_to_perspective(cyl_img, fov_deg=60, yaw_deg=0, pitch_deg=0,
                       out_w=800, out_h=600):

    H, W = cyl_img.shape[:2]

    fov = np.deg2rad(float(fov_deg))
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))

    fx = (out_w / 2) / math.tan(fov / 2)
    fy = fx
    f_cyl = W / (2 * math.pi)

    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))

    x = (u - out_w / 2) / fx
    y = (v - out_h / 2) / fy
    z = np.ones_like(x)

    # Pitch rotation (X axis)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    y2 = y * cos_p - z * sin_p
    z2 = y * sin_p + z * cos_p
    x2 = x

    # Yaw rotation (Y axis)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    x3 = x2 * cos_y + z2 * sin_y
    z3 = -x2 * sin_y + z2 * cos_y
    y3 = y2

    theta = np.arctan2(x3, z3)
    h = y3 / np.sqrt(x3**2 + z3**2)

    map_x = ((theta + np.pi) / (2 * np.pi) * W).astype(np.float32)
    map_y = (h * f_cyl + H / 2).astype(np.float32)

    persp = cv2.remap(
        cyl_img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return persp

# =====================================================================
# Undistorts cylindrical panorama to perspective crops left and right
# =====================================================================
def process_image(input_path, road_angle,
                  do_rotate=True, do_undistort=True,
                  fov_deg=90, out_w=800, out_h=600):
    """
    Batch-safe processing:
    - rotates pano to road center
    - creates left/right perspective views
    - saves files using the pano filename stem to avoid overwriting
    """

    pano = cv2.imread(input_path)
    if pano is None:
        raise FileNotFoundError(f"Could not read: {input_path}")

    H, W = pano.shape[:2]
    base = os.path.splitext(os.path.basename(input_path))[0]  # e.g., image_id

    # -------------------------
    # 1) ROTATE TO ROAD CENTER
    # -------------------------
    if do_rotate:
        road_angle = float(road_angle)
        shift_px = int((road_angle / 360) * W)
        pano_rot = np.roll(pano, -shift_px, axis=1)
    else:
        pano_rot = pano

    # Save rotated pano (unique filename)
    rot_folder = input_path.replace("raw_pano_images", "rotated_pano")
    rot_folder = os.path.dirname(rot_folder)
    os.makedirs(rot_folder, exist_ok=True)

    rotated_path = os.path.join(rot_folder, f"rotated_{base}.jpg")
    cv2.imwrite(rotated_path, pano_rot)

    # -------------------------
    # 2) LEFT + RIGHT VIEWS
    # -------------------------
    if do_undistort:
        # LEFT façade (yaw -90)
        left_np = cyl_to_perspective(
            pano_rot, fov_deg=fov_deg, yaw_deg=-90, pitch_deg=0,
            out_w=out_w, out_h=out_h
        )
        left_img = Image.fromarray(cv2.cvtColor(left_np, cv2.COLOR_BGR2RGB))

        left_folder = input_path.replace("raw_pano_images", "facade_crops_left")  # fixed typo
        left_folder = os.path.dirname(left_folder)
        os.makedirs(left_folder, exist_ok=True)

        left_path = os.path.join(left_folder, f"left_{base}.jpg")
        left_img.save(left_path)

        # RIGHT façade (yaw +90)
        right_np = cyl_to_perspective(
            pano_rot, fov_deg=fov_deg, yaw_deg=90, pitch_deg=0,
            out_w=out_w, out_h=out_h
        )
        right_img = Image.fromarray(cv2.cvtColor(right_np, cv2.COLOR_BGR2RGB))

        right_folder = input_path.replace("raw_pano_images", "facade_crops_right")
        right_folder = os.path.dirname(right_folder)
        os.makedirs(right_folder, exist_ok=True)

        right_path = os.path.join(right_folder, f"right_{base}.jpg")
        right_img.save(right_path)

        return left_path, right_path, rotated_path

    return None
