from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


def get_camera_params_from_config(
    frame_size_wh: Tuple[int, int],
    calib_output_path: str = "outputs/camera/calibration.npz",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Part 5 requirement helper:
    Return (K, dist) to be used by solvePnP/projectPoints.

    Notes:
    - Loads calibration from calib_output_path (produced in Part 2).
    - Scales intrinsics K if the calibration resolution differs from the current video frame.
    """
    # Local imports keep config.py lightweight and avoid import cycles.
    from camera import load_calibration_npz
    from ar_render import scale_K_to_new_size

    p = Path(calib_output_path)
    if not p.is_absolute():
        # Make relative paths robust when running from the project root vs module folder.
        p = Path(__file__).resolve().parent / p

    calib = load_calibration_npz(str(p))
    K_calib = np.asarray(calib.K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(calib.dist, dtype=np.float64).reshape(-1, 1)  # always (N,1)

    K = K_calib
    if tuple(calib.image_size) != tuple(frame_size_wh):
        K = scale_K_to_new_size(K_calib, calib.image_size, frame_size_wh)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    return K, dist

@dataclass(frozen=True)
class Part1Config:
    # Paths (Part 1)
    video_path: str = "data/video_part1.mp4"
    reference_path: str = "data/reference.JPG"
    template_path: str = "data/template.JPG"
    output_path: str = "outputs/videos/part1.mp4"

    # Feature + matching
    use_sift_only: bool = True  # SIFT may require opencv-contrib-python on some installations
    ratio_test: float = 0.7 
    min_matches: int = 30

    # Homography / RANSAC
    ransac_reproj_thresh: float = 3.5

    # --- Tracking stability (temporal) ---
    # Exponential smoothing on the *projected 4 corners* (more stable than smoothing H directly).
    # 0.0 disables smoothing. Typical: 0.6–0.85 (higher = smoother, more lag).
    corner_smoothing_alpha: float = 0.60

    # If tracking fails on a frame (too few matches/inliers), reuse the last good pose for a
    # few frames to avoid flicker/jumps.
    max_hold_frames: int = 3

    # Minimum inliers to accept a homography (in addition to min_matches).
    min_inliers: int = 20

    # Homography method: "RANSAC" or e.g. "USAC_MAGSAC" (if your OpenCV build supports it).
    homography_method: str = "RANSAC"

    # Re-estimate H from inliers only (least-squares) after RANSAC. Usually reduces jitter.
    refine_homography_with_inliers: bool = True

    # Template handling
    resize_template_to_reference: bool = True

    # Debug / UI
    draw_corners: bool = True
    draw_debug_text: bool = True



@dataclass(frozen=True)
class Part2Config:
    # --- Calibration (Part 2A) ---
    # NOTE: data/chessboard/ contains HEIC files renamed to .JPG (OpenCV can't read them).
    # Use chessboard_jpg/ after converting HEIC->JPG properly.
    calib_images_glob: str = "data/chessboard_jpg/*.[jJ][pP][gG]"
    chessboard_cols: int = 8
    chessboard_rows: int = 6
    square_size: float = 1.0
    calib_output_path: str = "outputs/camera/calibration.npz"
    show_detections: bool = False

    # --- Planar AR cube (Part 2B) ---
    video_path: str = "data/video_part1.mp4"
    reference_path: str = "data/reference.JPG"
    cube_output_path: str = "outputs/videos/part2_cube.mp4"

    # Define plane scale in "world units"
    plane_width: float = 1.0

    # Cube placement/size relative to the plane
    cube_size_frac: float = 0.28
    cube_offset_x_frac: float = 0.58
    cube_offset_y_frac: float = 0.48
    cube_height_frac: float = 1.0

    # Debug
    draw_debug_text: bool = True


@dataclass(frozen=True)
class Part3Config:
    """
    Part 3: planar AR with a full 3D model (mesh).
    Uses the same calibration file produced in Part 2.
    """
    calib_output_path: str = "outputs/camera/calibration.npz"
    video_path: str = "data/video_part1.mp4"
    reference_path: str = "data/reference.JPG"

    # Mesh input/output
    model_path: str = ""  # if empty or missing, a small demo mesh is used
    output_path: str = "outputs/videos/part3_model.mp4"

    # Define plane scale in "world units"
    plane_width: float = 1.0

    # Model placement/scale relative to the plane
    model_scale_frac: float = 0.30
    model_offset_x_frac: float = 0.55
    model_offset_y_frac: float = 0.55

    # Model rotation (degrees) before placement on the plane.
    # Use this to make the model "stand" instead of lying down.
    rotate_x_deg: float = -90.0
    rotate_y_deg: float = 0.0
    rotate_z_deg: float = 0.0

    # Rendering controls
    max_faces: int = 3000
    draw_debug_text: bool = True


@dataclass(frozen=True)
class Part5Config:
    """
    Part 5: Multi-plane tracking with portal visualization.

    Camera params (K, dist) must be obtained via config.py (see get_camera_params_from_config()).
    """

    # Inputs
    calib_output_path: str = "outputs/camera/calibration.npz"
    video_path: str = "data/multi_plane_video.mp4"
    reference1_path: str = "data/reference1.JPG"
    reference2_path: str = "data/reference2.JPG"
    reference3_path: str = "data/reference3.JPG"
    env1_360_path: str = "data/env1_360.jpg"
    env2_360_path: str = "data/env2_360.jpg"
    env3_360_path: str = "data/env3_360.jpg"

    # Feature + matching
    # Prefer SIFT when available; ORB is the fallback option (no deep learning).
    feature_type: str = "SIFT"  # "SIFT" or "ORB"
    ratio_test: float = 0.72
    min_matches: int = 30

    # Homography / RANSAC
    ransac_reproj_thresh: float = 3.5
    min_inliers: int = 20
    homography_method: str = "RANSAC"
    refine_homography_with_inliers: bool = True

    # Tracking stability
    max_hold_frames: int = 3
    # Exponential smoothing (0 disables; higher = smoother, more lag)
    outline_smoothing_alpha: float = 0.60  # smooth homography quad used for drawing
    pose_smoothing_alpha: float = 0.50     # smooth (rvec,tvec) used for portal rendering

    # Plane scale in "world units"
    plane_width: float = 1.0

    # Portal visualization (Variant A)
    portal_shape: str = "ellipse"  # "rect" or "ellipse"
    portal_size_frac: float = 0.50  # ~50% of plane dimensions
    portal_fill_bgr: tuple[int, int, int] = (40, 40, 200)
    portal_border_bgr: tuple[int, int, int] = (0, 255, 255)
    portal_border_thickness: int = 6
    portal_alpha: float = 0.90
    portal_tex_size: int = 1000  # portal texture resolution (square) for portal360
    portal_env_sphere_radius: float = 20.0  # keep camera inside sphere -> stable portal360; larger = weaker parallax
    portal_env_blur_ksize: int = 0  # optional blur to hide remap aliasing (0 disables)
    portal_env_fov_scale: float = 6.5  # base zoom OUT factor (bigger = see more)
    portal_env_fov_adaptive: bool = True  # adapt FOV to camera distance from the plane (reduces zoom-in)
    portal_env_fov_ref_distance: float = 1.0  # world-units: distance where adaptive factor==1
    portal_env_sharpen_amount: float = 0.7  # 0 disables; try 0.3–1.0

    # Debug / UI
    draw_plane_outline: bool = True
    draw_debug_text: bool = True

    # Outline visualization (neutral plane drawing)
    outline_color_bgr: tuple[int, int, int] = (0, 255, 0)
    outline_thickness: int = 2
    corner_color_bgr: tuple[int, int, int] = (0, 0, 255)
    corner_radius: int = 5
    stripes_color_bgr: tuple[int, int, int] = (255, 255, 255)
    stripes_thickness: int = 1
    stripes_count: int = 0