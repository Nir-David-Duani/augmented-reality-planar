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
    corner_smoothing_alpha: float = 0.05

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
    # Default to the project's main mesh model so Part 3 matches the real demo assets.
    # (You can override this via CLI: `python main.py --part 3 --model_path ...`)
    model_path: str = "data/models/86jfmjiufzv2.obj"
    output_path: str = "outputs/videos/part3_model.mp4"

    # Define plane scale in "world units"
    plane_width: float = 1.0

    # Model placement/scale relative to the plane
    model_scale_frac: float = 0.30
    model_offset_x_frac: float = 0.55
    model_offset_y_frac: float = 0.55

    # Model rotation (degrees) before placement on the plane.
    # Use this to make the model "stand" instead of lying down.
    rotate_x_deg: float = 90.0
    rotate_y_deg: float = 0.0
    rotate_z_deg: float = 90.0

    # Rendering controls
    max_faces: int = 65000
    draw_debug_text: bool = True


@dataclass(frozen=True)
class Part4Config:
    """
    Part 4: Occlusion handling.
    Track planar marker -> solvePnP -> render mesh -> composite with a classical CV foreground mask.
    """

    # Inputs/outputs
    calib_output_path: str = "outputs/camera/calibration.npz"
    video_path: str = "data/part4_occlusion_hand.mp4"
    reference_path: str = "data/part4_reference.JPG"
    model_path: str = "data/models/86jfmjiufzv2.obj"
    output_path: str = "outputs/videos/part4_occlusion.mp4"

    # Plane scale
    plane_width: float = 1.0

    # Mesh placement (same semantics as Part 3)
    model_scale_frac: float = 0.30
    model_offset_x_frac: float = 0.55
    model_offset_y_frac: float = 0.55
    rotate_x_deg: float = 90.0
    rotate_y_deg: float = 0.0
    rotate_z_deg: float = -90.0
    max_faces: int = 65000

    # Pose estimation
    # Notebook used IPPE when available.
    pnp_variant: str = "IPPE"  # "IPPE" or "ITERATIVE"
    use_pnp_ransac: bool = False

    # Foreground mask (HSV) 
    h_min: int = 0
    h_max: int = 10
    s_min: int = 70
    s_max: int = 235
    v_min: int = 75
    v_max: int = 235

    use_median: bool = True
    median_ksize: int = 3
    use_morph: bool = True
    open_ksize: int = 9
    close_ksize: int = 13
    iters: int = 2

    # Optional dilation (to make hand occlude slightly more)
    use_dilate: bool = True
    dilate_ksize: int = 5
    dilate_iters: int = 1


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
    min_matches: int = 20

    # Homography / RANSAC
    ransac_reproj_thresh: float = 4.0
    min_inliers: int = 20
    homography_method: str = "RANSAC"
    refine_homography_with_inliers: bool = True

    # Tracking stability
    # Do NOT hold stale poses when planes leave the view.
    max_hold_frames: int = 0
    # Require N consecutive valid frames before showing a plane/portal.
    # Helps avoid flicker and false-positive jumps when the target is not fully visible.
    min_visible_frames: int = 3
    # Exponential smoothing (0 disables; higher = smoother, more lag)
    outline_smoothing_alpha: float = 0.40  # smooth homography quad used for drawing
    pose_smoothing_alpha: float = 0.35     # smooth (rvec,tvec) used for portal rendering

    # Optional outlier rejection: if a single frame produces a huge pose jump, treat it as invalid.
    # This reduces "popping" at the cost of possibly missing very fast motion.
    reject_pose_jumps: bool = False
    pose_jump_max_trans: float = 0.25      # world units
    pose_jump_max_rot_deg: float = 25.0    # degrees

    # Plane scale in "world units"
    plane_width: float = 1.0

    # Portal visualization (Variant A)
    portal_shape: str = "ellipse"  # "rect" or "ellipse"
    portal_size_frac: float = 0.60  # ~60% of plane dimensions
    portal_fill_bgr: tuple[int, int, int] = (40, 40, 200)
    portal_border_bgr: tuple[int, int, int] = (0, 255, 255)
    portal_border_thickness: int = 6
    portal_alpha: float = 0.90
    portal_tex_size: int = 1000  # portal texture resolution (square) for portal360
    portal_env_blur_ksize: int = 0  # optional blur to hide remap aliasing (0 disables)
    portal_env_sharpen_amount: float = 0.7  # 0 disables; try 0.3–1.0

    # --- Back-wall portal (only mode used in Part 5) ---
    # Ellipse portal mask + a SINGLE textured back wall rectangle at z = -depth.
    portal_backwall_depth: float = 0.33     # world units behind the plane (typical 0.2–0.3)
    portal_backwall_size_frac: float = 1.90  # back wall size relative to portal (1.0 fills portal)
    portal_backwall_alpha: float = 0.85       # blend strength for the back wall texture

    # Debug: render the back wall as a SOLID color (no texture) to sanity-check depth/parallax direction.
    # If your portal "moves the wrong way", first fix the plane normal direction (pose disambiguation),
    # then validate again with this enabled.
    portal_debug_backwall_solid: bool = False
    portal_debug_backwall_bgr: tuple[int, int, int] = (40, 40, 40)

    # --- Portal "window" styling (thin rim + subtle glass reflection) ---
    # Make it feel like a real window to another world (not a chunky border).
    portal_window_style: bool = True

    # Thin rim (drawn on top of the portal content)
    portal_rim_outer_bgr: tuple[int, int, int] = (10, 10, 10)        # dark outer rim
    portal_rim_inner_bgr: tuple[int, int, int] = (230, 255, 255)     # bright inner rim (slight cyan)
    portal_rim_outer_thickness: int = 2
    portal_rim_inner_thickness: int = 1
    portal_rim_inner_alpha: float = 0.55

    # Subtle drop shadow for depth
    portal_shadow_bgr: tuple[int, int, int] = (0, 0, 0)
    portal_shadow_alpha: float = 0.22
    portal_shadow_dx: int = 2
    portal_shadow_dy: int = 3

    # Glass-like overlay inside the portal (applied after texture warp / fill)
    portal_glass_enable: bool = True
    portal_glass_vignette_strength: float = 0.18     # darken near edges (0 disables)
    portal_glass_reflection_strength: float = 0.22   # diagonal highlight (0 disables)
    portal_glass_reflection_blur_ksize: int = 11     # odd; 0 disables
    portal_glass_tint_bgr: tuple[int, int, int] = (255, 255, 255)    # white-ish reflection tint

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