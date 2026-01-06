from dataclasses import dataclass

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

    # Template handling
    resize_template_to_reference: bool = True

    # Debug / UI
    draw_corners: bool = True
    draw_debug_text: bool = True



@dataclass(frozen=True)
class Part2Config:
    # --- Calibration (Part 2A) ---
    calib_images_glob: str = "data/chessboard/*.[jJ][pP][gG]"
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