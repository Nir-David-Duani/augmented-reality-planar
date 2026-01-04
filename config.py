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
    ratio_test: float = 0.5  # Lowe ratio test threshold (increase to 0.7-0.8 if too few matches)
    min_matches: int = 20

    # Homography / RANSAC
    ransac_reproj_thresh: float = 3.0

    # Template handling
    resize_template_to_reference: bool = True

    # Debug / UI
    draw_corners: bool = True
    draw_debug_text: bool = True