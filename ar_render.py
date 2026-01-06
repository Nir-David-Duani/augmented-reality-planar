import cv2
import numpy as np


def scale_K_to_new_size(K: np.ndarray, from_size_wh: tuple[int, int], to_size_wh: tuple[int, int]) -> np.ndarray:
    """
    Scale intrinsics K when the image is resized from from_size to to_size.
    Sizes are (width, height).
    """
    from_w, from_h = from_size_wh
    to_w, to_h = to_size_wh
    sx = to_w / float(from_w)
    sy = to_h / float(from_h)

    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx  # fx
    K2[1, 1] *= sy  # fy
    K2[0, 2] *= sx  # cx
    K2[1, 2] *= sy  # cy
    return K2


def make_plane_object_points(plane_w: float, plane_h: float) -> np.ndarray:
    """
    4 corners on Z=0 plane, in this order:
      (0,0), (w,0), (w,h), (0,h)
    """
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [plane_w, 0.0, 0.0],
            [plane_w, plane_h, 0.0],
            [0.0, plane_h, 0.0],
        ],
        dtype=np.float32,
    )


def make_cube_points(
    plane_w: float,
    plane_h: float,
    size_frac: float,
    offset_x_frac: float,
    offset_y_frac: float,
    height_frac: float = 1.0,
) -> np.ndarray:
    """
    Returns 8 cube vertices (3D) on top of the plane.

    Cube base is a square inside the plane; cube height is along -Z.
    Fractions are relative to the plane dimensions.
    """
    cube_size = float(size_frac) * float(plane_w)
    cube_h = float(height_frac) * cube_size

    x0 = float(offset_x_frac) * float(plane_w)
    y0 = float(offset_y_frac) * float(plane_h)

    # Keep cube inside plane bounds
    x0 = float(np.clip(x0, 0.0, float(plane_w) - cube_size))
    y0 = float(np.clip(y0, 0.0, float(plane_h) - cube_size))

    return np.array(
        [
            [x0, y0, 0.0],
            [x0 + cube_size, y0, 0.0],
            [x0 + cube_size, y0 + cube_size, 0.0],
            [x0, y0 + cube_size, 0.0],
            [x0, y0, -cube_h],
            [x0 + cube_size, y0, -cube_h],
            [x0 + cube_size, y0 + cube_size, -cube_h],
            [x0, y0 + cube_size, -cube_h],
        ],
        dtype=np.float32,
    )


def draw_cube(img_bgr: np.ndarray, imgpts2d: np.ndarray) -> np.ndarray:
    """
    Draw cube with:
      - base: green filled
      - pillars: blue
      - top: red outline
    """
    img = img_bgr.copy()
    pts = np.int32(imgpts2d).reshape(-1, 2)

    # base (green filled)
    img = cv2.drawContours(img, [pts[:4]], -1, (0, 255, 0), -1)

    # pillars (blue)
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(pts[i]), tuple(pts[j]), (255, 0, 0), 3)

    # top (red outline)
    img = cv2.drawContours(img, [pts[4:]], -1, (0, 0, 255), 3)
    return img

