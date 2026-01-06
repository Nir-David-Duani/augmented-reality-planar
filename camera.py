from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    K: np.ndarray               # (3,3)
    dist: np.ndarray            # (N,1) or (1,N)
    image_size: Tuple[int, int] # (width, height)
    rms: float                  # RMS reprojection error


def expand_image_glob(glob_pattern: str) -> List[str]:
    return [str(p) for p in sorted(Path().glob(glob_pattern)) if p.is_file()]


def save_calibration_npz(path: str, calib: CameraCalibration) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out),
        K=calib.K,
        dist=calib.dist,
        image_size=np.array(calib.image_size, dtype=np.int32),
        rms=np.array([calib.rms], dtype=np.float32),
    )


def load_calibration_npz(path: str) -> CameraCalibration:
    data = np.load(path, allow_pickle=False)
    image_size = tuple(int(x) for x in data["image_size"].tolist())
    return CameraCalibration(
        K=data["K"],
        dist=data["dist"],
        image_size=image_size,
        rms=float(data["rms"].reshape(-1)[0]),
    )


def calibrate_from_chessboard_images(
    image_paths: List[str],
    pattern_size: Tuple[int, int],  # (cols, rows) inner corners
    square_size: float,
    show_detections: bool = False,
) -> CameraCalibration:
    if len(image_paths) == 0:
        raise ValueError("No calibration images provided.")

    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints = []
    imgpoints = []
    image_size_wh = None

    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[calib] skip unreadable: {p}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if image_size_wh is None:
            image_size_wh = (w, h)
        elif image_size_wh != (w, h):
            print(f"[calib] skip size mismatch: {p}")
            continue

        found, corners = cv2.findChessboardCorners(gray, pattern_size, find_flags)
        if not found or corners is None:
            print(f"[calib] no corners: {p}")
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

        objpoints.append(objp.copy())
        imgpoints.append(corners)

        if show_detections:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            cv2.imshow("calib corners", vis)
            cv2.waitKey(120)

    if show_detections:
        cv2.destroyAllWindows()

    if image_size_wh is None or len(objpoints) < 3:
        raise RuntimeError(f"Not enough valid views. Got {len(objpoints)} (need >= 3).")

    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size_wh, None, None
    )

    return CameraCalibration(K=K, dist=dist, image_size=image_size_wh, rms=float(rms))