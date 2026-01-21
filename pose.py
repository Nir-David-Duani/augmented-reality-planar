"""
Pose / PnP utilities.

Key invariant for planar AR portals:
- The plane normal (object +Z axis, in camera coordinates) must face the camera.
  Otherwise depth/parallax cues invert.

We enforce this deterministically by preferring an IPPE solution whose normal satisfies:
    (R[:,2] · (-t)) > 0
Where R,t are the solvePnP pose mapping object -> camera coordinates.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def plane_normal_camera_from_rvec(rvec: np.ndarray) -> np.ndarray:
    """Return plane normal in camera coordinates (3,), assuming object plane normal is +Z."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    n = R[:, 2].reshape(3)
    return n


def reprojection_rmse(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    """RMSE reprojection error in pixels (lower is better)."""
    obj = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    proj, _ = cv2.projectPoints(obj, np.asarray(rvec, dtype=np.float64).reshape(3, 1), np.asarray(tvec, dtype=np.float64).reshape(3, 1), K, dist)
    p = proj.reshape(-1, 2)
    d = p - img
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def solve_planar_pnp_facing_camera(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    prefer_ippe: bool = True,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Solve pose for planar target. If IPPE is available, use solvePnPGeneric(SOLVEPNP_IPPE)
    and deterministically pick the solution whose plane normal faces the camera.

    Returns (ok, rvec, tvec).
    """
    obj = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    # Prefer IPPE if possible: it can return multiple valid solutions for planar points.
    if prefer_ippe and hasattr(cv2, "solvePnPGeneric") and hasattr(cv2, "SOLVEPNP_IPPE"):
        try:
            ok, rvecs, tvecs, _extra = cv2.solvePnPGeneric(obj, img, K, dist, flags=int(cv2.SOLVEPNP_IPPE))
        except cv2.error:
            ok = False
            rvecs, tvecs = (), ()

        if bool(ok) and rvecs is not None and tvecs is not None and len(rvecs) > 0:
            candidates = []
            for rv, tv in zip(rvecs, tvecs):
                rv = np.asarray(rv, dtype=np.float64).reshape(3, 1)
                tv = np.asarray(tv, dtype=np.float64).reshape(3, 1)
                if not (np.isfinite(rv).all() and np.isfinite(tv).all()):
                    continue

                n_cam = plane_normal_camera_from_rvec(rv)      # camera coords
                to_cam = (-tv).reshape(3)                      # camera coords
                facing_penalty = 0 if float(n_cam.dot(to_cam)) > 0.0 else 1
                err = reprojection_rmse(obj, img, rv, tv, K, dist)
                candidates.append((facing_penalty, err, rv, tv))

            if candidates:
                # Primary: facing_penalty (0 preferred). Secondary: reprojection error.
                candidates.sort(key=lambda x: (x[0], x[1]))
                _pen, _err, rv_best, tv_best = candidates[0]
                return True, rv_best, tv_best

    # Fallback: standard iterative PnP (single solution).
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=int(cv2.SOLVEPNP_ITERATIVE))
        if not bool(ok):
            return False, None, None
        return True, np.asarray(rvec, dtype=np.float64).reshape(3, 1), np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    except cv2.error:
        return False, None, None





