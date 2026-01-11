import cv2
import numpy as np
from typing import Optional

from config import Part1Config


class PlanarTracker:
    """
    Simple planar tracker for Part 1 (SIFT + homography).

    Usage:
      tracker = PlanarTracker(reference_bgr, cfg)
      H, corners, dbg = tracker.track(frame_bgr)
    """

    def __init__(self, reference_bgr: np.ndarray, cfg: Part1Config):
        self.cfg = cfg

        # This project uses SIFT features (may require opencv-contrib-python on some installations).
        if cfg.use_sift_only and not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "SIFT is not available. Install opencv-contrib-python and retry."
            )

        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Temporal stabilization state
        self._prev_H: Optional[np.ndarray] = None
        self._prev_corners: Optional[np.ndarray] = None
        self._hold_left: int = 0

        ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        self.kp_ref, self.des_ref = self.sift.detectAndCompute(ref_gray, None)
        if self.des_ref is None or self.kp_ref is None or len(self.kp_ref) < 20:
            raise ValueError("Reference image has too few features. Pick a more feature-rich image.")

        h, w = ref_gray.shape[:2]
        self.ref_corners = np.float32(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        ).reshape(-1, 1, 2)

    def track(self, frame_bgr: np.ndarray):
        """
        Returns:
          H: homography (reference -> frame) or None
          corners: 4 reference corners projected into the frame, or None
          dbg: small dict for on-screen debugging
        """
        H, corners, dbg, _extra = self._track_impl(frame_bgr, want_debug=False)
        return H, corners, dbg

    def track_debug(self, frame_bgr: np.ndarray):
        """
        Like track(), but also returns match/keypoint data useful for notebook visualization.

        Returns:
          H, corners, dbg, extra

        extra contains:
          - kp_frame
          - good_matches (list[cv2.DMatch])
          - good_pairs (list[list[cv2.DMatch]]) for cv2.drawMatchesKnn
          - inlier_mask (Nx1 uint8) aligned with good_matches (or None)
        """
        return self._track_impl(frame_bgr, want_debug=True)

    def _track_impl(self, frame_bgr: np.ndarray, want_debug: bool):
        """
        Shared implementation for track() and track_debug().
        """
        dbg = {"good": 0, "inliers": 0, "held": 0, "smoothed": 0}

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp_fr, des_fr = self.sift.detectAndCompute(frame_gray, None)
        if des_fr is None or kp_fr is None:
            Hh, Ch, dbg2 = self._maybe_hold(dbg)
            extra = {"kp_frame": kp_fr, "good_matches": [], "good_pairs": [], "inlier_mask": None}
            return (Hh, Ch, dbg2, extra) if want_debug else (Hh, Ch, dbg2, None)

        good, good_pairs = _knn_ratio_test(
            self.matcher,
            self.des_ref,
            des_fr,
            k=2,
            ratio=float(self.cfg.ratio_test),
        )
        dbg["good"] = len(good)
        if len(good) < self.cfg.min_matches:
            Hh, Ch, dbg2 = self._maybe_hold(dbg)
            extra = {"kp_frame": kp_fr, "good_matches": good, "good_pairs": good_pairs, "inlier_mask": None}
            return (Hh, Ch, dbg2, extra) if want_debug else (Hh, Ch, dbg2, None)

        src = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_fr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        method = _get_homography_method(getattr(self.cfg, "homography_method", "RANSAC"))
        H, inlier_mask = cv2.findHomography(src, dst, method, float(self.cfg.ransac_reproj_thresh))
        if H is None or inlier_mask is None:
            Hh, Ch, dbg2 = self._maybe_hold(dbg)
            extra = {"kp_frame": kp_fr, "good_matches": good, "good_pairs": good_pairs, "inlier_mask": None}
            return (Hh, Ch, dbg2, extra) if want_debug else (Hh, Ch, dbg2, None)

        dbg["inliers"] = int(inlier_mask.sum())
        min_inliers = int(getattr(self.cfg, "min_inliers", max(10, self.cfg.min_matches // 2)))
        if dbg["inliers"] < min_inliers:
            Hh, Ch, dbg2 = self._maybe_hold(dbg)
            extra = {"kp_frame": kp_fr, "good_matches": good, "good_pairs": good_pairs, "inlier_mask": inlier_mask}
            return (Hh, Ch, dbg2, extra) if want_debug else (Hh, Ch, dbg2, None)

        # Optional refinement: recompute least-squares H using inliers only (often reduces jitter).
        if bool(getattr(self.cfg, "refine_homography_with_inliers", True)):
            try:
                inl = inlier_mask.ravel().astype(bool)
                if int(inl.sum()) >= 4:
                    H2, _ = cv2.findHomography(src[inl], dst[inl], 0)
                    if H2 is not None:
                        H = H2
            except Exception:
                pass

        corners = cv2.perspectiveTransform(self.ref_corners, H)
        H_use, corners_use = self._stabilize(H, corners, dbg)
        self._prev_H = H_use
        self._prev_corners = corners_use
        self._hold_left = int(getattr(self.cfg, "max_hold_frames", 0))

        extra = {"kp_frame": kp_fr, "good_matches": good, "good_pairs": good_pairs, "inlier_mask": inlier_mask}
        return (H_use, corners_use, dbg, extra) if want_debug else (H_use, corners_use, dbg, None)

    def _maybe_hold(self, dbg: dict):
        """
        If configured, reuse last good pose for a few frames to avoid flicker/jumps.
        """
        max_hold = int(getattr(self.cfg, "max_hold_frames", 0))
        if max_hold > 0 and self._prev_H is not None and self._prev_corners is not None and self._hold_left > 0:
            self._hold_left -= 1
            dbg["held"] = 1
            return self._prev_H, self._prev_corners, dbg
        return None, None, dbg

    def _stabilize(self, H: np.ndarray, corners: np.ndarray, dbg: dict):
        """
        Stabilize by smoothing the 4 projected corners in image space, then rebuilding H.
        """
        a = float(getattr(self.cfg, "corner_smoothing_alpha", 0.0))
        if self._prev_corners is None or not (0.0 < a < 1.0):
            return H, corners

        try:
            prev = np.asarray(self._prev_corners, dtype=np.float64).reshape(4, 2)
            curr = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            if not (np.isfinite(prev).all() and np.isfinite(curr).all()):
                return H, corners

            # Higher a = smoother (more weight on previous)
            sm = a * prev + (1.0 - a) * curr
            dbg["smoothed"] = 1

            ref = np.asarray(self.ref_corners, dtype=np.float64).reshape(4, 2)
            H_sm = cv2.getPerspectiveTransform(ref.astype(np.float32), sm.astype(np.float32))
            corners_sm = sm.reshape(-1, 1, 2).astype(np.float32)
            return H_sm, corners_sm
        except Exception:
            return H, corners


def _get_homography_method(name: str) -> int:
    """
    Map a string like "RANSAC" / "USAC_MAGSAC" to the corresponding cv2 constant.
    Falls back to cv2.RANSAC.
    """
    n = str(name or "").strip().upper()
    if not n:
        return cv2.RANSAC
    if hasattr(cv2, n):
        try:
            return int(getattr(cv2, n))
        except Exception:
            return cv2.RANSAC
    return cv2.RANSAC


def _knn_ratio_test(matcher: cv2.BFMatcher, des_ref: np.ndarray, des_fr: np.ndarray, k: int, ratio: float):
    # Always request at least 2 for Lowe's ratio test.
    k2 = max(2, int(k))
    pairs = matcher.knnMatch(des_ref, des_fr, k=k2)

    good = []
    good_pairs = []
    for p in pairs:
        if len(p) < 2:
            continue
        m, n = p[0], p[1]
        if m.distance < float(ratio) * n.distance:
            good.append(m)
            good_pairs.append([m, n])
    return good, good_pairs


def warp_and_overlay(frame_bgr: np.ndarray, template_bgr: np.ndarray, H_ref_to_frame: np.ndarray):
    """
    Warp template into the frame using H, then replace pixels inside the projected reference region.

    Note: We do NOT build the mask from template pixel intensities, because real black pixels
    in the template should still overwrite the frame (they are not "transparent").
    """
    h, w = frame_bgr.shape[:2]
    warped = cv2.warpPerspective(template_bgr, H_ref_to_frame, (w, h))

    # Build a geometric mask by warping an all-ones image (same size as the template).
    # This marks exactly the quadrilateral region where the template lands in the frame.
    tmp_h, tmp_w = template_bgr.shape[:2]
    ones = np.ones((tmp_h, tmp_w), dtype=np.uint8) * 255
    mask = cv2.warpPerspective(ones, H_ref_to_frame, (w, h))

    out = frame_bgr.copy()
    out[mask > 0] = warped[mask > 0]
    return out