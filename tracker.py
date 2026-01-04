import cv2
import numpy as np

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
        dbg = {"good": 0, "inliers": 0}

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp_fr, des_fr = self.sift.detectAndCompute(frame_gray, None)
        if des_fr is None or kp_fr is None:
            return None, None, dbg

        # KNN matching + Lowe ratio test.
        pairs = self.matcher.knnMatch(self.des_ref, des_fr, k=2)
        good = []
        for p in pairs:
            if len(p) != 2:
                continue
            m, n = p
            if m.distance < self.cfg.ratio_test * n.distance:
                good.append(m)

        dbg["good"] = len(good)
        if len(good) < self.cfg.min_matches:
            return None, None, dbg

        src = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_fr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, inlier_mask = cv2.findHomography(
            src, dst, cv2.RANSAC, self.cfg.ransac_reproj_thresh
        )
        if H is None or inlier_mask is None:
            return None, None, dbg

        dbg["inliers"] = int(inlier_mask.sum())
        if dbg["inliers"] < max(10, self.cfg.min_matches // 2):
            return None, None, dbg

        corners = cv2.perspectiveTransform(self.ref_corners, H)
        return H, corners, dbg


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