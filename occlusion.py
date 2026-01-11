from __future__ import annotations

import cv2
import numpy as np


def raw_hsv_mask(
    frame_bgr: np.ndarray,
    *,
    h_min: int,
    h_max: int,
    s_min: int,
    s_max: int,
    v_min: int,
    v_max: int,
) -> np.ndarray:
    """
    Binary mask (0/255) from HSV thresholds.
    OpenCV HSV ranges: H=0..179, S=0..255, V=0..255.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = (int(h_min), int(s_min), int(v_min))
    upper = (int(h_max), int(s_max), int(v_max))
    return cv2.inRange(hsv, lower, upper)


def clean_binary_mask(
    mask_u8: np.ndarray,
    *,
    use_median: bool = True,
    median_ksize: int = 3,
    use_morph: bool = True,
    open_ksize: int = 9,
    close_ksize: int = 13,
    iters: int = 2,
) -> np.ndarray:
    """
    Clean a binary mask (0/255).
    Matches the notebook pipeline: optional median -> OPEN -> CLOSE.
    """
    m = np.asarray(mask_u8, dtype=np.uint8).copy()

    if bool(use_median) and int(median_ksize) > 0:
        k = int(median_ksize) | 1
        m = cv2.medianBlur(m, k)

    if bool(use_morph):
        it = max(1, int(iters))

        if int(open_ksize) > 0:
            k = int(open_ksize) | 1
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker, iterations=it)

        if int(close_ksize) > 0:
            k = int(close_ksize) | 1
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=it)

    return m


def dilate_mask(mask_u8: np.ndarray, *, ksize: int = 5, iters: int = 1) -> np.ndarray:
    """
    Expand a binary mask (0/255) with dilation.
    """
    if int(ksize) <= 0:
        return np.asarray(mask_u8, dtype=np.uint8)
    k = int(ksize) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(np.asarray(mask_u8, dtype=np.uint8), ker, iterations=max(1, int(iters)))


def composite_occlusion(frame_bgr: np.ndarray, ar_bgr: np.ndarray, occ_mask_u8: np.ndarray) -> np.ndarray:
    """
    Composite so that:
      mask==255 -> keep real frame (foreground occludes AR)
      mask==0   -> show AR
    """
    m = np.asarray(occ_mask_u8, dtype=np.uint8)
    m3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) if m.ndim == 2 else m
    return np.where(m3 > 0, frame_bgr, ar_bgr).astype(np.uint8)


