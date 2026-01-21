"""
Part 5 – Multi-Plane Tracking with Portal Visualization (Variant A).

Goal:
- Track 3 different planar targets simultaneously (3 reference images).
- For each plane: feature matching -> homography -> 2D corners -> solvePnP -> pose (rvec/tvec)
- Render a "portal" (simple colored rectangle) in the center of each plane with correct perspective.

Constraints:
- Classical CV only (SIFT/ORB), no deep learning.
- Camera intrinsics/distortion must come from config.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import Part5Config, get_camera_params_from_config
from pose import solve_planar_pnp_facing_camera

# Small caches to avoid re-allocating large grids every frame.
_PORTAL_GRID_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _project_points_camera_frame(pts3_cam: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Project 3D points that are ALREADY in camera coordinates.
    We still pass dist so distortion is applied consistently.
    """
    pts3_cam = np.asarray(pts3_cam, dtype=np.float64).reshape(-1, 3)
    r0 = np.zeros((3, 1), dtype=np.float64)
    t0 = np.zeros((3, 1), dtype=np.float64)
    img, _ = cv2.projectPoints(pts3_cam, r0, t0, K, dist)
    return img.reshape(-1, 2)


def _object_to_camera_points(obj_pts: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pts_cam Nx3, R 3x3, t 3,) for object->camera pose."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    P = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3).T  # 3xN
    pts_cam = (R @ P + t).T
    return pts_cam, R, t.reshape(3)


def _plane_normal_cam_facing_camera(R: np.ndarray, t_cam: np.ndarray) -> np.ndarray:
    """
    Plane normal in camera coordinates (object +Z axis), forced to face the camera.
    Uses dot(normal, -t) > 0 criterion (all in camera frame).
    """
    n = np.asarray(R, dtype=np.float64).reshape(3, 3)[:, 2].reshape(3)
    to_cam = -np.asarray(t_cam, dtype=np.float64).reshape(3)
    if float(n.dot(to_cam)) < 0.0:
        n = -n
    return n


def _imread_any(path: str) -> Optional[np.ndarray]:
    """Windows-friendly image read (handles some unicode/path quirks). Returns BGR or None."""
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _get_homography_method(name: str) -> int:
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
    k2 = max(2, int(k))
    pairs = matcher.knnMatch(des_ref, des_fr, k=k2)

    good = []
    for p in pairs:
        if len(p) < 2:
            continue
        m, n = p[0], p[1]
        if m.distance < float(ratio) * n.distance:
            good.append(m)
    return good


def _make_centered_plane_object_points(plane_w: float, plane_h: float) -> np.ndarray:
    """
    4 corners on Z=0 plane, centered at (0,0,0), in this order:
      (-w/2,-h/2), (w/2,-h/2), (w/2,h/2), (-w/2,h/2)
    """
    hw = float(plane_w) * 0.5
    hh = float(plane_h) * 0.5
    return np.array(
        [
            [-hw, -hh, 0.0],
            [hw, -hh, 0.0],
            [hw, hh, 0.0],
            [-hw, hh, 0.0],
        ],
        dtype=np.float32,
    )


def _alpha_fill_convex_poly(img_bgr: np.ndarray, poly2d: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    """Fill a convex polygon with alpha blending."""
    a = float(alpha)
    if not (0.0 < a <= 1.0):
        return img_bgr
    pts = np.asarray(poly2d, dtype=np.int32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return img_bgr

    overlay = img_bgr.copy()
    cv2.fillConvexPoly(overlay, pts, color_bgr, lineType=cv2.LINE_AA)
    out = cv2.addWeighted(overlay, a, img_bgr, 1.0 - a, 0.0)
    return out


def _alpha_polyline(img_bgr: np.ndarray, pts2d: np.ndarray, color_bgr: tuple[int, int, int], thickness: int, alpha: float) -> np.ndarray:
    """Draw a polyline with alpha blending (useful for subtle rims/highlights)."""
    a = float(alpha)
    if a <= 0.0:
        return img_bgr
    a = min(a, 1.0)

    pts = np.asarray(pts2d, dtype=np.int32).reshape(-1, 1, 2)
    if pts.shape[0] < 3:
        return img_bgr

    overlay = img_bgr.copy()
    cv2.polylines(overlay, [pts], True, tuple(int(x) for x in color_bgr), int(thickness), cv2.LINE_AA)
    return cv2.addWeighted(overlay, a, img_bgr, 1.0 - a, 0.0)


def _portal_mask_from_poly(frame_shape_hw: tuple[int, int], poly2d: np.ndarray) -> np.ndarray:
    """Binary mask (0/255) for a convex polygon in frame coordinates."""
    h, w = int(frame_shape_hw[0]), int(frame_shape_hw[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(poly2d, dtype=np.int32).reshape(-1, 1, 2)
    if pts.shape[0] >= 3:
        cv2.fillConvexPoly(mask, pts, 255, lineType=cv2.LINE_AA)
    return mask


def _apply_portal_glass_effect(out_bgr: np.ndarray, mask_u8: np.ndarray, cfg: Part5Config) -> np.ndarray:
    """
    Subtle "window glass" overlay inside the portal:
    - gentle vignette (darken edges)
    - diagonal reflection highlight (soft, low alpha)
    """
    if not bool(getattr(cfg, "portal_glass_enable", True)):
        return out_bgr

    m = np.asarray(mask_u8, dtype=np.uint8)
    if m.ndim != 2 or m.size == 0:
        return out_bgr

    ys, xs = np.where(m > 0)
    if ys.size < 16:
        return out_bgr

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    # Clamp ROI bounds
    h, w = out_bgr.shape[:2]
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    if y1 <= y0 or x1 <= x0:
        return out_bgr

    roi = out_bgr[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    mroi = (m[y0 : y1 + 1, x0 : x1 + 1] > 0)

    H, W = roi.shape[:2]
    # Normalized coords in ROI (0..1)
    xv = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    yv = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]

    # Vignette strength: stronger near edges
    vig_s = float(getattr(cfg, "portal_glass_vignette_strength", 0.0))
    if vig_s > 0.0:
        dx = (xv - 0.5) / 0.5
        dy = (yv - 0.5) / 0.5
        r = np.sqrt(dx * dx + dy * dy)
        vig = np.clip(r, 0.0, 1.0)  # 0 center -> 1 edges
        # darken factor in [1-vig_s, 1]
        factor = (1.0 - vig_s * vig).astype(np.float32)
        for c in range(3):
            ch = roi[:, :, c]
            ch[mroi] = ch[mroi] * factor[mroi]
            roi[:, :, c] = ch

    # Reflection: diagonal ramp (top-left -> bottom-right)
    ref_s = float(getattr(cfg, "portal_glass_reflection_strength", 0.0))
    if ref_s > 0.0:
        ramp = np.clip(1.0 - (0.75 * xv + 0.95 * yv), 0.0, 1.0).astype(np.float32)
        k = int(getattr(cfg, "portal_glass_reflection_blur_ksize", 0))
        if k and k > 1:
            if k % 2 == 0:
                k += 1
            ramp = cv2.GaussianBlur(ramp, (k, k), 0.0)

        tint = np.asarray(getattr(cfg, "portal_glass_tint_bgr", (255, 255, 255)), dtype=np.float32).reshape(1, 1, 3)
        a = (ref_s * ramp).astype(np.float32)
        a3 = a[:, :, None]
        # blend only inside mask
        roi[mroi] = (1.0 - a3[mroi]) * roi[mroi] + a3[mroi] * tint

    out = out_bgr.copy()
    out[y0 : y1 + 1, x0 : x1 + 1] = np.clip(roi, 0, 255).astype(np.uint8)
    return out


def _draw_portal_window_rim(out_bgr: np.ndarray, border_pts2d: np.ndarray, cfg: Part5Config) -> np.ndarray:
    """
    Thin "window rim" styling (not thick):
    - subtle shadow offset
    - thin dark outer rim
    - thin bright inner rim with alpha
    """
    if not bool(getattr(cfg, "portal_window_style", True)):
        # keep legacy border if user wants it
        return out_bgr

    pts = np.asarray(border_pts2d, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3 or (not np.isfinite(pts).all()):
        return out_bgr

    shadow_alpha = float(getattr(cfg, "portal_shadow_alpha", 0.0))
    if shadow_alpha > 0.0:
        dx = float(getattr(cfg, "portal_shadow_dx", 2))
        dy = float(getattr(cfg, "portal_shadow_dy", 3))
        sh_pts = pts + np.array([dx, dy], dtype=np.float32)
        out_bgr = _alpha_polyline(
            out_bgr,
            sh_pts,
            tuple(getattr(cfg, "portal_shadow_bgr", (0, 0, 0))),
            thickness=int(getattr(cfg, "portal_rim_outer_thickness", 2)),
            alpha=shadow_alpha,
        )

    # Outer rim (thin)
    out_bgr = _alpha_polyline(
        out_bgr,
        pts,
        tuple(getattr(cfg, "portal_rim_outer_bgr", (10, 10, 10))),
        thickness=int(getattr(cfg, "portal_rim_outer_thickness", 2)),
        alpha=1.0,
    )

    # Inner rim (thin, semi-transparent)
    out_bgr = _alpha_polyline(
        out_bgr,
        pts,
        tuple(getattr(cfg, "portal_rim_inner_bgr", (230, 255, 255))),
        thickness=int(getattr(cfg, "portal_rim_inner_thickness", 1)),
        alpha=float(getattr(cfg, "portal_rim_inner_alpha", 0.55)),
    )
    return out_bgr


def _lerp2(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - float(t)) * a + float(t) * b


def _warp_texture_onto_quad(
    frame_bgr: np.ndarray,
    tex_bgr: np.ndarray,
    quad2d: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """
    Warp a texture image onto a convex quadrilateral region in the frame.
    quad2d: (4,2) float, order must match texture corners: TL, TR, BR, BL.
    """
    h, w = frame_bgr.shape[:2]
    tex_h, tex_w = tex_bgr.shape[:2]
    src = np.float32([[0, 0], [tex_w - 1, 0], [tex_w - 1, tex_h - 1], [0, tex_h - 1]])
    dst = np.asarray(quad2d, dtype=np.float32).reshape(4, 2)

    if not (np.isfinite(dst).all()):
        return frame_bgr

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tex_bgr, H, (w, h))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst), 255, lineType=cv2.LINE_AA)

    a = float(alpha)
    if a <= 0.0:
        return frame_bgr
    a = min(a, 1.0)

    out = frame_bgr.copy()
    # Blend only inside mask
    idx = mask > 0
    out[idx] = (a * warped[idx] + (1.0 - a) * out[idx]).astype(np.uint8)
    return out


def _warp_texture_onto_quad_with_frame_mask(
    frame_bgr: np.ndarray,
    tex_bgr: np.ndarray,
    quad2d: np.ndarray,
    frame_mask_u8: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """
    Like _warp_texture_onto_quad, but blends only where frame_mask_u8 > 0.
    frame_mask_u8 is in frame coordinates (HxW).
    """
    h, w = frame_bgr.shape[:2]
    if frame_mask_u8 is None or frame_mask_u8.shape[:2] != (h, w):
        return _warp_texture_onto_quad(frame_bgr, tex_bgr, quad2d, alpha=alpha)

    tex_h, tex_w = tex_bgr.shape[:2]
    src = np.float32([[0, 0], [tex_w - 1, 0], [tex_w - 1, tex_h - 1], [0, tex_h - 1]])
    dst = np.asarray(quad2d, dtype=np.float32).reshape(4, 2)

    if not (np.isfinite(dst).all()):
        return frame_bgr

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tex_bgr, H, (w, h))

    a = float(alpha)
    if a <= 0.0:
        return frame_bgr
    a = min(a, 1.0)

    out = frame_bgr.copy()
    idx = frame_mask_u8 > 0
    out[idx] = (a * warped[idx] + (1.0 - a) * out[idx]).astype(np.uint8)
    return out


def _project_portal_ellipse(
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    rx: float,
    ry: float,
    n: int = 96,
) -> np.ndarray:
    """Project an ellipse lying on the portal plane (Z=0), centered at origin."""
    n = int(n)
    if n < 12:
        n = 12
    ts = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    pts3 = np.stack([rx * np.cos(ts), ry * np.sin(ts), np.zeros_like(ts)], axis=1).astype(np.float64)
    img, _ = cv2.projectPoints(pts3, rvec, tvec, K, dist)
    return img.reshape(-1, 2)


def _camera_center_in_object(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Camera center in object coordinates: C = -R^T t."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    C = -R.T @ t
    return C.reshape(3)


def _render_portal_from_equirect_sphere(
    pano_bgr: np.ndarray,
    C_obj: np.ndarray,
    *,
    portal_w: float,
    portal_h: float,
    out_size: int,
    sphere_radius: float,
) -> np.ndarray:
    """
    Render a portal texture by raycasting from camera center (in plane/object coords)
    through portal plane points (Z=0) and intersecting a textured sphere.

    This creates a translation-dependent effect (parallax) because rays originate at C_obj.
    """
    out_size = int(out_size)
    if out_size <= 32:
        out_size = 32

    pano_h, pano_w = pano_bgr.shape[:2]
    if pano_h < 2 or pano_w < 2:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    # Ensure camera center is inside the sphere to avoid rays missing the sphere.
    # (If camera is outside, many rays won't intersect and remap becomes mostly black.)
    R = float(sphere_radius)
    C_obj = np.asarray(C_obj, dtype=np.float64).reshape(3)
    C_norm = float(np.linalg.norm(C_obj))
    if C_norm >= 0.98 * R:
        if C_norm < 1e-9:
            C_obj = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            C_obj = C_obj * ((0.95 * R) / C_norm)

    # Create grid on portal plane (centered) in object coords.
    # Cache the normalized grid to avoid meshgrid allocations every frame.
    if out_size not in _PORTAL_GRID_CACHE:
        xsn = np.linspace(-0.5, 0.5, out_size, dtype=np.float32)
        ysn = np.linspace(-0.5, 0.5, out_size, dtype=np.float32)
        Xn, Yn = np.meshgrid(xsn, ysn)
        _PORTAL_GRID_CACHE[out_size] = (Xn, Yn)
    Xn, Yn = _PORTAL_GRID_CACHE[out_size]
    X = (Xn * float(portal_w)).astype(np.float32, copy=False)
    Y = (Yn * float(portal_h)).astype(np.float32, copy=False)
    P = np.stack([X, Y, np.zeros_like(X)], axis=-1).reshape(-1, 3).astype(np.float64)  # (N,3)

    C = np.asarray(C_obj, dtype=np.float64).reshape(1, 3)
    d = P - C  # (N,3)
    dn = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / (dn + 1e-9)

    # Ray-sphere intersection: ||C + t d||^2 = R^2
    # quadratic: t^2 + 2*(C·d)*t + (||C||^2 - R^2) = 0 (since ||d||=1)
    Cd = np.sum(C * d, axis=1)  # (N,)
    C2 = float(np.sum(C * C))
    disc = Cd * Cd - (C2 - R * R)

    # Compute both roots and pick the nearest positive intersection.
    # If disc<0 for some rays (shouldn't happen once camera is inside), fall back to direction-only mapping.
    ok = disc > 1e-9
    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[ok] = np.sqrt(disc[ok])

    t1 = -Cd - sqrt_disc
    t2 = -Cd + sqrt_disc

    # choose smallest positive
    t = np.where((t1 > 1e-6), t1, np.where((t2 > 1e-6), t2, 1.0))
    Q = C + t[:, None] * d  # (N,3)

    # For rays without a valid intersection, use direction as point on sphere (no-parallax fallback).
    if not ok.all():
        Q[~ok] = (d[~ok] * R)

    # Map sphere point to equirectangular UV
    qx, qy, qz = Q[:, 0], Q[:, 1], Q[:, 2]
    lon = np.arctan2(qx, qz)  # [-pi, pi]
    lat = np.arcsin(np.clip(qy / (np.linalg.norm(Q, axis=1) + 1e-9), -1.0, 1.0))  # [-pi/2, pi/2]
    u = (lon / (2.0 * np.pi) + 0.5) * (pano_w - 1)
    v = (0.5 - lat / np.pi) * (pano_h - 1)

    map_x = u.reshape(out_size, out_size).astype(np.float32)
    map_y = v.reshape(out_size, out_size).astype(np.float32)

    tex = cv2.remap(pano_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # If the result is almost black (can happen if math goes wrong or pano is weird),
    # fall back to a simple direction-only mapping (no parallax) so something is visible.
    try:
        if float(tex.mean()) < 1.0:
            # Use rays from origin through portal points.
            d0 = P
            dn0 = np.linalg.norm(d0, axis=1, keepdims=True)
            d0 = d0 / (dn0 + 1e-9)
            Q0 = d0 * R
            qx0, qy0, qz0 = Q0[:, 0], Q0[:, 1], Q0[:, 2]
            lon0 = np.arctan2(qx0, qz0)
            lat0 = np.arcsin(np.clip(qy0 / (np.linalg.norm(Q0, axis=1) + 1e-9), -1.0, 1.0))
            u0 = (lon0 / (2.0 * np.pi) + 0.5) * (pano_w - 1)
            v0 = (0.5 - lat0 / np.pi) * (pano_h - 1)
            map_x0 = u0.reshape(out_size, out_size).astype(np.float32)
            map_y0 = v0.reshape(out_size, out_size).astype(np.float32)
            tex = cv2.remap(pano_bgr, map_x0, map_y0, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    except Exception:
        pass

    return tex


def _wrap_repeat01(x: np.ndarray) -> np.ndarray:
    """Wrap values into [0,1) for texture repeat mapping."""
    return x - np.floor(x)


def _sample_texture_repeat(tex_bgr: np.ndarray, u01: np.ndarray, v01: np.ndarray, *, repeat: float = 1.0) -> np.ndarray:
    """
    Sample a BGR texture using normalized UV in [0,1] with optional repeat.
    Uses bilinear sampling via cv2.remap.
    """
    h, w = tex_bgr.shape[:2]
    if h < 2 or w < 2:
        return np.zeros((u01.shape[0], u01.shape[1], 3), dtype=np.uint8)

    rep = float(repeat)
    if rep <= 0.0:
        rep = 1.0

    uu = _wrap_repeat01(u01 * rep)
    vv = _wrap_repeat01(v01 * rep)

    map_x = (uu * (w - 1)).astype(np.float32)
    map_y = (vv * (h - 1)).astype(np.float32)

    return cv2.remap(tex_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _render_portal_room_box(
    wall_tex_bgr: np.ndarray,
    C_obj: np.ndarray,
    *,
    portal_w: float,
    portal_h: float,
    out_size: int,
    depth: float,
    room_scale: float,
    cam_backoff: float,
    tex_repeat: float = 1.0,
) -> np.ndarray:
    """
    Render a REAL 3D "room" behind the portal plane (Z=0) using ray-box intersections.

    World (object) coords:
      - Portal plane is Z=0.
      - The room extends behind the portal towards negative Z.
      - The room is a rectangular box:
          x in [-W/2, W/2], y in [-H/2, H/2], z in [-D, 0]

    Rays:
      For each portal pixel we pick a point P on the portal plane and cast a ray
      from a virtual camera behind the portal (C_virt) through P into the room.
    """
    out_size = int(out_size)
    if out_size <= 32:
        out_size = 32

    D = float(depth)
    if D <= 1e-6:
        D = 1.0

    s = float(room_scale)
    if s <= 1e-3:
        s = 1.0

    # Room dimensions (slightly larger than the portal)
    W = float(portal_w) * s
    H = float(portal_h) * s
    hw = 0.5 * W
    hh = 0.5 * H

    # Virtual camera behind the portal:
    # - reflect camera across plane (z -> -z)
    # - then push it further behind by cam_backoff
    C = np.asarray(C_obj, dtype=np.float64).reshape(3)
    Cz = -float(C[2])
    Cz -= float(cam_backoff)
    Cvirt = np.array([float(C[0]), float(C[1]), Cz], dtype=np.float64)

    # Grid of points on portal plane (Z=0), centered.
    if out_size not in _PORTAL_GRID_CACHE:
        xsn = np.linspace(-0.5, 0.5, out_size, dtype=np.float32)
        ysn = np.linspace(-0.5, 0.5, out_size, dtype=np.float32)
        Xn, Yn = np.meshgrid(xsn, ysn)
        _PORTAL_GRID_CACHE[out_size] = (Xn, Yn)
    Xn, Yn = _PORTAL_GRID_CACHE[out_size]
    X = (Xn * float(portal_w)).astype(np.float64, copy=False)
    Y = (Yn * float(portal_h)).astype(np.float64, copy=False)
    P = np.stack([X, Y, np.zeros_like(X)], axis=-1)  # (H,W,3)

    # Ray directions d = normalize(P - Cvirt)
    d = P - Cvirt.reshape(1, 1, 3)
    dn = np.linalg.norm(d, axis=2, keepdims=True)
    d = d / (dn + 1e-9)

    # Helper to compute intersection t for plane axis=value
    def t_for_plane(axis: int, value: float) -> np.ndarray:
        denom = d[:, :, axis]
        numer = (value - Cvirt[axis])
        t = numer / (denom + 1e-9)
        return t

    # Candidate intersections with 6 planes
    t_back = t_for_plane(2, -D)      # z = -D
    t_front = t_for_plane(2, 0.0)    # z = 0 (should be ~ on portal plane)
    t_left = t_for_plane(0, -hw)     # x = -W/2
    t_right = t_for_plane(0, hw)     # x = +W/2
    t_bottom = t_for_plane(1, -hh)   # y = -H/2
    t_top = t_for_plane(1, hh)       # y = +H/2

    # Compute hit points for each plane
    def hit_point(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hx = Cvirt[0] + t * d[:, :, 0]
        hy = Cvirt[1] + t * d[:, :, 1]
        hz = Cvirt[2] + t * d[:, :, 2]
        return hx, hy, hz

    bx, by, bz = hit_point(t_back)
    lx, ly, lz = hit_point(t_left)
    rx, ry, rz = hit_point(t_right)
    tx, ty, tz = hit_point(t_top)
    fx, fy, fz = hit_point(t_bottom)

    # Validity masks (hit point within the face bounds) and t>0
    eps = 1e-6
    valid_back = (t_back > eps) & (np.abs(bx) <= hw + 1e-6) & (np.abs(by) <= hh + 1e-6)
    valid_left = (t_left > eps) & (lz <= 0.0 + 1e-6) & (lz >= -D - 1e-6) & (np.abs(ly) <= hh + 1e-6)
    valid_right = (t_right > eps) & (rz <= 0.0 + 1e-6) & (rz >= -D - 1e-6) & (np.abs(ry) <= hh + 1e-6)
    valid_top = (t_top > eps) & (tz <= 0.0 + 1e-6) & (tz >= -D - 1e-6) & (np.abs(tx) <= hw + 1e-6)
    valid_bottom = (t_bottom > eps) & (fz <= 0.0 + 1e-6) & (fz >= -D - 1e-6) & (np.abs(fx) <= hw + 1e-6)

    # Pick nearest valid intersection among {back,left,right,top,bottom}
    t_inf = np.full((out_size, out_size), 1e12, dtype=np.float64)
    t_best = t_inf.copy()
    face = np.full((out_size, out_size), -1, dtype=np.int8)  # 0 back, 1 left, 2 right, 3 top, 4 bottom

    def consider(t: np.ndarray, valid: np.ndarray, face_id: int):
        nonlocal t_best, face
        t_use = np.where(valid, t, t_inf)
        sel = t_use < t_best
        t_best = np.where(sel, t_use, t_best)
        face = np.where(sel, np.int8(face_id), face)

    consider(t_back, valid_back, 0)
    consider(t_left, valid_left, 1)
    consider(t_right, valid_right, 2)
    consider(t_top, valid_top, 3)
    consider(t_bottom, valid_bottom, 4)

    # Compute final hit point
    hx, hy, hz = hit_point(t_best)

    # UV mapping per face into the wall texture (normalized 0..1)
    u = np.zeros((out_size, out_size), dtype=np.float32)
    v = np.zeros((out_size, out_size), dtype=np.float32)

    # back face: u=x, v=y
    sel = face == 0
    if np.any(sel):
        u[sel] = ((hx[sel] + hw) / (W + 1e-9)).astype(np.float32)
        v[sel] = ((hy[sel] + hh) / (H + 1e-9)).astype(np.float32)

    # left face (x=-hw): u=z depth, v=y
    sel = face == 1
    if np.any(sel):
        u[sel] = ((-hz[sel]) / (D + 1e-9)).astype(np.float32)
        v[sel] = ((hy[sel] + hh) / (H + 1e-9)).astype(np.float32)

    # right face (x=+hw)
    sel = face == 2
    if np.any(sel):
        u[sel] = ((-hz[sel]) / (D + 1e-9)).astype(np.float32)
        v[sel] = ((hy[sel] + hh) / (H + 1e-9)).astype(np.float32)

    # top face (y=+hh): u=x, v=z
    sel = face == 3
    if np.any(sel):
        u[sel] = ((hx[sel] + hw) / (W + 1e-9)).astype(np.float32)
        v[sel] = ((-hz[sel]) / (D + 1e-9)).astype(np.float32)

    # bottom face (y=-hh)
    sel = face == 4
    if np.any(sel):
        u[sel] = ((hx[sel] + hw) / (W + 1e-9)).astype(np.float32)
        v[sel] = ((-hz[sel]) / (D + 1e-9)).astype(np.float32)

    # For pixels with no valid hit, return black
    ok = face >= 0
    if not ok.any():
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    # Clamp UV to [0,1] then sample with optional repeat
    u01 = np.clip(u, 0.0, 1.0)
    v01 = np.clip(v, 0.0, 1.0)
    tex = _sample_texture_repeat(wall_tex_bgr, u01, v01, repeat=float(tex_repeat))

    # Zero out invalid pixels
    tex[~ok] = 0
    return tex


def _make_centered_rect_points_z(w: float, h: float, z: float) -> np.ndarray:
    """4 corners of a centered rectangle on plane z=constant (object coords)."""
    hw = 0.5 * float(w)
    hh = 0.5 * float(h)
    zz = float(z)
    return np.array(
        [
            [-hw, -hh, zz],
            [hw, -hh, zz],
            [hw, hh, zz],
            [-hw, hh, zz],
        ],
        dtype=np.float64,
    )


def _render_portal_backwall_texture(
    frame_bgr: np.ndarray,
    texture_bgr: np.ndarray,
    *,
    portal_ellipse2d: np.ndarray,
    backwall_quad2d: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Render a textured back wall quad and clip it by an ellipse portal mask.
    Returns (out_frame, ellipse_mask_u8).
    """
    # Ellipse mask in frame coords (fill polygon approximating ellipse curve)
    ell2d = np.asarray(portal_ellipse2d, dtype=np.float64).reshape(-1, 2)
    if ell2d.shape[0] < 12 or (not np.isfinite(ell2d).all()):
        return frame_bgr, np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

    ell_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(ell_mask, [np.int32(ell2d).reshape(-1, 1, 2)], 255, lineType=cv2.LINE_AA)

    quad = np.asarray(backwall_quad2d, dtype=np.float64).reshape(4, 2)
    if quad.shape != (4, 2) or (not np.isfinite(quad).all()):
        return frame_bgr, ell_mask

    # Limit blending to ellipse AND backwall quad area
    quad_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(quad_mask, np.int32(quad), 255, lineType=cv2.LINE_AA)
    frame_mask = cv2.bitwise_and(ell_mask, quad_mask)

    out = _warp_texture_onto_quad_with_frame_mask(frame_bgr, texture_bgr, quad, frame_mask, alpha=float(alpha))
    return out, ell_mask


def _draw_plane_outline_corners_stripes(
    img_bgr: np.ndarray,
    corners2d: np.ndarray,
    *,
    outline_color_bgr: tuple[int, int, int] = (0, 255, 0),
    outline_thickness: int = 2,
    corner_color_bgr: tuple[int, int, int] = (0, 0, 255),
    corner_radius: int = 5,
    stripes_color_bgr: tuple[int, int, int] = (255, 255, 255),
    stripes_thickness: int = 1,
    stripes_count: int = 6,
) -> np.ndarray:
    """
    Neutral visualization for the tracked plane:
    - green outline
    - red corners
    - white stripes (projective grid cue)

    corners2d order is assumed to be:
      0:(-,-), 1:(+,-), 2:(+,+), 3:(-,+)
    (this matches _make_centered_plane_object_points / projectPoints ordering).
    """
    out = img_bgr.copy()
    c = np.asarray(corners2d, dtype=np.float64).reshape(4, 2)
    if c.shape != (4, 2) or (not np.isfinite(c).all()):
        return out

    # Outline
    cv2.polylines(out, [np.int32(c).reshape(-1, 1, 2)], True, outline_color_bgr, int(outline_thickness), cv2.LINE_AA)

    # Corners
    for i in range(4):
        cv2.circle(out, (int(c[i, 0]), int(c[i, 1])), int(corner_radius), corner_color_bgr, -1, cv2.LINE_AA)

    # Optional stripes: connect bottom edge (0-1) to top edge (3-2)
    n = int(stripes_count)
    if n > 0:
        for k in range(1, n + 1):
            t = k / float(n + 1)
            p0 = _lerp2(c[0], c[1], t)
            p1 = _lerp2(c[3], c[2], t)
            cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), stripes_color_bgr, int(stripes_thickness), cv2.LINE_AA)
    return out


@dataclass
class PlanePose:
    rvec: np.ndarray
    tvec: np.ndarray
    visible: bool
    held: bool = False


class PlaneTracker:
    """
    Per-plane tracker state:
    - Reference keypoints/descriptors
    - Last pose (rvec/tvec) + graceful loss (hold frames)
    """

    def __init__(self, name: str, reference_bgr: np.ndarray, cfg: Part5Config):
        self.name = str(name)
        self.cfg = cfg

        # Feature detector + matcher per plane (data isolated).
        feat = str(cfg.feature_type or "SIFT").strip().upper()
        if feat == "SIFT":
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError("SIFT is not available. Install opencv-contrib-python or use feature_type='ORB'.")
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif feat == "ORB":
            self.detector = cv2.ORB_create(nfeatures=1500)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Unknown feature_type: {cfg.feature_type} (expected 'SIFT' or 'ORB')")

        ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        self.kp_ref, self.des_ref = self.detector.detectAndCompute(ref_gray, None)
        if self.des_ref is None or self.kp_ref is None or len(self.kp_ref) < 20:
            raise ValueError(f"[{self.name}] reference has too few features. Pick a more feature-rich image.")

        h, w = ref_gray.shape[:2]
        self.ref_corners_px = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)

        # Plane size in world units, centered coordinate system.
        ref_aspect = h / float(w)
        self.plane_w = float(cfg.plane_width)
        self.plane_h = float(cfg.plane_width) * float(ref_aspect)
        self.obj_plane = np.ascontiguousarray(_make_centered_plane_object_points(self.plane_w, self.plane_h), dtype=np.float64).reshape(-1, 3)

        # Portal geometry (in plane local coords, centered at origin).
        pw = float(cfg.portal_size_frac) * self.plane_w
        ph = float(cfg.portal_size_frac) * self.plane_h
        self.obj_portal = np.ascontiguousarray(_make_centered_plane_object_points(pw, ph), dtype=np.float64).reshape(-1, 3)
        self.portal_w = float(pw)
        self.portal_h = float(ph)
        # Precompute portal ellipse points in portal-local coords (used when portal_shape == "ellipse").
        # This avoids sin/cos per frame.
        ts = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False, dtype=np.float64)
        self._portal_ellipse_pts3 = np.stack(
            [(self.portal_w * 0.5) * np.cos(ts), (self.portal_h * 0.5) * np.sin(ts), np.zeros_like(ts)],
            axis=1,
        ).astype(np.float64)

        # Temporal state (graceful tracking loss)
        self._rvec: Optional[np.ndarray] = None
        self._tvec: Optional[np.ndarray] = None
        self._hold_left: int = 0
        self._last_corners_img_raw: Optional[np.ndarray] = None   # (4,2) float, from homography (unsmoothed)
        self._last_corners_img_draw: Optional[np.ndarray] = None  # (4,2) float, smoothed for drawing
        self._stable_count: int = 0
        self._was_visible: bool = False

        # Debug stats
        self.good = 0
        self.inliers = 0

    def update_from_frame_features(
        self,
        kp_fr,
        des_fr: Optional[np.ndarray],
        K: np.ndarray,
        dist: np.ndarray,
    ) -> PlanePose:
        """
        Update pose from already-computed frame features (kp_fr/des_fr).
        Returns PlanePose with visible flag and held flag.
        """
        self.good = 0
        self.inliers = 0

        if des_fr is None or kp_fr is None:
            return self._maybe_hold()

        good = _knn_ratio_test(self.matcher, self.des_ref, des_fr, k=2, ratio=float(self.cfg.ratio_test))
        self.good = len(good)
        if self.good < int(self.cfg.min_matches):
            return self._maybe_hold()

        src = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_fr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        method = _get_homography_method(getattr(self.cfg, "homography_method", "RANSAC"))
        H, inlier_mask = cv2.findHomography(src, dst, method, float(self.cfg.ransac_reproj_thresh))
        if H is None or inlier_mask is None:
            return self._maybe_hold()

        self.inliers = int(inlier_mask.sum())
        if self.inliers < int(getattr(self.cfg, "min_inliers", 10)):
            return self._maybe_hold()

        # Optional refinement: recompute least-squares H using inliers only.
        if bool(getattr(self.cfg, "refine_homography_with_inliers", True)):
            try:
                inl = inlier_mask.ravel().astype(bool)
                if int(inl.sum()) >= 4:
                    H2, _ = cv2.findHomography(src[inl], dst[inl], 0)
                    if H2 is not None:
                        H = H2
            except Exception:
                pass

        corners = cv2.perspectiveTransform(self.ref_corners_px, H)
        img_pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if img_pts.shape != (4, 2) or not np.isfinite(img_pts).all():
            return self._maybe_hold()

        # Store raw homography corners, and also a smoothed version for drawing (reduces jitter).
        self._last_corners_img_raw = img_pts.astype(np.float64)
        a_c = float(getattr(self.cfg, "outline_smoothing_alpha", 0.0))
        if self._last_corners_img_draw is None or not (0.0 < a_c < 1.0):
            self._last_corners_img_draw = self._last_corners_img_raw.copy()
        else:
            prev = np.asarray(self._last_corners_img_draw, dtype=np.float64).reshape(4, 2)
            curr = np.asarray(self._last_corners_img_raw, dtype=np.float64).reshape(4, 2)
            self._last_corners_img_draw = (a_c * prev + (1.0 - a_c) * curr).astype(np.float64)

        # Pose (PnP):
        # For planar targets, IPPE can yield two valid poses (normal in either direction).
        # We deterministically pick the one whose plane normal faces the camera.
        ok_pnp, rvec, tvec = solve_planar_pnp_facing_camera(self.obj_plane, img_pts, K, dist, prefer_ippe=True)

        if not ok_pnp:
            return self._maybe_hold()

        r_new = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        t_new = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if not (np.isfinite(r_new).all() and np.isfinite(t_new).all()):
            # Guard against rare numerical issues; don't poison the temporal state with NaNs.
            return self._maybe_hold()

        # Optional outlier rejection: if pose "pops" too much in one frame, ignore it.
        if bool(getattr(self.cfg, "reject_pose_jumps", False)) and (self._rvec is not None) and (self._tvec is not None):
            try:
                # Translation jump (world units)
                dt = float(np.linalg.norm(t_new.reshape(3) - np.asarray(self._tvec, dtype=np.float64).reshape(3)))
                max_dt = float(getattr(self.cfg, "pose_jump_max_trans", 0.0) or 0.0)

                # Rotation jump (degrees) using relative rotation
                R_prev, _ = cv2.Rodrigues(np.asarray(self._rvec, dtype=np.float64).reshape(3, 1))
                R_curr, _ = cv2.Rodrigues(r_new)
                R_rel = R_prev.T @ R_curr
                tr = float(np.trace(R_rel))
                cosang = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
                ang_deg = float(np.degrees(np.arccos(cosang)))
                max_ang = float(getattr(self.cfg, "pose_jump_max_rot_deg", 0.0) or 0.0)

                if (max_dt > 0.0 and dt > max_dt) or (max_ang > 0.0 and ang_deg > max_ang):
                    return self._maybe_hold()
            except Exception:
                pass

        # Pose smoothing (reduces portal jitter). We smooth in axis-angle space for simplicity.
        a_p = float(getattr(self.cfg, "pose_smoothing_alpha", 0.0))
        if self._rvec is None or self._tvec is None or not (0.0 < a_p < 1.0):
            self._rvec = r_new
            self._tvec = t_new
        else:
            self._rvec = (a_p * self._rvec + (1.0 - a_p) * r_new).astype(np.float64)
            self._tvec = (a_p * self._tvec + (1.0 - a_p) * t_new).astype(np.float64)

        self._hold_left = int(getattr(self.cfg, "max_hold_frames", 0))

        # Visibility gating: require N consecutive good frames before showing.
        min_vis = int(getattr(self.cfg, "min_visible_frames", 1))
        if min_vis < 1:
            min_vis = 1
        self._stable_count = min(self._stable_count + 1, 1000000)
        visible_now = self._stable_count >= min_vis
        self._was_visible = bool(visible_now)
        return PlanePose(rvec=self._rvec, tvec=self._tvec, visible=visible_now, held=False)

    def _maybe_hold(self) -> PlanePose:
        max_hold = int(getattr(self.cfg, "max_hold_frames", 0))
        if (
            max_hold > 0
            and self._rvec is not None
            and self._tvec is not None
            and self._hold_left > 0
        ):
            self._hold_left -= 1
            # Keep visibility only if it was already stable/visible before.
            return PlanePose(rvec=self._rvec, tvec=self._tvec, visible=bool(self._was_visible), held=True)

        # Hard reset on failure (prevents flicker/jumps when target leaves view)
        self._stable_count = 0
        self._was_visible = False
        return PlanePose(rvec=np.zeros((3, 1), dtype=np.float64), tvec=np.zeros((3, 1), dtype=np.float64), visible=False, held=False)

    def last_corners_img(self) -> Optional[np.ndarray]:
        """
        Last 2D quad from homography (4x2) in image pixels.
        This is typically the most stable way to draw an outline/stripes overlay.
        """
        if self._last_corners_img_draw is None:
            return None
        c = np.asarray(self._last_corners_img_draw, dtype=np.float64).reshape(4, 2)
        if not np.isfinite(c).all():
            return None
        return c

    def project_portal(self, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """Project portal corners (4x3) to image pixels (4x2)."""
        pts_cam, _R, _t = _object_to_camera_points(self.obj_portal, rvec, tvec)
        return _project_points_camera_frame(pts_cam, K, dist)

    def project_portal_ellipse(self, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """Project a portal ellipse curve (Nx3) to image pixels (Nx2)."""
        pts_cam, _R, _t = _object_to_camera_points(self._portal_ellipse_pts3, rvec, tvec)
        pts = _project_points_camera_frame(pts_cam, K, dist)
        # Leave validation to caller, but keep shape consistent.
        return pts

    def project_plane_outline(self, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
        pts_cam, _R, _t = _object_to_camera_points(self.obj_plane, rvec, tvec)
        return _project_points_camera_frame(pts_cam, K, dist)


def run_part5_multiplane(cfg: Part5Config) -> None:
    """Compatibility wrapper. Prefer calling export_part5_video() directly."""
    export_part5_video(
        cfg,
        out_path="outputs/videos/part5_portal.mp4",
        draw_portal=True,
        use_env_portals=False,
        draw_debug_text=False,
        draw_plane_outline=True,
    )


def export_part5_video(
    cfg: Part5Config,
    out_path: str,
    *,
    draw_portal: bool,
    use_env_portals: bool = False,
    draw_debug_text: bool = False,
    draw_plane_outline: bool = False,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> None:
    """
    Export helper:
    - draw_portal=False: runs tracking but writes frames without any visualization.
    - draw_portal=True: draws portal (and optional outlines/debug).
    """
    # Keep a local alias for readability; do NOT rebuild Part5Config here (it can drop newer fields).
    cfg2 = cfg

    base_dir = Path(__file__).resolve().parent

    # Load references
    refs = []
    for p in [cfg2.reference1_path, cfg2.reference2_path, cfg2.reference3_path]:
        pp = Path(p)
        if not pp.is_absolute():
            pp = base_dir / pp
        img = _imread_any(str(pp))
        if img is None:
            raise FileNotFoundError(f"Could not read reference image: {p}")
        refs.append(img)

    vp = Path(cfg2.video_path)
    if not vp.is_absolute():
        vp = base_dir / vp
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {cfg2.video_path}")

    # Optional: start from a specific frame (useful for notebook preview exports).
    sf = int(start_frame) if start_frame is not None else 0
    if sf < 0:
        sf = 0
    if sf > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    K, dist = get_camera_params_from_config((out_w, out_h), calib_output_path=cfg2.calib_output_path)

    feat = str(cfg2.feature_type or "SIFT").strip().upper()
    if feat == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is not available. Install opencv-contrib-python or set feature_type='ORB'.")
        frame_detector = cv2.SIFT_create()
    else:
        frame_detector = cv2.ORB_create(nfeatures=2000)

    planes = [
        PlaneTracker("P1", refs[0], cfg2),
        PlaneTracker("P2", refs[1], cfg2),
        PlaneTracker("P3", refs[2], cfg2),
    ]

    # Optional 360° environments (portal360)
    envs = [None, None, None]
    if bool(use_env_portals):
        env_paths = [cfg2.env1_360_path, cfg2.env2_360_path, cfg2.env3_360_path]
        for j, p in enumerate(env_paths):
            pp = Path(p)
            if not pp.is_absolute():
                pp = base_dir / pp
            img = _imread_any(str(pp))
            envs[j] = img

        if bool(draw_debug_text):
            print(
                "[part5 portal360] env loaded:",
                [(j + 1, (envs[j] is not None), (None if envs[j] is None else tuple(envs[j].shape))) for j in range(3)],
            )

    outp = Path(out_path)
    if not outp.is_absolute():
        outp = base_dir / outp
    outp.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    frame_i = 0
    last_vis = 0
    max_n = None if max_frames is None else int(max_frames)
    if max_n is not None and max_n <= 0:
        max_n = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_fr, des_fr = frame_detector.detectAndCompute(frame_gray, None)

        # Always run tracking (so runtime is representative), but optionally skip drawing.
        if not draw_portal and not draw_plane_outline and not draw_debug_text:
            for plane in planes:
                _ = plane.update_from_frame_features(kp_fr, des_fr, K, dist)
            writer.write(frame)
            frame_i += 1
            if frame_i % 60 == 0:
                print(f"[part5 export] frames={frame_i}")
            continue

        # Outline-only mode: draw tracked plane geometry (no portal fill)
        if (not draw_portal) and bool(draw_plane_outline):
            out = frame
            vis_count = 0
            for plane in planes:
                pose = plane.update_from_frame_features(kp_fr, des_fr, K, dist)
                if not pose.visible:
                    continue
                vis_count += 1
                # Prefer homography-based corners (matches the notebook and is K/dist independent).
                outline2d = plane.last_corners_img()
                if outline2d is None:
                    outline2d = plane.project_plane_outline(pose.rvec, pose.tvec, K, dist)
                out = _draw_plane_outline_corners_stripes(
                    out,
                    outline2d,
                    outline_color_bgr=getattr(cfg2, "outline_color_bgr", (0, 255, 0)),
                    outline_thickness=int(getattr(cfg2, "outline_thickness", 2)),
                    corner_color_bgr=getattr(cfg2, "corner_color_bgr", (0, 0, 255)),
                    corner_radius=int(getattr(cfg2, "corner_radius", 5)),
                    stripes_color_bgr=getattr(cfg2, "stripes_color_bgr", (255, 255, 255)),
                    stripes_thickness=int(getattr(cfg2, "stripes_thickness", 1)),
                    stripes_count=int(getattr(cfg2, "stripes_count", 6)),
                )

            if bool(draw_debug_text):
                cv2.putText(out, f"feat={cfg2.feature_type} frame={frame_i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            writer.write(out)
            frame_i += 1
            last_vis = int(vis_count)
            if frame_i % 60 == 0:
                print(f"[part5 export] frames={frame_i} visible={last_vis}/3 mode=outline")
            continue

        out = frame
        vis_count = 0
        for plane in planes:
            pose = plane.update_from_frame_features(kp_fr, des_fr, K, dist)
            if not pose.visible:
                continue
            vis_count += 1
            portal2d = plane.project_portal(pose.rvec, pose.tvec, K, dist)

            # Portal fill: if env images exist, render portal360 texture; else fallback to solid fill.
            pano = None
            if bool(use_env_portals):
                pano = envs[0] if plane.name == "P1" else envs[1] if plane.name == "P2" else envs[2]

            portal_shape = str(getattr(cfg2, "portal_shape", "rect") or "rect").strip().lower()
            if portal_shape not in ("rect", "ellipse"):
                portal_shape = "rect"

            # Debug-first option: solid back wall (no texture). Recommended to validate parallax direction.
            # We support it both with and without env panoramas.
            if bool(getattr(cfg2, "portal_debug_backwall_solid", False)):
                # Debug stage: always use ellipse portal for clean parallax sanity checks.
                ell2d = plane.project_portal_ellipse(pose.rvec, pose.tvec, K, dist)
                if ell2d.shape[0] < 12 or (not np.isfinite(ell2d).all()):
                    continue

                bw = float(getattr(cfg2, "portal_backwall_size_frac", 0.85)) * float(plane.portal_w)
                bh = float(getattr(cfg2, "portal_backwall_size_frac", 0.85)) * float(plane.portal_h)
                depth = abs(float(getattr(cfg2, "portal_backwall_depth", 0.25)))

                obj_back0 = _make_centered_rect_points_z(bw, bh, z=0.0)
                back0_cam, R, t_cam = _object_to_camera_points(obj_back0, pose.rvec, pose.tvec)
                n_cam = _plane_normal_cam_facing_camera(R, t_cam)
                back_cam = back0_cam + (-n_cam.reshape(1, 3)) * depth
                back2d = _project_points_camera_frame(back_cam, K, dist)

                portal_mask = np.zeros(out.shape[:2], dtype=np.uint8)
                cv2.fillPoly(portal_mask, [np.int32(ell2d).reshape(-1, 1, 2)], 255, lineType=cv2.LINE_AA)
                quad = np.asarray(back2d, dtype=np.float64).reshape(4, 2)
                quad_mask = np.zeros(out.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(quad_mask, np.int32(quad), 255, lineType=cv2.LINE_AA)
                frame_mask = cv2.bitwise_and(portal_mask, quad_mask)

                a = float(getattr(cfg2, "portal_backwall_alpha", cfg2.portal_alpha))
                if a > 0.0:
                    col = np.array(getattr(cfg2, "portal_debug_backwall_bgr", (40, 40, 40)), dtype=np.float32).reshape(1, 1, 3)
                    out_f = out.astype(np.float32)
                    idx = frame_mask > 0
                    out_f[idx] = a * col + (1.0 - a) * out_f[idx]
                    out = out_f.astype(np.uint8)

                out = _apply_portal_glass_effect(out, portal_mask, cfg2)
                out = _draw_portal_window_rim(out, ell2d, cfg2)
            elif pano is not None:
                # Best-practice back-wall portal:
                # - enforce normal facing camera (camera frame)
                # - build back wall by shifting geometry by (-normal * depth) in camera frame
                # - no manual parallax hacks; projection alone produces parallax
                if portal_shape != "ellipse":
                    portal_shape = "ellipse"

                ell2d = plane.project_portal_ellipse(pose.rvec, pose.tvec, K, dist)
                if ell2d.shape[0] < 12 or (not np.isfinite(ell2d).all()):
                    continue

                # Build back-wall quad in camera frame (shift along -normal_cam).
                bw = float(getattr(cfg2, "portal_backwall_size_frac", 0.85)) * float(plane.portal_w)
                bh = float(getattr(cfg2, "portal_backwall_size_frac", 0.85)) * float(plane.portal_h)
                depth = float(getattr(cfg2, "portal_backwall_depth", 0.25))
                depth = abs(float(depth))

                obj_back0 = _make_centered_rect_points_z(bw, bh, z=0.0)  # plane-local, z=0 baseline
                back0_cam, R, t_cam = _object_to_camera_points(obj_back0, pose.rvec, pose.tvec)
                n_cam = _plane_normal_cam_facing_camera(R, t_cam)
                back_cam = back0_cam + (-n_cam.reshape(1, 3)) * depth
                back2d = _project_points_camera_frame(back_cam, K, dist)

                # Portal mask (ellipse), used for both debug solid and textured back wall.
                portal_mask = np.zeros(out.shape[:2], dtype=np.uint8)
                cv2.fillPoly(portal_mask, [np.int32(ell2d).reshape(-1, 1, 2)], 255, lineType=cv2.LINE_AA)

                quad = np.asarray(back2d, dtype=np.float64).reshape(4, 2)
                quad_mask = np.zeros(out.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(quad_mask, np.int32(quad), 255, lineType=cv2.LINE_AA)
                frame_mask = cv2.bitwise_and(portal_mask, quad_mask)

                # Texture only AFTER debug stage works.
                k = int(getattr(cfg2, "portal_env_blur_ksize", 0))
                tex = pano
                if k and k > 1:
                    kk = int(k)
                    if kk % 2 == 0:
                        kk += 1
                    tex = cv2.GaussianBlur(tex, (kk, kk), 0.0)
                sharp = float(getattr(cfg2, "portal_env_sharpen_amount", 0.0))
                if sharp > 0.0:
                    blur = cv2.GaussianBlur(tex, (0, 0), 1.0)
                    tex = np.clip((1.0 + sharp) * tex.astype(np.float32) - sharp * blur.astype(np.float32), 0, 255).astype(np.uint8)

                out = _warp_texture_onto_quad_with_frame_mask(
                    out,
                    tex,
                    quad,
                    frame_mask,
                    alpha=float(getattr(cfg2, "portal_backwall_alpha", cfg2.portal_alpha)),
                )
                out = _apply_portal_glass_effect(out, portal_mask, cfg2)
                out = _draw_portal_window_rim(out, ell2d, cfg2)
            else:
                if portal_shape == "ellipse":
                    ell2d = plane.project_portal_ellipse(pose.rvec, pose.tvec, K, dist)
                    if ell2d.shape[0] < 12 or (not np.isfinite(ell2d).all()):
                        continue
                    mask = np.zeros(out.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [np.int32(ell2d).reshape(-1, 1, 2)], 255, lineType=cv2.LINE_AA)
                    # Fill with constant color only inside ellipse
                    a = float(cfg2.portal_alpha)
                    if a > 0:
                        idx = mask > 0
                        col = np.array(cfg2.portal_fill_bgr, dtype=np.float32).reshape(1, 1, 3)
                        out_f = out.astype(np.float32)
                        out_f[idx] = a * col + (1.0 - a) * out_f[idx]
                        out = out_f.astype(np.uint8)
                    out = _apply_portal_glass_effect(out, mask, cfg2)
                    out = _draw_portal_window_rim(out, ell2d, cfg2)
                else:
                    out = _alpha_fill_convex_poly(out, portal2d, cfg2.portal_fill_bgr, alpha=float(cfg2.portal_alpha))
                    mask = _portal_mask_from_poly(out.shape[:2], portal2d)
                    out = _apply_portal_glass_effect(out, mask, cfg2)
                    out = _draw_portal_window_rim(out, portal2d, cfg2)
                if bool(use_env_portals):
                    # Helpful visual cue: env missing
                    c = portal2d.mean(axis=0)
                    cv2.putText(out, "NO ENV", (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(out, "NO ENV", (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            if bool(draw_plane_outline):
                # Prefer homography quad as the "detection outline" (matches tracking result)
                outline2d = plane.last_corners_img()
                if outline2d is None:
                    outline2d = plane.project_plane_outline(pose.rvec, pose.tvec, K, dist)
                cv2.polylines(out, [np.int32(outline2d).reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)

        if bool(draw_debug_text):
            cv2.putText(out, f"visible={vis_count}/3 feat={cfg2.feature_type} frame={frame_i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        writer.write(out)
        frame_i += 1
        last_vis = int(vis_count)
        if max_n is not None and frame_i >= max_n:
            break
        if frame_i % 60 == 0:
            print(f"[part5 export] frames={frame_i} visible={last_vis}/3 mode=portal")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Saved:", str(outp))


if __name__ == "__main__":
    run_part5_multiplane(Part5Config())


