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


def load_mesh_trimesh(path: str):
    """
    Load a mesh from .obj/.ply using trimesh.
    Returns: (vertices Nx3 float32, faces Mx3 int32)
    """
    try:
        import trimesh
    except Exception as e:
        raise RuntimeError("trimesh is required for Part 3. Install it with: pip install trimesh") from e

    loaded = trimesh.load(path)
    if loaded is None:
        raise ValueError(f"Failed to load mesh from: {path}")

    # If it's a scene with multiple geometries, merge them (colors may be approximated later)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if hasattr(g, "vertices") and hasattr(g, "faces")]
        if not geoms:
            raise ValueError(f"No mesh geometries found in: {path}")

        # Convert each geometry to per-face colors first (preserves materials/textures),
        # then concatenate manually while keeping face_colors aligned.
        v_all = []
        f_all = []
        c_all = []
        v_offset = 0

        for g in geoms:
            gg = g.copy()
            try:
                if hasattr(gg, "visual") and hasattr(gg.visual, "to_color"):
                    gg.visual = gg.visual.to_color()
            except Exception:
                pass

            v = np.asarray(gg.vertices, dtype=np.float32)
            f = np.asarray(gg.faces, dtype=np.int32)
            if v.size == 0 or f.size == 0:
                continue

            fc = None
            try:
                if hasattr(gg, "visual") and hasattr(gg.visual, "face_colors"):
                    fc = np.asarray(gg.visual.face_colors)
                    if fc.ndim == 2 and fc.shape[0] == f.shape[0] and fc.shape[1] in (3, 4):
                        if fc.shape[1] == 3:
                            alpha = np.full((fc.shape[0], 1), 255, dtype=fc.dtype)
                            fc = np.concatenate([fc, alpha], axis=1)
                        fc = fc.astype(np.uint8)
                    else:
                        fc = None
            except Exception:
                fc = None

            v_all.append(v)
            f_all.append(f + v_offset)
            if fc is None:
                # placeholder: will be filled with black; caller can wireframe instead if desired
                c_all.append(np.zeros((f.shape[0], 4), dtype=np.uint8))
            else:
                c_all.append(fc)

            v_offset += v.shape[0]

        if not v_all:
            raise ValueError(f"No valid mesh triangles found in: {path}")

        v = np.vstack(v_all)
        f = np.vstack(f_all)
        face_colors = np.vstack(c_all)
        return v, f, face_colors
    else:
        mesh = loaded

    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise ValueError(f"Failed to load mesh from: {path}")

    # Try to convert textures -> per-face colors (flat shading).
    face_colors = None
    try:
        if hasattr(mesh, "visual") and hasattr(mesh.visual, "to_color"):
            mesh.visual = mesh.visual.to_color()
        if hasattr(mesh, "visual") and hasattr(mesh.visual, "face_colors"):
            fc = np.asarray(mesh.visual.face_colors)
            if fc.ndim == 2 and fc.shape[1] in (3, 4) and len(fc) == len(mesh.faces):
                if fc.shape[1] == 3:
                    alpha = np.full((fc.shape[0], 1), 255, dtype=fc.dtype)
                    fc = np.concatenate([fc, alpha], axis=1)
                face_colors = fc.astype(np.uint8)
    except Exception:
        face_colors = None

    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"Unexpected mesh shapes: v={v.shape}, f={f.shape}")
    return v, f, face_colors


def make_demo_tetrahedron():
    """Small demo mesh (tetrahedron) so Part 3 works even without model files."""
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.2886, -0.8],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=np.int32,
    )
    return v, f


def transform_mesh_to_plane(
    vertices: np.ndarray,
    plane_w: float,
    plane_h: float,
    scale_frac: float,
    offset_x_frac: float,
    offset_y_frac: float,
    z_up: bool = True,
    rotate_x_deg: float = 0.0,
    rotate_y_deg: float = 0.0,
    rotate_z_deg: float = 0.0,
) -> np.ndarray:
    """
    Normalize mesh to fit on the plane, then scale/translate.
    - scale_frac is relative to plane_w (like the cube).
    - offset fractions place the mesh base inside the plane.
    - z_up=True means negative Z goes "up" from the plane (matches cube convention).
    """
    v = np.asarray(vertices, dtype=np.float32)

    # Center
    v0 = v - v.mean(axis=0, keepdims=True)

    # Optional rotation (degrees) around model axes before scaling/placement
    rx = np.deg2rad(float(rotate_x_deg))
    ry = np.deg2rad(float(rotate_y_deg))
    rz = np.deg2rad(float(rotate_z_deg))

    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32)

    R = (Rz @ Ry @ Rx).astype(np.float32)
    v0 = (R @ v0.T).T

    # Normalize to unit-ish size
    span = np.max(np.linalg.norm(v0, axis=1))
    if span <= 1e-8:
        span = 1.0
    v0 = v0 / span

    # Scale relative to plane width
    s = float(scale_frac) * float(plane_w)
    v0 = v0 * s

    # Place on plane: translate XY into plane, and shift Z so base touches Z=0
    min_z = float(v0[:, 2].min())
    v0[:, 2] -= min_z  # now base is at z=0 (positive up)
    if z_up:
        v0[:, 2] *= -1.0  # OpenCV cube uses negative Z as "up"

    # Position within plane bounds
    x0 = float(np.clip(float(offset_x_frac) * plane_w, 0.0, plane_w))
    y0 = float(np.clip(float(offset_y_frac) * plane_h, 0.0, plane_h))
    v0[:, 0] += x0
    v0[:, 1] += y0
    return v0


def draw_mesh_wireframe(img_bgr: np.ndarray, verts2d: np.ndarray, faces: np.ndarray, color=(0, 255, 255), thickness: int = 1):
    """
    Draw a triangle mesh as wireframe by drawing triangle edges.
    verts2d: Nx2 float
    faces: Mx3 int
    """
    img = img_bgr.copy()
    pts = np.int32(verts2d).reshape(-1, 2)
    f = np.asarray(faces, dtype=np.int32)

    for tri in f:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
        cv2.line(img, tuple(pts[b]), tuple(pts[c]), color, thickness, cv2.LINE_AA)
        cv2.line(img, tuple(pts[c]), tuple(pts[a]), color, thickness, cv2.LINE_AA)
    return img


def draw_mesh_flat(img_bgr: np.ndarray, verts2d: np.ndarray, faces: np.ndarray, face_colors_rgba: np.ndarray | None, order: np.ndarray | None = None):
    """
    Flat-shaded mesh rendering (fills triangles with per-face colors).
    If face_colors_rgba is None, falls back to a constant cyan.
    order: optional face indices order (e.g., far-to-near) for painter's algorithm.
    """
    img = img_bgr.copy()
    pts = np.int32(verts2d).reshape(-1, 2)
    f = np.asarray(faces, dtype=np.int32)

    if order is None:
        order = np.arange(f.shape[0])

    if face_colors_rgba is None:
        colors = None
    else:
        colors = np.asarray(face_colors_rgba, dtype=np.uint8)

    for idx in order:
        tri = f[int(idx)]
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        poly = np.array([pts[a], pts[b], pts[c]], dtype=np.int32)
        if colors is None:
            col = (0, 255, 255)
        else:
            r, g, b2, _a = colors[int(idx)].tolist()
            col = (int(b2), int(g), int(r))  # RGBA -> BGR
        cv2.fillConvexPoly(img, poly, col, lineType=cv2.LINE_AA)
    return img

