import os
import argparse

import cv2
import numpy as np

from config import Part1Config, Part2Config, Part3Config, Part4Config
from config import Part5Config
from tracker import PlanarTracker, warp_and_overlay
from camera import calibrate_from_chessboard_images, expand_image_glob, save_calibration_npz, load_calibration_npz
from ar_render import (
    draw_cube,
    draw_mesh_flat,
    draw_mesh_wireframe,
    load_mesh_trimesh,
    make_cube_points,
    make_demo_tetrahedron,
    make_plane_object_points,
    scale_K_to_new_size,
    transform_mesh_to_plane,
)
from multiplane import export_part5_video, run_part5_multiplane
from occlusion import raw_hsv_mask, clean_binary_mask, dilate_mask, composite_occlusion


def _read_image_or_raise(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img


def _video_props(cap: cv2.VideoCapture) -> tuple[float, int, int]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return float(fps), out_w, out_h


def _load_K_dist_for_video(calib_path: str, out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray]:
    calib = load_calibration_npz(calib_path)
    K_calib = np.asarray(calib.K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(calib.dist, dtype=np.float64).reshape(-1, 1)
    K = K_calib
    if tuple(calib.image_size) != (out_w, out_h):
        K = scale_K_to_new_size(K_calib, tuple(calib.image_size), (out_w, out_h))
    return np.asarray(K, dtype=np.float64).reshape(3, 3), dist


def _plane_w_h_from_reference(ref_bgr: np.ndarray, plane_width: float) -> tuple[float, float]:
    ref_h, ref_w = ref_bgr.shape[:2]
    ref_aspect = ref_h / float(ref_w)
    plane_w = float(plane_width)
    plane_h = float(plane_width) * ref_aspect
    return plane_w, plane_h


def _solve_pnp_ippe_fallback(
    obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    try:
        if hasattr(cv2, "SOLVEPNP_IPPE"):
            try:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=int(cv2.SOLVEPNP_IPPE))
            except cv2.error:
                ok = False
            if not ok:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=int(cv2.SOLVEPNP_ITERATIVE))
        else:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=int(cv2.SOLVEPNP_ITERATIVE))
        return bool(ok), rvec, tvec
    except cv2.error:
        return False, None, None


def run_part1(cfg: Part1Config):
    """
    Part 1 runner:
    SIFT -> ratio test -> homography (RANSAC) -> warp template -> write output video.
    """

    ref = _read_image_or_raise(cfg.reference_path)
    template = _read_image_or_raise(cfg.template_path)

    if cfg.resize_template_to_reference:
        h, w = ref.shape[:2]
        template = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {cfg.video_path}")

    fps, out_w, out_h = _video_props(cap)

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    writer = cv2.VideoWriter(cfg.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    # Build tracker (SIFT on reference)
    tracker = PlanarTracker(ref, cfg)

    # Process video frames and write output
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, corners, dbg = tracker.track(frame)
        out = frame

        if H is not None:
            out = warp_and_overlay(frame, template, H)

            if cfg.draw_corners and corners is not None:
                cv2.polylines(out, [corners.astype(int)], True, (0, 255, 0), 2)

        if cfg.draw_debug_text:
            cv2.putText(
                out,
                f"good={dbg.get('good',0)} inliers={dbg.get('inliers',0)} held={dbg.get('held',0)} sm={dbg.get('smoothed',0)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        writer.write(out)

        frame_i += 1
        if frame_i % 60 == 0:
            print(f"Processed {frame_i} frames...")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Saved:", cfg.output_path)


def run_part2_calibration(cfg: Part2Config):
    """
    Part 2 (step 1) runner:
    Calibrate camera intrinsics (K) and distortion (dist) from chessboard images,
    then save them to an .npz file for later use (solvePnP + AR rendering).
    """
    paths = expand_image_glob(cfg.calib_images_glob)
    print(f"[calib] glob: {cfg.calib_images_glob}")
    print(f"[calib] found {len(paths)} images")

    calib = calibrate_from_chessboard_images(
        paths,
        pattern_size=(cfg.chessboard_cols, cfg.chessboard_rows),
        square_size=cfg.square_size,
        show_detections=cfg.show_detections,
    )
    save_calibration_npz(cfg.calib_output_path, calib)

    print("[calib] saved:", cfg.calib_output_path)
    print("[calib] image_size (w,h):", calib.image_size)
    print("[calib] rms:", calib.rms)
    print("[calib] K:\n", calib.K)
    print("[calib] dist:", calib.dist.ravel())


def run_part2_cube(cfg: Part2Config, tracker_cfg: Part1Config):
    """
    Part 2 (step 2) runner:
    Track planar marker -> solvePnP -> project + draw a cube -> write output video.
    """
    if not os.path.exists(cfg.calib_output_path):
        raise FileNotFoundError(
            f"Calibration file not found: {cfg.calib_output_path}\n"
            f"Run calibration first: python main.py --part 2 --mode calib"
        )

    ref = _read_image_or_raise(cfg.reference_path)

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {cfg.video_path}")

    fps, out_w, out_h = _video_props(cap)
    K, dist = _load_K_dist_for_video(cfg.calib_output_path, out_w, out_h)

    plane_w, plane_h = _plane_w_h_from_reference(ref, cfg.plane_width)

    obj_plane = np.ascontiguousarray(make_plane_object_points(plane_w, plane_h), dtype=np.float64).reshape(-1, 3)
    cube_3d = make_cube_points(
        plane_w=plane_w,
        plane_h=plane_h,
        size_frac=cfg.cube_size_frac,
        offset_x_frac=cfg.cube_offset_x_frac,
        offset_y_frac=cfg.cube_offset_y_frac,
        height_frac=cfg.cube_height_frac,
    )

    os.makedirs(os.path.dirname(cfg.cube_output_path), exist_ok=True)
    writer = cv2.VideoWriter(cfg.cube_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    tracker = PlanarTracker(ref, tracker_cfg)

    frame_i = 0
    pose_ok = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, corners, dbg = tracker.track(frame)
        out = frame

        if corners is not None:
            img_pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)

            # Guard against bad/empty corners arrays (avoid OpenCV assertion crashes)
            if img_pts.shape != (4, 2) or not np.isfinite(img_pts).all():
                pass
            else:
                ok_pnp, rvec, tvec = _solve_pnp_ippe_fallback(obj_plane, img_pts, K, dist)
                if ok_pnp and rvec is not None and tvec is not None:
                    pose_ok += 1
                    imgpts, _ = cv2.projectPoints(cube_3d, rvec, tvec, K, dist)
                    out = draw_cube(out, imgpts.reshape(-1, 2))

        if cfg.draw_debug_text:
            cv2.putText(
                out,
                f"good={dbg['good']} inliers={dbg['inliers']} pose_ok={pose_ok}/{max(1, frame_i+1)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        writer.write(out)
        frame_i += 1
        if frame_i % 60 == 0:
            print(f"[cube] frames={frame_i} pose_ok={pose_ok}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Saved:", cfg.cube_output_path)
    print("Pose success rate:", pose_ok, "/", frame_i)


def run_part3_model(cfg: Part3Config, tracker_cfg: Part1Config):
    """
    Part 3 runner:
    Track planar marker -> solvePnP -> project + render a 3D mesh (wireframe) -> write output video.
    """
    if not os.path.exists(cfg.calib_output_path):
        raise FileNotFoundError(
            f"Calibration file not found: {cfg.calib_output_path}\n"
            f"Run Part 2 calibration first: python main.py --part 2 --mode calib"
        )

    ref = _read_image_or_raise(cfg.reference_path)

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {cfg.video_path}")

    fps, out_w, out_h = _video_props(cap)
    K, dist = _load_K_dist_for_video(cfg.calib_output_path, out_w, out_h)
    plane_w, plane_h = _plane_w_h_from_reference(ref, cfg.plane_width)
    obj_plane = np.ascontiguousarray(make_plane_object_points(plane_w, plane_h), dtype=np.float64).reshape(-1, 3)

    # Load mesh (or demo)
    if cfg.model_path and os.path.exists(cfg.model_path):
        v, f, face_colors = load_mesh_trimesh(cfg.model_path)
    else:
        v, f = make_demo_tetrahedron()
        face_colors = None

    # Downsample faces for speed if needed.
    # IMPORTANT: don't take the first N faces (it shows only a "piece" of the model).
    # Instead, sample faces across the whole mesh so the shape stays recognizable.
    if cfg.max_faces and int(cfg.max_faces) > 0 and f.shape[0] > int(cfg.max_faces):
        rng = np.random.default_rng(0)
        idx = rng.choice(f.shape[0], size=int(cfg.max_faces), replace=False)
        f = f[idx]
        if face_colors is not None and face_colors.shape[0] >= idx.max() + 1:
            face_colors = face_colors[idx]

    v_plane = transform_mesh_to_plane(
        v,
        plane_w=plane_w,
        plane_h=plane_h,
        scale_frac=cfg.model_scale_frac,
        offset_x_frac=cfg.model_offset_x_frac,
        offset_y_frac=cfg.model_offset_y_frac,
        z_up=True,
        rotate_x_deg=cfg.rotate_x_deg,
        rotate_y_deg=cfg.rotate_y_deg,
        rotate_z_deg=cfg.rotate_z_deg,
    )

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    writer = cv2.VideoWriter(cfg.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    tracker = PlanarTracker(ref, tracker_cfg)
    frame_i = 0
    pose_ok = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, corners, dbg = tracker.track(frame)
        out = frame

        if corners is not None:
            img_pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
            if img_pts.shape == (4, 2) and np.isfinite(img_pts).all():
                ok_pnp, rvec, tvec = _solve_pnp_ippe_fallback(obj_plane, img_pts, K, dist)
                if ok_pnp and rvec is not None and tvec is not None:
                    pose_ok += 1
                    proj, _ = cv2.projectPoints(v_plane, rvec, tvec, K, dist)
                    verts2d = proj.reshape(-1, 2)
                    # Painter's algorithm using vertex depths in camera coordinates (far -> near)
                    R, _ = cv2.Rodrigues(rvec)
                    cam = (R @ v_plane.T + tvec).T
                    z = cam[:, 2]
                    face_depth = z[f].mean(axis=1)
                    order = np.argsort(face_depth)[::-1]

                    if face_colors is not None:
                        out = draw_mesh_flat(out, verts2d, f, face_colors, order=order)
                    else:
                        out = draw_mesh_wireframe(out, verts2d, f, color=(0, 255, 255), thickness=1)

        if cfg.draw_debug_text:
            cv2.putText(
                out,
                f"good={dbg['good']} inliers={dbg['inliers']} pose_ok={pose_ok}/{max(1, frame_i+1)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        writer.write(out)
        frame_i += 1
        if frame_i % 60 == 0:
            print(f"[model] frames={frame_i} pose_ok={pose_ok}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Saved:", cfg.output_path)
    print("Pose success rate:", pose_ok, "/", frame_i)


def _get_pnp_flag_from_name(name: str) -> int:
    n = str(name or "").strip().upper()
    if n == "IPPE" and hasattr(cv2, "SOLVEPNP_IPPE"):
        return int(cv2.SOLVEPNP_IPPE)
    return int(cv2.SOLVEPNP_ITERATIVE)


def run_part4_occlusion(cfg: Part4Config, tracker_cfg: Part1Config):
    """
    Part 4 runner:
    Track planar marker -> solvePnP -> render mesh -> composite with a hand mask.

    Output is clean (no debug overlays / no plane outline), matching Part 3 style.
    """
    if not os.path.exists(cfg.calib_output_path):
        raise FileNotFoundError(
            f"Calibration file not found: {cfg.calib_output_path}\n"
            f"Run Part 2 calibration first: python main.py --part 2 --mode calib"
        )

    ref = _read_image_or_raise(cfg.reference_path)

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {cfg.video_path}")

    fps, out_w, out_h = _video_props(cap)
    K, dist = _load_K_dist_for_video(cfg.calib_output_path, out_w, out_h)
    plane_w, plane_h = _plane_w_h_from_reference(ref, cfg.plane_width)
    obj_plane = np.ascontiguousarray(make_plane_object_points(plane_w, plane_h), dtype=np.float64).reshape(-1, 3)

    # Load mesh + place on plane
    if cfg.model_path and os.path.exists(cfg.model_path):
        v, f, face_colors = load_mesh_trimesh(cfg.model_path)
    else:
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")

    if cfg.max_faces and int(cfg.max_faces) > 0 and f.shape[0] > int(cfg.max_faces):
        rng = np.random.default_rng(0)
        idx = rng.choice(f.shape[0], size=int(cfg.max_faces), replace=False)
        f = f[idx]
        if face_colors is not None and face_colors.shape[0] >= idx.max() + 1:
            face_colors = face_colors[idx]

    v_plane = transform_mesh_to_plane(
        v,
        plane_w=plane_w,
        plane_h=plane_h,
        scale_frac=cfg.model_scale_frac,
        offset_x_frac=cfg.model_offset_x_frac,
        offset_y_frac=cfg.model_offset_y_frac,
        z_up=True,
        rotate_x_deg=cfg.rotate_x_deg,
        rotate_y_deg=cfg.rotate_y_deg,
        rotate_z_deg=cfg.rotate_z_deg,
    )

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    writer = cv2.VideoWriter(cfg.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    tracker = PlanarTracker(ref, tracker_cfg)
    frame_i = 0
    pose_ok = 0

    flag = _get_pnp_flag_from_name(cfg.pnp_variant)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- AR render (Part 3 style) ---
        H, corners, dbg = tracker.track(frame)
        ar_frame = frame.copy()
        ok_pose = False

        if corners is not None:
            img_pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
            if img_pts.shape == (4, 2) and np.isfinite(img_pts).all():
                try:
                    if bool(cfg.use_pnp_ransac) and hasattr(cv2, "solvePnPRansac"):
                        ok_pnp, rvec, tvec, _inl = cv2.solvePnPRansac(
                            obj_plane,
                            img_pts,
                            K,
                            dist,
                            flags=int(flag),
                            reprojectionError=3.0,
                            iterationsCount=100,
                            confidence=0.99,
                        )
                    else:
                        ok_pnp, rvec, tvec = cv2.solvePnP(obj_plane, img_pts, K, dist, flags=int(flag))
                except cv2.error:
                    ok_pnp = False

                if ok_pnp:
                    ok_pose = True
                    pose_ok += 1
                    proj, _ = cv2.projectPoints(v_plane, rvec, tvec, K, dist)
                    verts2d = proj.reshape(-1, 2)

                    # painter order (far->near)
                    R, _ = cv2.Rodrigues(rvec)
                    cam = (R @ v_plane.T + tvec).T
                    z = cam[:, 2]
                    face_depth = z[f].mean(axis=1)
                    order = np.argsort(face_depth)[::-1]

                    if face_colors is not None:
                        ar_frame = draw_mesh_flat(ar_frame, verts2d, f, face_colors, order=order)
                    else:
                        ar_frame = draw_mesh_wireframe(ar_frame, verts2d, f, color=(0, 255, 255), thickness=1)

        # --- Foreground mask (hand) ---
        mask = raw_hsv_mask(
            frame,
            h_min=int(cfg.h_min),
            h_max=int(cfg.h_max),
            s_min=int(cfg.s_min),
            s_max=int(cfg.s_max),
            v_min=int(cfg.v_min),
            v_max=int(cfg.v_max),
        )
        mask = clean_binary_mask(
            mask,
            use_median=bool(cfg.use_median),
            median_ksize=int(cfg.median_ksize),
            use_morph=bool(cfg.use_morph),
            open_ksize=int(cfg.open_ksize),
            close_ksize=int(cfg.close_ksize),
            iters=int(cfg.iters),
        )
        if bool(cfg.use_dilate):
            mask = dilate_mask(mask, ksize=int(cfg.dilate_ksize), iters=int(cfg.dilate_iters))

        # --- Composite (no debug overlays) ---
        out = composite_occlusion(frame, ar_frame, mask)
        writer.write(out)

        frame_i += 1
        if frame_i % 60 == 0:
            print(f"[part4] frames={frame_i} pose_ok={pose_ok}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Saved:", cfg.output_path)
    print("Pose success rate:", pose_ok, "/", frame_i)

def parse_args():
    p = argparse.ArgumentParser(description="Project 2 runner (Parts 1-5).")
    p.add_argument("--part", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Which part to run.")
    p.add_argument("--mode", type=str, default="both", choices=["calib", "cube", "both"],
                   help="Part 2 mode: calibrate intrinsics, render cube, or both.")
    p.add_argument("--part5_video", type=str, default=None, help="Input video path for Part 5.")
    p.add_argument("--ref1", type=str, default=None, help="Reference image 1 for Part 5.")
    p.add_argument("--ref2", type=str, default=None, help="Reference image 2 for Part 5.")
    p.add_argument("--ref3", type=str, default=None, help="Reference image 3 for Part 5.")
    p.add_argument("--feature", type=str, default=None, choices=["SIFT", "ORB"], help="Feature type for Part 5.")
    p.add_argument("--part5_mode", type=str, default="portal", choices=["raw", "outline", "portal", "portal360"], help="Part 5 export: raw, outline-only, solid portal, or portal360 (panorama).")
    p.add_argument("--part5_out", type=str, default=None, help="Output video path for Part 5 (optional).")

    # Optional Part 2 overrides
    p.add_argument("--calib_glob", type=str, default=None, help="e.g. data/chessboard/*.jpg")
    p.add_argument("--chessboard_cols", type=int, default=None, help="INNER corners cols, e.g. 8")
    p.add_argument("--chessboard_rows", type=int, default=None, help="INNER corners rows, e.g. 6")
    p.add_argument("--square_size", type=float, default=None, help="Square size (same unit as you want translations in)")
    p.add_argument("--calib_out", type=str, default=None, help="e.g. outputs/camera/calibration.npz")
    p.add_argument("--show_detections", action="store_true", help="Show detected chessboard corners briefly")

    # Optional cube overrides
    p.add_argument("--video_path", type=str, default=None, help="Input video for planar AR")
    p.add_argument("--reference_path", type=str, default=None, help="Reference image (printed marker)")
    p.add_argument("--cube_out", type=str, default=None, help="Output video path for cube render")
    p.add_argument("--plane_width", type=float, default=None, help="Plane width in world units")
    p.add_argument("--cube_size_frac", type=float, default=None, help="Cube size as fraction of plane width")
    p.add_argument("--cube_offset_x_frac", type=float, default=None, help="Cube X offset fraction on plane")
    p.add_argument("--cube_offset_y_frac", type=float, default=None, help="Cube Y offset fraction on plane")
    p.add_argument("--cube_height_frac", type=float, default=None, help="Cube height as fraction of cube size")

    # Part 3 args (mesh)
    p.add_argument("--model_path", type=str, default=None, help="Path to .obj/.ply model (Part 3)")
    p.add_argument("--model_out", type=str, default=None, help="Output video path for Part 3")
    p.add_argument("--model_scale_frac", type=float, default=None, help="Model scale as fraction of plane width")
    p.add_argument("--model_offset_x_frac", type=float, default=None, help="Model X offset fraction on plane")
    p.add_argument("--model_offset_y_frac", type=float, default=None, help="Model Y offset fraction on plane")
    p.add_argument("--max_faces", type=int, default=None, help="Max faces to render (wireframe) for speed")
    p.add_argument("--rotate_x_deg", type=float, default=None, help="Rotate model around X before placement (degrees)")
    p.add_argument("--rotate_y_deg", type=float, default=None, help="Rotate model around Y before placement (degrees)")
    p.add_argument("--rotate_z_deg", type=float, default=None, help="Rotate model around Z before placement (degrees)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.part == 1:
        run_part1(Part1Config())
    elif args.part == 2:
        base = Part2Config()
        cfg = Part2Config(
            calib_images_glob=(args.calib_glob if args.calib_glob is not None else base.calib_images_glob),
            chessboard_cols=(args.chessboard_cols if args.chessboard_cols is not None else base.chessboard_cols),
            chessboard_rows=(args.chessboard_rows if args.chessboard_rows is not None else base.chessboard_rows),
            square_size=(args.square_size if args.square_size is not None else base.square_size),
            calib_output_path=(args.calib_out if args.calib_out is not None else base.calib_output_path),
            show_detections=(args.show_detections or base.show_detections),
            video_path=(args.video_path if args.video_path is not None else base.video_path),
            reference_path=(args.reference_path if args.reference_path is not None else base.reference_path),
            cube_output_path=(args.cube_out if args.cube_out is not None else base.cube_output_path),
            plane_width=(args.plane_width if args.plane_width is not None else base.plane_width),
            cube_size_frac=(args.cube_size_frac if args.cube_size_frac is not None else base.cube_size_frac),
            cube_offset_x_frac=(args.cube_offset_x_frac if args.cube_offset_x_frac is not None else base.cube_offset_x_frac),
            cube_offset_y_frac=(args.cube_offset_y_frac if args.cube_offset_y_frac is not None else base.cube_offset_y_frac),
            cube_height_frac=(args.cube_height_frac if args.cube_height_frac is not None else base.cube_height_frac),
            draw_debug_text=base.draw_debug_text,
        )

        if args.mode in ("calib", "both"):
            run_part2_calibration(cfg)

        if args.mode in ("cube", "both"):
            # Reuse Part 1 tracking params (ratio/min_matches/ransac thresholds)
            run_part2_cube(cfg, Part1Config())
    elif args.part == 3:
        base = Part3Config()
        cfg3 = Part3Config(
            calib_output_path=base.calib_output_path,
            video_path=base.video_path,
            reference_path=base.reference_path,
            model_path=(args.model_path if args.model_path is not None else base.model_path),
            output_path=(args.model_out if args.model_out is not None else base.output_path),
            plane_width=base.plane_width,
            model_scale_frac=(args.model_scale_frac if args.model_scale_frac is not None else base.model_scale_frac),
            model_offset_x_frac=(args.model_offset_x_frac if args.model_offset_x_frac is not None else base.model_offset_x_frac),
            model_offset_y_frac=(args.model_offset_y_frac if args.model_offset_y_frac is not None else base.model_offset_y_frac),
            max_faces=(args.max_faces if args.max_faces is not None else base.max_faces),
            rotate_x_deg=(args.rotate_x_deg if args.rotate_x_deg is not None else base.rotate_x_deg),
            rotate_y_deg=(args.rotate_y_deg if args.rotate_y_deg is not None else base.rotate_y_deg),
            rotate_z_deg=(args.rotate_z_deg if args.rotate_z_deg is not None else base.rotate_z_deg),
            draw_debug_text=base.draw_debug_text,
        )
        run_part3_model(cfg3, Part1Config())
    elif args.part == 4:
        # Part 4: occlusion handling (clean output, no debug overlays)
        run_part4_occlusion(Part4Config(), Part1Config())
    elif args.part == 5:
        base5 = Part5Config()
        cfg5 = Part5Config(
            calib_output_path=base5.calib_output_path,
            video_path=(args.part5_video if args.part5_video is not None else base5.video_path),
            reference1_path=(args.ref1 if args.ref1 is not None else base5.reference1_path),
            reference2_path=(args.ref2 if args.ref2 is not None else base5.reference2_path),
            reference3_path=(args.ref3 if args.ref3 is not None else base5.reference3_path),
            feature_type=(args.feature if args.feature is not None else base5.feature_type),
            ratio_test=base5.ratio_test,
            min_matches=base5.min_matches,
            ransac_reproj_thresh=base5.ransac_reproj_thresh,
            min_inliers=base5.min_inliers,
            homography_method=base5.homography_method,
            refine_homography_with_inliers=base5.refine_homography_with_inliers,
            max_hold_frames=base5.max_hold_frames,
            plane_width=base5.plane_width,
            portal_size_frac=base5.portal_size_frac,
            portal_fill_bgr=base5.portal_fill_bgr,
            portal_border_bgr=base5.portal_border_bgr,
            portal_border_thickness=base5.portal_border_thickness,
            portal_alpha=base5.portal_alpha,
            draw_plane_outline=base5.draw_plane_outline,
            draw_debug_text=base5.draw_debug_text,
        )
        mode = str(args.part5_mode or "portal").lower().strip()
        if mode == "raw":
            outp = args.part5_out if args.part5_out is not None else "outputs/videos/part5_raw.mp4"
            export_part5_video(cfg5, outp, draw_portal=False)
        elif mode == "outline":
            outp = args.part5_out if args.part5_out is not None else "outputs/videos/part5_outline.mp4"
            export_part5_video(cfg5, outp, draw_portal=False, draw_plane_outline=True, draw_debug_text=False)
        elif mode == "portal":
            outp = args.part5_out if args.part5_out is not None else "outputs/videos/part5_portal.mp4"
            export_part5_video(cfg5, outp, draw_portal=True, use_env_portals=False, draw_plane_outline=True, draw_debug_text=False)
        elif mode == "portal360":
            outp = args.part5_out if args.part5_out is not None else "outputs/videos/part5_portal360.mp4"
            export_part5_video(cfg5, outp, draw_portal=True, use_env_portals=True, draw_plane_outline=True, draw_debug_text=False)
        else:
            run_part5_multiplane(cfg5)

