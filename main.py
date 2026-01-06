import os
import argparse

import cv2
import numpy as np

from config import Part1Config, Part2Config
from tracker import PlanarTracker, warp_and_overlay
from camera import calibrate_from_chessboard_images, expand_image_glob, save_calibration_npz, load_calibration_npz
from ar_render import draw_cube, make_cube_points, make_plane_object_points, scale_K_to_new_size


def run_part1(cfg: Part1Config):
    """
    Part 1 runner:
    SIFT -> ratio test -> homography (RANSAC) -> warp template -> write output video.
    """

    # Read inputs
    ref = cv2.imread(cfg.reference_path)
    if ref is None:
        raise FileNotFoundError(f"Could not read: {cfg.reference_path}")

    template = cv2.imread(cfg.template_path)
    if template is None:
        raise FileNotFoundError(f"Could not read: {cfg.template_path}")

    if cfg.resize_template_to_reference:
        h, w = ref.shape[:2]
        template = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {cfg.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
            cv2.putText(out, f"good={dbg['good']} inliers={dbg['inliers']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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

    ref = cv2.imread(cfg.reference_path)
    if ref is None:
        raise FileNotFoundError(f"Could not read: {cfg.reference_path}")

    calib = load_calibration_npz(cfg.calib_output_path)
    K_calib = calib.K
    dist = calib.dist

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {cfg.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale K if calibration resolution differs from the video resolution.
    K = K_calib
    if calib.image_size != (out_w, out_h):
        K = scale_K_to_new_size(K_calib, calib.image_size, (out_w, out_h))

    # Plane dimensions in world units (Z=0).
    ref_h, ref_w = ref.shape[:2]
    ref_aspect = ref_h / float(ref_w)
    plane_w = float(cfg.plane_width)
    plane_h = float(cfg.plane_width) * ref_aspect

    obj_plane = make_plane_object_points(plane_w, plane_h)
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
            img_pts = corners.reshape(-1, 2).astype(np.float32)
            ok_pnp, rvec, tvec = cv2.solvePnP(obj_plane, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok_pnp:
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


def parse_args():
    p = argparse.ArgumentParser(description="Project 2 runner (Parts 1-5).")
    p.add_argument("--part", type=int, default=1, choices=[1, 2], help="Which part to run (implemented: 1-2).")
    p.add_argument("--mode", type=str, default="both", choices=["calib", "cube", "both"],
                   help="Part 2 mode: calibrate intrinsics, render cube, or both.")

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

