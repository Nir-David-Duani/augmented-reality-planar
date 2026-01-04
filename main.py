import os
import argparse

import cv2

from config import Part1Config
from tracker import PlanarTracker, warp_and_overlay


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


def parse_args():
    p = argparse.ArgumentParser(description="Project 2 runner (Parts 1-5).")
    p.add_argument("--part", type=int, default=1, choices=[1], help="Which part to run (implemented: 1).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.part == 1:
        run_part1(Part1Config())

