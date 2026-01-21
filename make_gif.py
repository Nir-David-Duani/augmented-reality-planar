import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _get_duration_sec(video_path: str) -> float | None:
    """Return duration in seconds using ffprobe, or None if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return float(out)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a high-quality GIF from a video segment.")
    ap.add_argument("--input", required=True, help="Path to input video")
    ap.add_argument("--output", required=True, help="Path to output GIF")
    ap.add_argument("--start", type=float, required=True, help="Start time in seconds")
    ap.add_argument("--end", type=float, required=True, help="End time in seconds")
    ap.add_argument("--fps", type=int, default=15, help="GIF FPS (default: 15)")
    ap.add_argument("--width", type=int, default=720, help="Output width (keep aspect). Default: 720")
    ap.add_argument("--palette", type=str, default="palette.png", help="Temporary palette path")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    if args.end <= args.start:
        raise ValueError("end must be > start")

    dur = _get_duration_sec(str(inp))
    if dur is not None and args.start >= dur:
        raise ValueError(f"start ({args.start}) is beyond video duration ({dur:.2f}s)")
    if dur is not None and args.end > dur:
        print(f"[warn] end ({args.end}) > duration ({dur:.2f}s); truncating to {dur:.2f}s")
        args.end = dur

    duration = args.end - args.start
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Write palette next to output to avoid cwd confusion.
    palette = Path(args.palette)
    if not palette.is_absolute():
        palette = outp.parent / palette
    vf = f"fps={args.fps},scale={args.width}:-1:flags=lanczos"

    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(args.start),
            "-t",
            str(duration),
            "-i",
            str(inp),
            "-vf",
            vf + ",palettegen=stats_mode=full",
            str(palette),
        ]
    )

    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(args.start),
            "-t",
            str(duration),
            "-i",
            str(inp),
            "-i",
            str(palette),
            "-lavfi",
            vf + " [x]; [x][1:v] paletteuse=dither=sierra2_4a",
            str(outp),
        ]
    )

    try:
        palette.unlink(missing_ok=True)
    except Exception:
        pass

    print("Saved:", str(outp))


if __name__ == "__main__":
    main()

