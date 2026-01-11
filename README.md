## Planar Augmented Reality (Computer Vision Project)

This repository implements a **classical (non‑learning) planar AR pipeline** across 5 parts:

- **Part 1**: Planar tracking (SIFT → matches → RANSAC homography) + template warp
- **Part 2**: Camera calibration (chessboard) + AR cube via pose estimation (`solvePnP`)
- **Part 3**: AR 3D mesh rendering on the planar target
- **Part 4**: Occlusion handling (foreground mask → composite)
- **Part 5**: Multi‑plane tracking (3 targets) + portal / portal360 visualization

---

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run parts (from inside `augmented-reality-planar/`):

```bash
python main.py --part 1
python main.py --part 2 --mode both
python main.py --part 3
python main.py --part 4
python main.py --part 5 --part5_mode portal
```

Outputs are written under `outputs/` (videos under `outputs/videos/`).

---

## Clean notebooks (recommended for reviewing the project)

These notebooks are aligned with the **actual code** in the repository and are written to be readable:

- `01_Part1_Planar_Tracking_and_Warp.ipynb`
- `02_Part2_Calibration_and_AR_Cube.ipynb`
- `03_Part3_AR_Mesh_Rendering.ipynb`
- `04_Part4_Occlusion_Masking.ipynb`
- `05_Part5_MultiPlane_Portals.ipynb`

Notes:
- Parts 2–5 include a `frame_idx` variable to pick a specific frame for single‑frame debugging/visualization.
- Part 5 can show either a solid portal fill or a portal360 texture (from `data/env*_360.jpg`).

---

## Results (sample images)

### Part 1 — Tracking + warp

![Part 1: keypoints and matches](../results%20part%201/SIFT%20keypoints%20both.png)
![Part 1: overlay result](../results%20part%201/result.png)

### Part 2 — Calibration + cube

![Part 2: undistortion](../results%20part%202/undistorted.png)
![Part 2: cube render](../results%20part%202/cube.png)

### Part 3 — Mesh render

![Part 3: mesh render](../results%20part%203/result.png)

### Part 4 — Occlusion

![Part 4: mask + morphology](../results%20part%204/morphology%20and%20mask.png)
![Part 4: final composite](../results%20part%204/final.png)

### Part 5 — Multi‑plane portals

![Part 5: portals](../results%20part%205/output%20portals.png)

---

## Links

- **Videos (Google Drive)**: [Drive folder](https://drive.google.com/file/d/1-Vjy_mMf9JbnEZ5UbXa0gbWikDwWZ7bt/view?usp=drive_link)
- **Report (PDF)**: _Coming soon_ (add link here once the PDF is ready)

---

## Project structure (core modules)

- `main.py`: Entry point / orchestration (selects which part to run, reads inputs, writes videos).
- `config.py`: Central configuration (paths + algorithm parameters).
- `tracker.py`: Planar tracking (SIFT + matching + homography) used in Parts 1–4.
- `camera.py`: Camera calibration helpers + calibration I/O (`outputs/camera/calibration.npz`).
- `ar_render.py`: 3D utilities (cube points, mesh loading, projection, rendering).
- `occlusion.py`: Foreground mask + compositing (Part 4).
- `multiplane.py`: Multi‑plane tracking + portal rendering (Part 5).

---

## Inputs and outputs

- **Inputs**: `data/` (videos, reference images, chessboard images, models, portal environments).
- **Outputs**: `outputs/`
  - `outputs/camera/calibration.npz`
  - `outputs/videos/*.mp4`

---

## CLI options (useful)

- Part 2:
  - `--mode calib | cube | both`
- Part 3:
  - `--model_path ...` / `--model_out ...`
  - `--rotate_x_deg ...` / `--rotate_y_deg ...` / `--rotate_z_deg ...`
- Part 5:
  - `--part5_mode raw | outline | portal | portal360`
  - `--part5_out ...`

---
