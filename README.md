## Project Structure and File Responsibilities

This project is organized as a modular planar augmented reality system.
Each file has a single, well-defined responsibility.
This separation is intentional and is designed to support incremental
development, debugging, and clear reasoning.

--------------------------------------------------

### main.py
Entry point of the project.

Responsibilities:
- Open video file or camera stream
- Run the main processing loop
- Call the appropriate modules based on the current mode (Part 1–5)
- Save output videos and display results

main.py contains orchestration logic only.
It does not implement core algorithms.

--------------------------------------------------

### config.py
Single source of truth for the entire project.

Responsibilities:
- File paths (data, outputs, models)
- Algorithm parameters (feature detection, RANSAC, thresholds)
- Camera and AR parameters (cube size, visualization mode)
- Global configuration flags

No algorithmic logic is implemented here.

--------------------------------------------------

### tracker.py
Planar surface detection and tracking (Part 1).

Responsibilities:
- Detect features in the reference image
- Match features between reference and video frames
- Estimate planar homography using RANSAC
- Extract 2D corner locations of the tracked planar surface

Outputs:
- Homography matrix
- 2D corner coordinates
- Optional debug visualizations

--------------------------------------------------

### camera.py
Camera modeling and pose estimation (Part 2).

Responsibilities:
- Camera calibration using a chessboard
- Load and store camera intrinsics and distortion coefficients
 - (Used by Part 2) Provide intrinsics/distortion for pose estimation

Outputs:
- Camera intrinsics matrix K
- Distortion coefficients

--------------------------------------------------

### ar_render.py
Augmented reality rendering (Parts 2 and 3).

Responsibilities:
- Project 3D points onto the image plane
- Render a 3D cube aligned with the planar surface
- Render a full 3D mesh model
- Handle basic depth ordering if needed

This module assumes a valid pose and camera model are provided.

--------------------------------------------------

## Implemented Parts

### Part 1
- Run: `python main.py --part 1`
- Output: `outputs/videos/part1.mp4`

### Part 2
- Run calibration + cube: `python main.py --part 2 --mode both`
- Run only calibration: `python main.py --part 2 --mode calib`
- Run only cube (requires existing calibration): `python main.py --part 2 --mode cube`
- Outputs:
  - `outputs/camera/calibration.npz`
  - `outputs/videos/part2_cube.mp4`

--------------------------------------------------

### occlusion.py
Occlusion handling (Part 4).

Responsibilities:
- Detect foreground objects using classical computer vision techniques
- Generate a binary occlusion mask
- Composite AR content with correct occlusion behavior

No tracking or pose estimation is performed here.

--------------------------------------------------

### multi_plane.py
Multi-plane tracking and visualization (Part 5).

Responsibilities:
- Manage multiple planar trackers simultaneously
- Estimate pose for each tracked plane
- Implement visualization logic (portal or particle flow)
- Handle tracking loss and recovery

--------------------------------------------------

### data/
Contains all static input data:
- Reference images
- Template images
- Camera calibration images
- 3D models

--------------------------------------------------

### outputs/
Contains all generated results:
- Output videos
- Debug visualizations

--------------------------------------------------

### PROJECT_NOTES.md
Internal development notes.

Purpose:
- Design decisions
- Rules of thumb
- Debugging observations
- Known limitations and failures

This file is not part of the final report, but supports development.

--------------------------------------------------

### requirements.txt
Lists all Python dependencies required to run the project.

--------------------------------------------------

## Design Philosophy
- One responsibility per file
- No duplicated logic
- Debugging before optimization
- Stability over visual perfection

The system is built as a classical computer vision pipeline,
not as a learning-based system.
