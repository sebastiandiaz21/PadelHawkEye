# Padel HawkEye

This repository contains a simple implementation to process padel match videos from a fixed camera. The program detects the ball, estimates its trajectory and checks that the camera view remains calibrated.

## Requirements

- Python 3
- OpenCV
- NumPy
- SciPy (for polynomial fitting)

Install dependencies:

```bash
pip install opencv-python numpy scipy
```

## Usage

Run the main program with a video file or camera source:

```bash
python main.py --video path/to/video.mp4
```

To specify initial calibration points (pixels) use `--points` with eight comma-separated values `x1,y1,x2,y2,x3,y3,x4,y4` representing the court corners in order.

Press `q` to quit.
