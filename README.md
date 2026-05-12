# Moving Object Tracking System

A beginner-friendly, college-level computer vision project that implements real-time moving object tracking using Lucas-Kanade Optical Flow and KLT feature tracking.

## Features
- Real-time webcam or video file support
- KLT (Kanade-Lucas-Tomasi) feature detection
- Lucas-Kanade Optical Flow tracking
- Clean motion trail drawing with smooth fading effect
- Intelligent noise filtering to ignore tiny camera vibrations
- Live FPS display
- Manual (`r`) and automatic feature re-detection
- Option to save output video

## Requirements
- Python 3.8+
- OpenCV
- NumPy

## Installation
1. Ensure Python is installed on your Windows machine.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script:
```bash
python main.py
```

- Press `q` to quit the application.
- Press `r` to manually re-detect tracking features.

## Architecture
- `config.py`: Centralized configuration parameters.
- `utils.py`: Drawing and visualization helper functions.
- `tracker.py`: Optical flow and feature detection implementation.
- `main.py`: Video capture loop and application logic.
