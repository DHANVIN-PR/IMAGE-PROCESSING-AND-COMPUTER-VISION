# config.py
"""
Configuration parameters for the Object Tracking project.
"""

# Video source (0 for default webcam, or path to video file)
VIDEO_SOURCE = 0 

# ShiTomasi corner detection parameters
# These parameters decide how features are detected initially
FEATURE_PARAMS = dict(
    maxCorners=100,      # Increased to 100 for better tracking on complex objects
    qualityLevel=0.15,   # Lowered to 0.15 to detect more features
    minDistance=6,       # Reduced to 6 to allow features slightly closer together
    blockSize=7          # Size of block used for computing derivative covariation matrix
)

# Visualization settings
FADE_RATE = 0.85         # Balanced to 0.85 for faster and cleaner trail fading
MIN_MOVEMENT = 1.0       # Minimum distance (pixels) to register as movement

# Lucas-Kanade optical flow parameters
# These parameters dictate how tracking behaves between frames
LK_PARAMS = dict(
    winSize=(15, 15),    # Size of the search window at each pyramid level
    maxLevel=2,          # 0-based maximal pyramid level number
    criteria=(3, 10, 0.03) # Termination criteria (type, maxCount, epsilon)
)

# Output video settings
SAVE_VIDEO = False       # Set to True to save the output video
OUTPUT_FILE = "output.avi"
OUTPUT_FPS = 30
