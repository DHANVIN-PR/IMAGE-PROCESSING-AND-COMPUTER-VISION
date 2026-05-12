# tracker.py
"""
Core tracking logic using Lucas-Kanade Optical Flow.
"""
import cv2
import numpy as np
from config import FEATURE_PARAMS, LK_PARAMS

class OpticalFlowTracker:
    def __init__(self):
        """Initializes the tracker state."""
        self.old_gray = None
        self.p0 = None
        # Create random colors for maximum 100 features
        self.colors = np.random.randint(0, 255, (100, 3))
        self.mask = None

    def initialize(self, first_frame):
        """
        Detects initial features to track in the first frame.
        
        Args:
            first_frame: The initial video frame.
        """
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        # Find features to track using Shi-Tomasi corner detection (KLT)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **FEATURE_PARAMS)
        # Create a mask image for drawing purposes
        self.mask = np.zeros_like(first_frame)

    def track(self, current_frame):
        """
        Tracks features in the current frame based on previous frame.
        
        Args:
            current_frame: The new video frame to process.
            
        Returns:
            good_new: Successfully tracked points in current frame.
            good_old: Corresponding points from previous frame.
            mask: The accumulated drawing mask.
        """
        frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # If no features to track, return empty
        if self.p0 is None or len(self.p0) == 0:
            return np.array([]), np.array([]), self.mask

        # Calculate optical flow using Lucas-Kanade method
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_gray, frame_gray, self.p0, None, **LK_PARAMS
        )

        if p1 is not None:
            # Select good points (status == 1 means feature was found)
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
        else:
            good_new, good_old = np.array([]), np.array([])

        # Update previous frame and previous points for next iteration
        self.old_gray = frame_gray.copy()
        if len(good_new) > 0:
            self.p0 = good_new.reshape(-1, 1, 2)
        else:
            self.p0 = None
        
        return good_new, good_old, self.mask

    def update_mask(self, new_mask):
        """Updates the drawing mask."""
        self.mask = new_mask
