# utils.py
"""
Utility functions for drawing and visualization.
"""
import cv2
import time

def draw_tracks(frame, mask, good_new, good_old, colors, min_movement=0.0):
    """
    Draws the tracking lines (motion trail) and current feature points on the frame.
    
    Args:
        frame: The current video frame.
        mask: An image containing the accumulated motion trails.
        good_new: The tracked feature points in the current frame.
        good_old: The corresponding feature points in the previous frame.
        colors: A list of colors for each tracked point.
        min_movement: Minimum Euclidean distance to consider as valid movement.
        
    Returns:
        The frame overlaid with tracking visuals.
    """
    import math
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel() # Current point coordinates
        c, d = old.ravel() # Previous point coordinates
        
        # Calculate Euclidean distance
        dist = math.hypot(a - c, b - d)
        
        # Convert coordinates to integers for drawing
        a, b, c, d = int(a), int(b), int(c), int(d)
        
        # Ignore tiny noisy movements for the trail
        if dist > min_movement:
            # Draw the line representing the trajectory on the mask
            mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
        
        # Draw a circle at the current point location on the frame
        frame = cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)
        
    # Combine the frame and the mask
    img = cv2.add(frame, mask)
    return img, mask

def display_fps(frame, start_time):
    """
    Calculates and displays the frames per second (FPS) on the frame.
    
    Args:
        frame: The current video frame.
        start_time: The time when the frame processing started.
        
    Returns:
        The frame with FPS text drawn, and the end time for the next frame calculation.
    """
    end_time = time.time()
    fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame, end_time
