# main.py
"""
Main entry point for the Moving Object Tracking application.
"""
import cv2
import time
import numpy as np
from config import VIDEO_SOURCE, SAVE_VIDEO, OUTPUT_FILE, OUTPUT_FPS, FADE_RATE, MIN_MOVEMENT
from tracker import OpticalFlowTracker
from utils import draw_tracks, display_fps

def main():
    # 1. Initialize Video Capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return

    # 2. Setup Video Writer (if saving is enabled)
    out = None
    if SAVE_VIDEO:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Use XVID codec for Windows compatibility
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(OUTPUT_FILE, fourcc, OUTPUT_FPS, (frame_width, frame_height))

    # 3. Read the first frame and initialize Tracker
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    tracker = OpticalFlowTracker()
    tracker.initialize(first_frame)

    print("Tracking started. Press 'q' to quit. Press 'r' to re-detect features.")

    # 4. Main Processing Loop
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or error reading frame.")
            break

        # Track features using Lucas-Kanade
        good_new, good_old, mask = tracker.track(frame)
        
        # Automatic feature re-detection if tracking points are lost
        if len(good_new) == 0:
            tracker.initialize(frame)
            continue

        # Draw trajectories and points
        output_frame, updated_mask = draw_tracks(frame, mask, good_new, good_old, tracker.colors, MIN_MOVEMENT)
        
        # Apply fading effect to the mask
        # cv2.addWeighted blends the updated_mask with a black image (zeros_like)
        # using FADE_RATE (e.g. 0.95), effectively fading the existing pixels.
        updated_mask = cv2.addWeighted(updated_mask, FADE_RATE, np.zeros_like(updated_mask), 0.0, 0)
        
        tracker.update_mask(updated_mask)

        # Display FPS
        output_frame, prev_time = display_fps(output_frame, prev_time)

        # Show the result in a window
        cv2.imshow('Optical Flow Tracking', output_frame)

        # Save frame if enabled
        if out is not None:
            out.write(output_frame)

        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            # Quit application
            break
        elif key == ord('r'):
            # Manually re-detect features
            tracker.initialize(frame)

    # 5. Cleanup resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
