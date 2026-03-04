import cv2
import numpy as np
from ultralytics import YOLO
import os

def get_dominant_color_kmeans(crop_bgr, k=3):
    """
    Extracts the dominant color of the crop using K-Means clustering in HSV space.
    Ignores green background (pitch) pixels to isolate the jersey.
    """
    crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for green (grass/pitch)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green and invert it to get non-green pixels
    mask_green = cv2.inRange(crop_hsv, lower_green, upper_green)
    mask_non_green = cv2.bitwise_not(mask_green)
    
    # Extract only non-green pixels
    pixels = crop_hsv[mask_non_green > 0]
    
    # If there are fewer pixels than clusters, fallback
    if len(pixels) < k:
        return None
        
    pixels = np.float32(pixels)
    
    # Apply K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Find the most frequent cluster label
    counts = np.bincount(labels.flatten())
    dominant_cluster_idx = np.argmax(counts)
    dominant_hsv = centers[dominant_cluster_idx]
    
    return dominant_hsv

def main():
    # Load the trained custom model weights
    # Load the heavier YOLOv8 Medium model weights for better detection
    model_path = 'yolov8m.pt' # Switching to the medium model
    print(f"Loading medium model from {model_path}...")
    model = YOLO(model_path)

    input_video_path = 'input_video2.mp4'
    output_video_path = 'runs/track/football_match_tracking/custom_output2.mp4'
    screenshot_path = 'runs/track/football_match_tracking/screenshot_autocalibration.jpg'
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Starting custom K-Means auto-calibration tracking on {input_video_path}...")
    
    frame_count = 0
    saved_screenshot = False
    
    # Auto-calibration state variables
    calibration_colors = []
    team_a_color_hsv = None
    team_b_color_hsv = None
    
    # Label Smoothing tracker
    label_history = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        objects_detected = False
        
        # Run YOLO tracking on the frame with increased resolution and extremely low confidence for distant small objects
        results = model.track(frame, persist=True, tracker='botsort.yaml', conf=0.08, imgsz=1280, verbose=False)[0]
        
        if results.boxes is not None and results.boxes.id is not None:
            objects_detected = True
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            class_ids = results.boxes.cls.int().cpu().tolist()
            confs = results.boxes.conf.cpu().numpy()
            
            for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1
                
                # ISSUE 1: Aspect Ratio Filter (skip tall and skinny like corner flags)
                if h > 0:
                    aspect_ratio = w / h
                    if aspect_ratio < 0.35:
                        continue
                
                label = model.names[class_id]
                color = (0, 255, 0) # Default
                
                # ISSUE 2: Implement Pitch Filter to ignore background people
                if label in ['player', 'referee', 'goalkeeper', 'person']:
                    x_center = int((x1 + x2) / 2)
                    y_feet = min(y2, height - 1)
                    x_center = max(0, min(x_center, width - 1))
                    y_feet = max(0, y_feet)
                    
                    # Define a 10x10 patch at the bottom center
                    patch_y1 = max(0, y_feet - 10)
                    patch_y2 = y_feet
                    patch_x1 = max(0, x_center - 5)
                    patch_x2 = min(width - 1, x_center + 5)
                    
                    patch_bgr = frame[patch_y1:patch_y2, patch_x1:patch_x2]
                    
                    if patch_bgr.size > 0:
                        patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
                        # Calculate median HSV of the patch
                        pH = np.median(patch_hsv[:, :, 0])
                        pS = np.median(patch_hsv[:, :, 1])
                        pV = np.median(patch_hsv[:, :, 2])
                        
                        # Check if the median pixel at their feet is NOT standard green grass
                        if not (35 <= pH <= 85 and pS >= 40 and pV >= 40):
                            continue # Discard this bounding box (person is in stands/sideline)
                    else:
                        continue
                
                # Dynamic AUTO-CALIBRATION Team Color Classification
                if label in ['player', 'referee', 'goalkeeper', 'person'] and h > 10:
                    # Crop 20% to 60% of the height to strictly focus on the torso/jersey
                    # This avoids the head (top 20%) and the shorts (bottom 40%)
                    crop_bgr = frame[max(0, y1 + int(h * 0.2)):max(0, y1 + int(h * 0.6)), max(0, x1):max(0, x2)]
                    
                    if crop_bgr.size > 0:
                        dominant_hsv = get_dominant_color_kmeans(crop_bgr, k=3)
                        
                        if dominant_hsv is not None:
                            if frame_count <= 60:
                                # INITIALIZATION PHASE
                                calibration_colors.append(dominant_hsv)
                                label = "Calibrating..."
                                color = (200, 200, 200)
                            else:
                                # GLOBAL CLUSTERING AT FRAME 61
                                if frame_count == 61 and team_a_color_hsv is None:
                                    if len(calibration_colors) >= 2:
                                        print(f"Collected {len(calibration_colors)} colors for calibration. Running K-Means...")
                                        pixels = np.array(calibration_colors, dtype=np.float32)
                                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                                        _, _, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                                        team_a_color_hsv = centers[0]
                                        team_b_color_hsv = centers[1]
                                        print("--- AUTO CALIBRATION COMPLETE ---")
                                        print(f"Team A Base Color (HSV): {team_a_color_hsv}")
                                        print(f"Team B Base Color (HSV): {team_b_color_hsv}")
                                        print("---------------------------------")
                                    else:
                                        print("Warning: Not enough colors collected for rigorous calibration.")
                                        team_a_color_hsv = np.array([0, 255, 255])
                                        team_b_color_hsv = np.array([120, 255, 255])
                                
                                # DYNAMIC ASSIGNMENT PHASE
                                if team_a_color_hsv is not None and team_b_color_hsv is not None:
                                    # Calculate Euclidean distance in HSV space
                                    dist_a = np.linalg.norm(dominant_hsv - team_a_color_hsv)
                                    dist_b = np.linalg.norm(dominant_hsv - team_b_color_hsv)
                                    
                                    # Outlier Detection threshold (relax tolerance for shadowed players)
                                    outlier_threshold = 80.0
                                    
                                    if min(dist_a, dist_b) > outlier_threshold:
                                        # Check dominant HSV to distinguish Goalkeeper (dark) and Referee (bright/yellowish)
                                        h, s, v = dominant_hsv
                                        # Increase brightness threshold to Value < 100
                                        if v < 100:
                                            label = "Goalkeeper"
                                            color = (0, 0, 0) # Black box
                                        else:
                                            label = "Referee"
                                            color = (0, 255, 255) # Yellow box
                                    elif dist_a < dist_b:
                                        label = "Team A"
                                        color = (0, 0, 255) # Red box
                                    else:
                                        label = "Team B"
                                        color = (255, 255, 255) # White box
                        else:
                            # Fallback if unclassified (e.g. box had only green grass pixels)
                            label = "Player"
                            color = (0, 255, 0)
                
                # Ensure no generic "person" labels remain
                if label == 'person':
                    label = "Player"
                
                # ISSUE: Label Smoothing (Majority Voting)
                # Store the current frame's raw classification in the track ID's history
                if track_id not in label_history:
                    label_history[track_id] = []
                
                # Append the new label
                label_history[track_id].append((label, color))
                
                # Restrict to the last 30 frames for strict sliding window tracking
                label_history[track_id] = label_history[track_id][-30:]
                
                # Calculate Majority Vote (Mode)
                # Extract just the labels from the history list of tuples
                recent_labels = [item[0] for item in label_history[track_id]]
                
                # Find the most frequent label in the last 30 frames
                smoothed_label = max(set(recent_labels), key=recent_labels.count)
                
                # Find the corresponding color for that smoothed label
                smoothed_color = color # fallback
                for item in label_history[track_id]:
                    if item[0] == smoothed_label:
                        smoothed_color = item[1]
                        break
                        
                label = smoothed_label
                color = smoothed_color
                
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # ISSUE 3: Improve ID Text Visibility
                display_text = f"ID:{track_id} {label}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
                
                # Compute coordinates for background rectangle
                bg_x1 = x1
                bg_y1 = y1 - text_height - 6
                if bg_y1 < 0: # Ensure background isn't drawn off-screen
                    bg_y1 = 0
                bg_x2 = x1 + text_width + 4
                bg_y2 = y1
                
                # Draw filled dark rectangle background for text
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                
                # Draw bright white text over the background
                cv2.putText(frame, display_text, (x1 + 2, y1 - 3), font, font_scale, (255, 255, 255), thickness)
                
        # Save a screenshot of the frame AFTER calibration completes (e.g. frame 65)
        if objects_detected and not saved_screenshot and frame_count > 65:
            cv2.imwrite(screenshot_path, frame)
            saved_screenshot = True
            
        out.write(frame)
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Tracking complete! The Auto-Calibrated tracked video and screenshot are saved.")

if __name__ == '__main__':
    main()
