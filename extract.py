from ultralytics import YOLO

# Load the custom model you trained for the tracker (best.pt)
# I changed this from 'yolov8n.pt' so it correctly IDs your 4 specific classes!
model = YOLO('runs/train/football_tracking_model6/weights/best.pt') 

# Run the tracker on your video (stream=True is best for videos so it doesn't overload memory)
results = model('input_video.mp4', stream=True)

# Loop through every frame in your 843-frame video
for frame_number, result in enumerate(results):
    print(f"\n--- Processing Frame {frame_number} ---")
    
    # Grab all the bounding boxes detected in this specific frame
    boxes = result.boxes
    
    # Loop through each individual box
    for box in boxes:
        # Extract x_center, y_center, width, and height
        x_c, y_c, w, h = box.xywh[0]
        
        # Extract the class ID (e.g., 0 for ball, 1 for goalkeeper, 2 for player, 3 for referee)
        class_id = int(box.cls[0])
        
        # Extract the confidence score (how sure the model is)
        confidence = float(box.conf[0])
        
        # Print the data cleanly to your terminal
        print(f"Class: {class_id} | Center: ({x_c:.1f}, {y_c:.1f}) | Size: {w:.1f}x{h:.1f} | Conf: {confidence:.2f}")
