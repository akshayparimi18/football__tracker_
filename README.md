# ⚽ AI Sports Tracking & Player Classification Pipeline
A robust, real-time computer vision pipeline for tracking and classifying sports players, goalkeepers, referees, and the ball. Built with YOLOv8 and OpenCV, this project goes beyond basic object detection by implementing dynamic color clustering, background filtering, and temporal label smoothing to handle real-world match conditions like shadows, lighting changes, and background crowds.

![Demo](demo.gif)

## ✨ Key Features
- **Advanced Object Detection:** Utilizes `yolov8m.pt` (Medium) for high-accuracy detection of distant players, goalkeepers, referees, and the sports ball, even in wide-angle broadcast footage.
- **Dynamic Auto-Calibration (K-Means):** Automatically determines "Team A" and "Team B" jersey colors in the first 60 frames using K-Means clustering, making the system adaptable to any match without hardcoding color thresholds.
- **Smart "Torso Cropping":** Extracts only the 20% to 60% vertical range of a bounding box to isolate the jersey, completely ignoring heads, hair, and shorts (e.g., preventing a referee's black shorts from being misclassified as a goalkeeper).
- **Pitch Mask Filtering:** Implements a 10x10 pixel patch test at the bottom of every bounding box to ensure the person is standing on the grass, successfully filtering out coaches, fans, and people behind advertising boards.
- **Temporal Label Smoothing:** Uses a 30-frame history buffer and majority voting (mode) for each unique Track ID to eliminate classification flickering caused by momentary lighting changes or shadows.
- **Outlier Detection:** Strictly separates Goalkeepers (dark kits) and Referees (yellow/bright kits) from the main team clusters using tailored HSV brightness and hue thresholds.

## 🛠️ Tech Stack
- **Language:** Python
- **Computer Vision:** OpenCV, Ultralytics YOLOv8
- **Machine Learning / Math:** K-Means Clustering, NumPy

## 🚀 How It Works
1. **Detection:** YOLOv8 detects all persons and sports balls in the frame.
2. **Filtering:** The Pitch Filter checks the HSV values at the feet of each person. If they aren't on the pitch, the detection is dropped.
3. **Cropping & Extraction:** For valid players, the bounding box is cropped to the torso, and the dominant HSV color is extracted.
4. **Classification:** The dominant color is compared against the dynamically calibrated Team A and Team B colors. Outliers are routed to Goalkeeper or Referee logic based on brightness and hue.
5. **Smoothing:** The classification is added to the tracker's history buffer, and the most frequent recent label is drawn on the screen with high-contrast UI elements.
6. **Looping & Display:** The process repeats for every frame of the video.

## 📥 Installation & Usage
1. Clone the repository:
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the tracking script on a video file:
```bash
python predict.py
```

## 👨💻 Author
**Akshay**  
Information Technology Developer specializing in Python and Computer Vision.
