from ultralytics import YOLO

def main():
    # Load the YOLOv8 nano model (pre-trained weights)
    # Using 'yolov8n.pt' as they are the fastest and lightest, which is ideal for CPU-only environments.
    model = YOLO('yolov8n.pt')

    # Train the model
    # data: Path to your data.yaml file (downloaded from Roboflow and correctly configured)
    # epochs: Set to 25 as requested
    # imgsz: Image size for training (default is 640). Adjust if your dataset is exported at a different resolution.
    # batch: Batch size. A smaller batch like 8 or 16 is recommended for CPU training to manage system memory.
    # device: Explicitly set to 'cpu' to prevent CUDA errors on machines without an NVIDIA GPU.
    # workers: Set to 0 or 2 for CPU to prevent multiprocessing issues, especially on Windows.
    print("Starting YOLOv8 training on CPU...")
    results = model.train(
        data='data.yaml',
        epochs=25,
        imgsz=640,
        batch=16,
        device='cpu',
        workers=2,
        project='runs/train',
        name='football_tracking_model'
    )

    print("Training complete! Your trained model weights are saved at:")
    print("runs/train/football_tracking_model/weights/best.pt")

if __name__ == '__main__':
    # Required for Windows multiprocessing safety
    main()
