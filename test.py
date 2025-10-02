from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/aug_21/weights/best(125epochs).pt")
model.predict(source="test.jpg", show=True)