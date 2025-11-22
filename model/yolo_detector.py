from ultralytics import YOLO
import numpy as np
from PIL import Image
import logging

# Suppress YOLO verbose output
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Vehicle classes from COCO dataset that YOLO can detect
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# Global model variable to avoid reloading
_model = None

def load_yolo_model(model_path='yolov8n.pt'):
    """
    Load YOLO model. Uses YOLOv8 nano for faster inference.
    Model will be downloaded automatically on first use.
    """
    global _model
    if _model is None:
        try:
            _model = YOLO(model_path)
            print(f"YOLO model {model_path} loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise e
    return _model

def detect_vehicles(image, confidence_threshold=0.5):
    """
    Detect vehicles in the given image using YOLO.

    Args:
        image: PIL Image object
        confidence_threshold: Minimum confidence for detections (0.0 to 1.0)

    Returns:
        tuple: (vehicle_count, detections_list)
    """
    try:
        # Load model if not already loaded
        model = load_yolo_model()

        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image

        # Run inference
        results = model(image_array, verbose=False)

        vehicle_detections = []
        vehicle_count = 0

        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Check if it's a vehicle class and meets confidence threshold
                    if class_id in VEHICLE_CLASSES and confidence >= confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        detection = {
                            'class': VEHICLE_CLASSES[class_id],
                            'confidence': round(confidence, 2),
                            'bbox': {
                                'x1': round(x1, 1),
                                'y1': round(y1, 1),
                                'x2': round(x2, 1),
                                'y2': round(y2, 1)
                            }
                        }

                        vehicle_detections.append(detection)
                        vehicle_count += 1

        return vehicle_count, vehicle_detections

    except Exception as e:
        print(f"Error in vehicle detection: {e}")
        # Return default values in case of error
        return 0, []

def get_vehicle_summary(detections):
    """
    Get a summary of detected vehicles by type.

    Args:
        detections: List of detection dictionaries

    Returns:
        dict: Summary of vehicle counts by type
    """
    summary = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}

    for detection in detections:
        vehicle_type = detection['class']
        if vehicle_type in summary:
            summary[vehicle_type] += 1

    return summary

# Test function for standalone use
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        try:
            image = Image.open(image_path)
            count, detections = detect_vehicles(image)
            print(f"Vehicle count: {count}")
            print(f"Detections: {detections}")
            print(f"Summary: {get_vehicle_summary(detections)}")
        except Exception as e:
            print(f"Error testing with image {image_path}: {e}")
    else:
        print("YOLO detector module loaded successfully")
        print("Usage: python yolo_detector.py <image_path>")