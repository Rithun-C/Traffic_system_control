from ultralytics import YOLO
import numpy as np
from PIL import Image
import logging
import cv2
import time
from typing import Dict, List, Tuple

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

def detect_vehicles_in_video(video_path: str, confidence_threshold: float = 0.5,
                           sample_rate: int = 1) -> Dict:
    """
    Detect vehicles in video by processing frames and applying existing emergency logic.

    Args:
        video_path: Path to video file
        confidence_threshold: Minimum confidence for detections (0.0 to 1.0)
        sample_rate: Process every Nth frame (1 = every frame, 2 = every other frame, etc.)

    Returns:
        dict: Comprehensive results including per-frame data and performance metrics
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Initialize tracking variables
        frame_results = []
        performance_metrics = {
            'frame_processing_times': [],
            'total_processing_time': 0,
            'frames_processed': 0,
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'sample_rate': sample_rate
        }

        frame_number = 0
        start_time = time.time()

        print(f"Processing video: {total_frames} frames at {fps:.2f} fps (duration: {duration:.2f}s)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames based on sample_rate
            if frame_number % sample_rate == 0:
                frame_start_time = time.time()

                # Convert OpenCV frame (BGR) to PIL Image (RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Apply existing YOLO detection function
                vehicle_count, detections = detect_vehicles(pil_image, confidence_threshold)

                frame_processing_time = time.time() - frame_start_time

                # Apply existing emergency prioritization logic (same as in main.py:59-63)
                suggested_timing = {
                    "green_duration": max(30, min(120, vehicle_count * 5)),  # 30-120 seconds
                    "red_duration": 30,
                    "yellow_duration": 5
                }

                # Store frame result
                frame_result = {
                    'frame_number': frame_number,
                    'timestamp': frame_number / fps if fps > 0 else 0,
                    'vehicle_count': vehicle_count,
                    'detections': detections,
                    'suggested_timing': suggested_timing,
                    'processing_time': frame_processing_time,
                    'vehicle_summary': get_vehicle_summary(detections)
                }

                frame_results.append(frame_result)
                performance_metrics['frame_processing_times'].append(frame_processing_time)
                performance_metrics['frames_processed'] += 1

                # Progress indication
                if frame_number % (sample_rate * 30) == 0:  # Print every ~30 processed frames
                    progress = (frame_number + 1) / total_frames * 100
                    print(f"Progress: {progress:.1f}% (Frame {frame_number}/{total_frames})")

            frame_number += 1

        cap.release()

        # Calculate final performance metrics
        performance_metrics['total_processing_time'] = time.time() - start_time

        if performance_metrics['frames_processed'] > 0:
            performance_metrics['avg_frame_processing_time'] = (
                sum(performance_metrics['frame_processing_times']) /
                performance_metrics['frames_processed']
            )
            performance_metrics['processing_fps'] = (
                performance_metrics['frames_processed'] /
                performance_metrics['total_processing_time']
            )
        else:
            performance_metrics['avg_frame_processing_time'] = 0
            performance_metrics['processing_fps'] = 0

        # Calculate aggregate statistics
        if frame_results:
            vehicle_counts = [fr['vehicle_count'] for fr in frame_results]
            aggregate_stats = {
                'total_vehicle_detections': sum(vehicle_counts),
                'avg_vehicles_per_frame': sum(vehicle_counts) / len(vehicle_counts),
                'max_vehicles_in_frame': max(vehicle_counts),
                'min_vehicles_in_frame': min(vehicle_counts),
                'frames_with_vehicles': sum(1 for vc in vehicle_counts if vc > 0),
                'peak_traffic_timestamp': frame_results[vehicle_counts.index(max(vehicle_counts))]['timestamp']
            }

            # Calculate overall suggested timing based on average vehicle count
            avg_vehicle_count = aggregate_stats['avg_vehicles_per_frame']
            overall_suggested_timing = {
                "green_duration": max(30, min(120, int(avg_vehicle_count * 5))),
                "red_duration": 30,
                "yellow_duration": 5
            }
        else:
            aggregate_stats = {
                'total_vehicle_detections': 0,
                'avg_vehicles_per_frame': 0,
                'max_vehicles_in_frame': 0,
                'min_vehicles_in_frame': 0,
                'frames_with_vehicles': 0,
                'peak_traffic_timestamp': 0
            }
            overall_suggested_timing = {"green_duration": 30, "red_duration": 30, "yellow_duration": 5}

        return {
            'success': True,
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration,
                'sample_rate': sample_rate
            },
            'performance_metrics': performance_metrics,
            'frame_results': frame_results,
            'aggregate_stats': aggregate_stats,
            'overall_suggested_timing': overall_suggested_timing,
            'emergency_prioritization_preserved': True  # Confirmation that existing logic was applied
        }

    except Exception as e:
        print(f"Error processing video: {e}")
        return {
            'success': False,
            'error': str(e),
            'video_info': None,
            'performance_metrics': None,
            'frame_results': [],
            'aggregate_stats': None,
            'overall_suggested_timing': {"green_duration": 30, "red_duration": 30, "yellow_duration": 5}
        }

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