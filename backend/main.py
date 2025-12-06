from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from PIL import Image
import sys
import os

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.yolo_detector import detect_vehicles, detect_vehicles_in_video
from model.graph_generator import generate_all_performance_graphs

app = FastAPI(title="Traffic Control System", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory traffic signal state
traffic_signal_state = {
    "current_signal": "RED",  # RED, YELLOW, GREEN
    "emergency_override": False,
    "auto_mode": True
}

class EmergencyOverrideRequest(BaseModel):
    signal_state: str  # RED, YELLOW, GREEN
    override: bool

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Traffic Control System API"}

@app.post("/detect_traffic")
async def detect_traffic(file: UploadFile = File(...)):
    """
    Detect vehicles in uploaded image and return count.
    Based on vehicle count, suggest optimal signal timing.
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Detect vehicles using YOLO
        vehicle_count, detections = detect_vehicles(image)

        # Simple logic for traffic signal timing based on vehicle count
        suggested_timing = {
            "green_duration": max(30, min(120, vehicle_count * 5)),  # 30-120 seconds
            "red_duration": 30,
            "yellow_duration": 5
        }

        return {
            "vehicle_count": vehicle_count,
            "detections": detections,
            "suggested_timing": suggested_timing,
            "current_signal": traffic_signal_state["current_signal"],
            "emergency_override": traffic_signal_state["emergency_override"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect_traffic_video")
async def detect_traffic_video(file: UploadFile = File(...), sample_rate: int = 1, confidence_threshold: float = 0.5):
    """
    Detect vehicles in uploaded video and generate performance graphs.
    Applies existing emergency prioritization logic per-frame.

    Args:
        file: Video file upload
        sample_rate: Process every Nth frame (1 = every frame, 2 = every other frame, etc.)
        confidence_threshold: Minimum confidence for detections (0.0 to 1.0)
    """
    try:
        # Validate file type
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File must be a video")

        # Create temporary directory for video processing
        temp_dir = "temp_videos"
        os.makedirs(temp_dir, exist_ok=True)

        # Save uploaded video temporarily
        temp_video_path = os.path.join(temp_dir, f"temp_{file.filename}")

        try:
            with open(temp_video_path, "wb") as buffer:
                video_data = await file.read()
                buffer.write(video_data)

            print(f"Processing video: {file.filename} (size: {len(video_data)} bytes)")
            print(f"Sample rate: {sample_rate}, Confidence threshold: {confidence_threshold}")

            # Process video using YOLO detector with existing emergency logic
            video_results = detect_vehicles_in_video(
                video_path=temp_video_path,
                confidence_threshold=confidence_threshold,
                sample_rate=sample_rate
            )

            if not video_results['success']:
                raise HTTPException(
                    status_code=500,
                    detail=f"Video processing failed: {video_results.get('error', 'Unknown error')}"
                )

            # Generate performance graphs and save to experiments/<timestamp>/graphs/
            graphs_directory = generate_all_performance_graphs(
                video_results,
                experiment_name=f"video_{file.filename}"
            )

            # Prepare response with comprehensive results
            response_data = {
                "success": True,
                "video_filename": file.filename,
                "video_info": video_results['video_info'],
                "aggregate_stats": video_results['aggregate_stats'],
                "overall_suggested_timing": video_results['overall_suggested_timing'],
                "performance_metrics": {
                    "total_processing_time": video_results['performance_metrics']['total_processing_time'],
                    "frames_processed": video_results['performance_metrics']['frames_processed'],
                    "avg_frame_processing_time": video_results['performance_metrics']['avg_frame_processing_time'],
                    "processing_fps": video_results['performance_metrics']['processing_fps'],
                    "realtime_capable": video_results['performance_metrics']['processing_fps'] >= video_results['video_info']['fps']
                },
                "graphs_directory": graphs_directory,
                "emergency_prioritization_applied": video_results['emergency_prioritization_preserved'],
                "current_signal": traffic_signal_state["current_signal"],
                "emergency_override": traffic_signal_state["emergency_override"],

                # Include sample frame results (first 10 frames for response size management)
                "sample_frame_results": video_results['frame_results'][:10] if video_results['frame_results'] else [],
                "total_frames_analyzed": len(video_results['frame_results']),

                # Processing parameters
                "processing_parameters": {
                    "sample_rate": sample_rate,
                    "confidence_threshold": confidence_threshold
                }
            }

            print(f"✅ Video processing completed successfully!")
            print(f"📊 Processed {video_results['performance_metrics']['frames_processed']} frames")
            print(f"📈 Generated graphs in: {graphs_directory}")

            return response_data

        finally:
            # Clean up temporary video file
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                    print(f"Cleaned up temporary file: {temp_video_path}")
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temporary file: {cleanup_error}")

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in video processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/emergency_override")
async def emergency_override(request: EmergencyOverrideRequest):
    """
    Emergency override to manually control traffic signal state.
    """
    try:
        valid_signals = ["RED", "YELLOW", "GREEN"]

        if request.signal_state not in valid_signals:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid signal state. Must be one of: {valid_signals}"
            )

        # Update traffic signal state
        traffic_signal_state["current_signal"] = request.signal_state
        traffic_signal_state["emergency_override"] = request.override
        traffic_signal_state["auto_mode"] = not request.override

        return {
            "message": "Traffic signal updated successfully",
            "current_signal": traffic_signal_state["current_signal"],
            "emergency_override": traffic_signal_state["emergency_override"],
            "auto_mode": traffic_signal_state["auto_mode"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating signal: {str(e)}")

@app.get("/signal_status")
async def get_signal_status():
    """Get current traffic signal status"""
    return traffic_signal_state

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)