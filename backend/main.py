from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from PIL import Image
import sys
import os

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.yolo_detector import detect_vehicles

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