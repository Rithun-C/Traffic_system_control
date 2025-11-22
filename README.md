# Traffic Control System Prototype

A basic traffic density detection and control system using OpenCV, YOLO, FastAPI, and Streamlit.

## 🎯 Features

- **Traffic Detection**: Upload images to detect vehicle density using YOLOv8
- **Emergency Override**: Manually control traffic signal states
- **Real-time Dashboard**: Streamlit interface for monitoring and control
- **REST API**: FastAPI backend with traffic detection and signal control endpoints

## 📁 Project Structure

```
traffic-control-system/
│
├── backend/
│   └── main.py              # FastAPI server with detection and override endpoints
│
├── frontend/
│   └── dashboard.py         # Streamlit dashboard for UI
│
├── model/
│   └── yolo_detector.py     # YOLO vehicle detection logic
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ File Explanations

### `backend/main.py`
- FastAPI server with CORS enabled
- `/detect_traffic`: Accepts image uploads, runs YOLO detection, returns vehicle count and timing suggestions
- `/emergency_override`: Allows manual control of traffic signal state
- `/signal_status`: Returns current traffic signal status
- Stores signal state in memory (no database required)

### `model/yolo_detector.py`
- Uses ultralytics YOLOv8n (nano) model for fast inference
- Detects vehicles: cars, trucks, buses, motorcycles
- Returns vehicle count and detection details with bounding boxes
- Automatically downloads YOLO model on first use

### `frontend/dashboard.py`
- Streamlit web interface for the system
- Image upload and traffic analysis
- Emergency override controls
- Real-time signal status display
- Auto-refresh functionality

## ⚙️ Installation

1. **Clone or create the project structure**
   ```bash
   # Navigate to your project directory
   cd traffic-control-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: First run will download YOLOv8n model (~6MB) automatically.

## 🚀 How to Run

### 1. Start the Backend (Terminal 1)
```bash
uvicorn backend.main:app --reload
```
- API will be available at `http://localhost:8000`
- Interactive docs at `http://localhost:8000/docs`

### 2. Start the Frontend (Terminal 2)
```bash
streamlit run frontend/dashboard.py
```
- Dashboard will open at `http://localhost:8501`

## 📱 How to Use

### Traffic Detection
1. Open the Streamlit dashboard
2. Upload an image containing vehicles (JPG/PNG)
3. Click "Analyze Traffic"
4. View vehicle count and detection results
5. See suggested green light timing based on density

### Emergency Override
1. In the dashboard, go to "Signal Control" section
2. Select desired signal state (RED/YELLOW/GREEN)
3. Click "Activate Override" for emergency control
4. Click "Return to Auto" to resume automatic mode

## 🧪 Testing the System

### Sample Test Images
- Use any traffic images with visible vehicles
- Highway scenes work well for multiple vehicle detection
- Single car images for basic functionality testing

### API Testing (Optional)
```bash
# Test health check
curl http://localhost:8000/

# Test signal status
curl http://localhost:8000/signal_status
```

## 🔧 System Behavior

- **Auto Mode**: Signal timing based on detected vehicle density
- **Emergency Mode**: Manual override of signal state
- **Detection**: Focuses on cars, trucks, buses, motorcycles
- **Confidence**: Minimum 50% confidence for vehicle detection
- **Timing**: Green light duration scales with vehicle count (30-120 seconds)

## 📝 Limitations (Prototype)

- No persistent storage (signals reset on restart)
- No real traffic light hardware integration
- Basic YOLO model (can be upgraded to YOLOv8s/m/l for better accuracy)
- No user authentication
- Single intersection simulation

## 🛠️ Troubleshooting

### Common Issues

1. **"Cannot connect to backend API"**
   - Ensure FastAPI server is running on port 8000
   - Check if port 8000 is available

2. **YOLO model download issues**
   - Ensure internet connection for first-time model download
   - Model will be cached locally after first download

3. **Image upload errors**
   - Ensure image file is JPG/PNG format
   - Check image file size (recommended < 10MB)

### Dependencies Issues
```bash
# If torch installation fails, try:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For M1/M2 Macs, use:
pip install torch torchvision
```

## 🚀 Future Enhancements

- Real-time video stream processing
- Database integration for historical data
- Multiple intersection management
- Advanced traffic optimization algorithms
- Mobile app interface
- Hardware integration with actual traffic lights

---

**Built with**: Python, FastAPI, Streamlit, YOLOv8, OpenCV