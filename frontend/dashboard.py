import streamlit as st
import requests
import json
from PIL import Image
import time

# Backend API configuration
BACKEND_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Traffic Control System",
    page_icon="🚦",
    layout="wide"
)

def get_signal_color(signal):
    """Return appropriate color for traffic signal display"""
    colors = {
        "RED": "🔴",
        "YELLOW": "🟡",
        "GREEN": "🟢"
    }
    return colors.get(signal, "⚪")

def call_detect_traffic_api(image_file):
    """Call the backend API to detect traffic"""
    try:
        files = {"file": ("image.jpg", image_file, "image/jpeg")}
        response = requests.post(f"{BACKEND_URL}/detect_traffic", files=files)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API. Make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def call_detect_traffic_video_api(video_file, sample_rate=1, confidence_threshold=0.5):
    """Call the backend API to detect traffic in video"""
    try:
        files = {"file": (video_file.name, video_file, "video/mp4")}
        data = {
            "sample_rate": sample_rate,
            "confidence_threshold": confidence_threshold
        }

        response = requests.post(f"{BACKEND_URL}/detect_traffic_video", files=files, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API. Make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def call_emergency_override_api(signal_state, override):
    """Call the backend API for emergency override"""
    try:
        data = {
            "signal_state": signal_state,
            "override": override
        }
        response = requests.post(f"{BACKEND_URL}/emergency_override", json=data)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API. Make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def get_signal_status():
    """Get current signal status from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/signal_status")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Main dashboard
def main():
    st.title("🚦 Traffic Control System Dashboard")
    st.markdown("Upload an image or video to detect vehicle density and control traffic signals")

    # Create tabs for different modes
    tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Analysis", "🚦 Signal Control"])

    with tab1:
        image_analysis_tab()

    with tab2:
        video_analysis_tab()

    with tab3:
        signal_control_tab()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <small>Traffic Control System Prototype | FastAPI + Streamlit + YOLO</small>
        </div>
        """,
        unsafe_allow_html=True
    )

def image_analysis_tab():
    """Image analysis functionality (original)"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Traffic Detection from Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload traffic image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing vehicles for traffic density analysis"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Traffic Image", use_column_width=True)

            # Analyze button
            if st.button("🔍 Analyze Traffic", type="primary", key="image_analyze"):
                with st.spinner("Detecting vehicles..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    result = call_detect_traffic_api(uploaded_file)

                if result:
                    # Display results
                    st.success(f"Analysis complete!")

                    # Metrics display
                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    with metric_col1:
                        st.metric("Vehicles Detected", result['vehicle_count'])

                    with metric_col2:
                        st.metric("Current Signal", result['current_signal'])

                    with metric_col3:
                        green_duration = result['suggested_timing']['green_duration']
                        st.metric("Suggested Green Time", f"{green_duration}s")

                    # Detailed detection results
                    if result['detections']:
                        st.subheader("Detection Details")

                        # Vehicle summary
                        vehicle_summary = {}
                        for detection in result['detections']:
                            vehicle_type = detection['class']
                            vehicle_summary[vehicle_type] = vehicle_summary.get(vehicle_type, 0) + 1

                        summary_cols = st.columns(4)
                        vehicle_icons = {'car': '🚗', 'truck': '🚛', 'bus': '🚌', 'motorcycle': '🏍️'}

                        for i, (vehicle_type, count) in enumerate(vehicle_summary.items()):
                            with summary_cols[i % 4]:
                                icon = vehicle_icons.get(vehicle_type, '🚗')
                                st.metric(f"{icon} {vehicle_type.title()}", count)

                        # Show detection table
                        with st.expander("View All Detections"):
                            for i, detection in enumerate(result['detections']):
                                st.write(f"{i+1}. **{detection['class'].title()}** - Confidence: {detection['confidence']}")

def video_analysis_tab():
    """Video analysis functionality (new)"""
    st.header("Traffic Detection from Video")
    st.markdown("Upload a video file to analyze traffic patterns over time and generate performance graphs.")

    # Video uploader
    uploaded_video = st.file_uploader(
        "Upload traffic video",
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Upload a video file containing traffic for comprehensive analysis"
    )

    if uploaded_video is not None:
        # Display video info
        st.video(uploaded_video)

        # Processing parameters
        st.subheader("Processing Parameters")
        param_col1, param_col2 = st.columns(2)

        with param_col1:
            sample_rate = st.slider(
                "Frame Sample Rate",
                min_value=1, max_value=10, value=2,
                help="Process every Nth frame (higher = faster but less detailed)"
            )

        with param_col2:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1, max_value=0.9, value=0.5, step=0.1,
                help="Minimum confidence for vehicle detections"
            )

        # Analysis button
        if st.button("🎬 Analyze Video", type="primary", key="video_analyze"):
            with st.spinner("Processing video... This may take a few minutes."):
                # Reset file pointer
                uploaded_video.seek(0)
                result = call_detect_traffic_video_api(
                    uploaded_video,
                    sample_rate=sample_rate,
                    confidence_threshold=confidence_threshold
                )

            if result and result.get('success'):
                st.success("🎉 Video analysis complete!")

                # Display key metrics
                st.subheader("Analysis Results")

                # Main metrics row
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

                video_info = result['video_info']
                aggregate_stats = result['aggregate_stats']
                performance_metrics = result['performance_metrics']

                with metrics_col1:
                    st.metric("Video Duration", f"{video_info['duration']:.1f}s")

                with metrics_col2:
                    st.metric("Frames Processed", f"{performance_metrics['frames_processed']:,}")

                with metrics_col3:
                    st.metric("Total Vehicles", aggregate_stats['total_vehicle_detections'])

                with metrics_col4:
                    st.metric("Avg Vehicles/Frame", f"{aggregate_stats['avg_vehicles_per_frame']:.1f}")

                # Performance metrics
                st.subheader("Processing Performance")
                perf_col1, perf_col2, perf_col3 = st.columns(3)

                with perf_col1:
                    st.metric("Processing Time", f"{performance_metrics['total_processing_time']:.1f}s")

                with perf_col2:
                    processing_fps = performance_metrics['processing_fps']
                    st.metric("Processing FPS", f"{processing_fps:.1f}")

                with perf_col3:
                    realtime = "✅ Yes" if performance_metrics['realtime_capable'] else "⚠️ No"
                    st.metric("Real-time Capable", realtime)

                # Traffic signal recommendations
                st.subheader("Traffic Signal Recommendations")
                timing = result['overall_suggested_timing']
                timing_col1, timing_col2, timing_col3 = st.columns(3)

                with timing_col1:
                    st.metric("🟢 Green Duration", f"{timing['green_duration']}s")

                with timing_col2:
                    st.metric("🔴 Red Duration", f"{timing['red_duration']}s")

                with timing_col3:
                    st.metric("🟡 Yellow Duration", f"{timing['yellow_duration']}s")

                # Performance graphs
                st.subheader("📊 Performance Graphs Generated")
                graphs_dir = result.get('graphs_directory')
                if graphs_dir:
                    st.success(f"📁 Graphs saved to: `{graphs_dir}`")

                    st.markdown("**Generated visualizations:**")
                    st.markdown("• Vehicle count timeline over video duration")
                    st.markdown("• Processing performance metrics and efficiency analysis")
                    st.markdown("• Vehicle type distribution and trends")
                    st.markdown("• Traffic signal timing recommendations")

                    st.info("💡 Check the experiments folder in your project directory to view the generated graph files.")
                else:
                    st.warning("Graphs directory not found in response")

                # Sample frame results
                sample_frames = result.get('sample_frame_results', [])
                if sample_frames:
                    with st.expander("View Sample Frame Analysis (First 10 Frames)"):
                        for i, frame_data in enumerate(sample_frames):
                            st.write(f"**Frame {frame_data['frame_number']}** (t={frame_data['timestamp']:.1f}s): "
                                   f"{frame_data['vehicle_count']} vehicles detected")

def signal_control_tab():
    """Signal control functionality"""
    st.header("Traffic Signal Control")

    # Current status
    status = get_signal_status()
    if status:
        current_signal = status.get('current_signal', 'UNKNOWN')
        emergency_mode = status.get('emergency_override', False)
        auto_mode = status.get('auto_mode', True)

        # Signal display
        st.markdown(f"### Current Signal: {get_signal_color(current_signal)} {current_signal}")

        # Mode display
        if emergency_mode:
            st.warning("🚨 Emergency Override Active")
        else:
            st.info("🤖 Automatic Mode")

    st.markdown("---")

    # Emergency override section
    st.subheader("Emergency Override")

    # Signal selection
    override_signal = st.selectbox(
        "Select Signal State:",
        ["RED", "YELLOW", "GREEN"],
        help="Choose the signal state for emergency override"
    )

    # Override buttons
    col_activate, col_deactivate = st.columns(2)

    with col_activate:
        if st.button("🚨 Activate Override", type="primary", key="activate_override"):
            result = call_emergency_override_api(override_signal, True)
            if result:
                st.success(f"Override activated: {override_signal}")
                time.sleep(1)
                st.experimental_rerun()

    with col_deactivate:
        if st.button("🔄 Return to Auto", key="deactivate_override"):
            result = call_emergency_override_api("GREEN", False)
            if result:
                st.success("Returned to automatic mode")
                time.sleep(1)
                st.experimental_rerun()

    # Auto-refresh status
    if st.checkbox("Auto-refresh status", value=True):
        # Refresh every 5 seconds
        time.sleep(5)
        st.experimental_rerun()

if __name__ == "__main__":
    main()