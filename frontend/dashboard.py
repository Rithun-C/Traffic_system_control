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
    st.markdown("Upload an image to detect vehicle density and control traffic signals")

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Traffic Detection")

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
            if st.button("🔍 Analyze Traffic", type="primary"):
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

    with col2:
        st.header("Signal Control")

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
            if st.button("🚨 Activate Override", type="primary"):
                result = call_emergency_override_api(override_signal, True)
                if result:
                    st.success(f"Override activated: {override_signal}")
                    time.sleep(1)
                    st.experimental_rerun()

        with col_deactivate:
            if st.button("🔄 Return to Auto"):
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

if __name__ == "__main__":
    main()