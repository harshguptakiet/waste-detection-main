from pathlib import Path
import streamlit as st
import helper
import settings

st.set_page_config(page_title="Waste Detection")

st.sidebar.title("Detect Console")
model_path = Path(settings.DETECTION_MODEL)

st.title("Intelligent Waste Segregation System")
st.write("Start detecting objects in the webcam stream by clicking the button below. To stop the detection, click stop button in the top right corner of the webcam stream.")

# Styling for detected waste types
st.markdown("""
<style>
    .stRecyclable { background-color: rgba(233,192,78,255); padding: 1rem 0.75rem; margin-bottom: 1rem; border-radius: 0.5rem; font-size:18px !important; }
    .stNonRecyclable { background-color: rgba(94,128,173,255); padding: 1rem 0.75rem; margin-bottom: 1rem; border-radius: 0.5rem; font-size:18px !important; }
    .stHazardous { background-color: rgba(194,84,85,255); padding: 1rem 0.75rem; margin-bottom: 1rem; border-radius: 0.5rem; font-size:18px !important; }
</style>
""", unsafe_allow_html=True)

# Load model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Webcam detection
st.subheader("Live Webcam Detection")
helper.play_webcam(model)

# Video upload detection
st.subheader("Analyze Uploaded Video")
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
if uploaded_video:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    st.video("temp_video.mp4")

    if st.button("Analyze Uploaded Video"):
        helper.analyze_video("temp_video.mp4", model)

st.sidebar.markdown("This is a demo of the waste detection model.", unsafe_allow_html=True)
