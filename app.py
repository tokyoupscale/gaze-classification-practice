import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import requests

import av 
import cv2

import numpy as np


st.set_page_config(
    page_title="лучший в мире за работой",
    layout="wide"
)

API_URL = "http://localhost:5252"

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r
    except Exception as e:
        return e
    

st.title("лучший в мире за работой")

api_status = check_api()

col1, col2 = st.columns([1, 3])

with col1:
    if api_status:
        st.success("api fullwork")
    else:
        st.error("api slomalos")
        st.stop()

with col2:
    st.info("погнал")

class Tracker(VideoProcessorBase):
    def __init__(self):
        self.direction = "straight"
        self.confidence = 0
        self.counter = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        self.counter +=1
        # фпс/5
        if self.counter % 5 == 0:
            try:
                _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                r = requests.post(
                    f"{API_URL}/predict",
                    files={"file": buf.tobytes()},
                    timeout=2
                )
                if r.status_code == 200:
                    r = r.json()
                    self.direction = r['predicted_class']
                    self.confidence = int(r['confidence'] * 100)
            except:
                pass

        cv2.putText(img, f"{self.direction.upper()} {self.confidence}%", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
webrtc_streamer(
    key="gaze",
    video_processor_factory=Tracker,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.1.google.com:19302"]}]}
)