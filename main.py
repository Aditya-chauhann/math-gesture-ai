import streamlit as st
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import google.generativeai as genai

# Configure Google Gemini API
genai.configure(api_key="AIzaSyAEjyGgySlqAg_D6_J6hqkbvbGhGFdStc8")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Streamlit
st.title("MATH GESTURE AI")
st.sidebar.title("Options")

# Sidebar options
start_camera = st.sidebar.button("Start Camera")
submit_canvas = st.sidebar.button("Submit Canvas to AI")

# Initialize webcam variables
if "cap" not in st.session_state:
    st.session_state.cap = None
if "canvas" not in st.session_state:
    st.session_state.canvas = None

# Initialize hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas, drawing, erase_mode):
    fingers, lmlist = info
    current_pos = tuple(map(int, lmlist[8][0:2]))
    if fingers == [1, 1, 0, 0, 0]:  # Drawing mode
        if prev_pos is None or not drawing:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, (255, 0, 255), 10)
        prev_pos = current_pos
        drawing = True
    elif fingers == [1, 1, 1, 1, 1]:  # Erasing mode
        cv2.circle(canvas, current_pos, 50, (0, 0, 0), -1)
    elif fingers == [0, 1, 0, 0, 1]:  # Reset canvas
        canvas[:] = 0
    else:
        drawing = False
    return prev_pos, drawing, canvas

def sendToAI(model, canvas):
    pil_image = Image.fromarray(canvas)
    response = model.generate_content(["Solve this math problem", pil_image])
    return response.text

# Main app logic
if start_camera:
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.cap.set(3, 1280)
        st.session_state.cap.set(4, 720)
        st.session_state.canvas = None
    prev_pos = None
    drawing = False

    st_frame = st.empty()
    while True:
        ret, img = st.session_state.cap.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        if st.session_state.canvas is None:
            st.session_state.canvas = np.zeros_like(img)

        info = getHandInfo(img)
        if info:
            prev_pos, drawing, st.session_state.canvas = draw(
                info, prev_pos, st.session_state.canvas, drawing, erase_mode=False
            )

        image_combined = cv2.addWeighted(img, 0.7, st.session_state.canvas, 0.3, 0)
        st_frame.image(image_combined, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if submit_canvas and st.session_state.canvas is not None:
    st.write("Submitting canvas to AI...")
    ai_response = sendToAI(model, st.session_state.canvas)
    st.write("AI Response:")
    st.write(ai_response)

if st.session_state.cap:
    st.sidebar.button("Stop Camera", on_click=lambda: st.session_state.cap.release())
