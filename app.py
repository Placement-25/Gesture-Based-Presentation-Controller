import cv2
import os
import numpy as np
import streamlit as st
from HandTracker import HandDetector
from dottedline import drawrect

# Constants
WIDTH, HEIGHT = 1280, 720
FRAMES_FOLDER = "Images"
HS, WS = int(120 * 1.2), int(213 * 1.2)
GE_THRESH_Y = 400
GE_THRESH_X = 750
DELAY = 15

# Session State Initialization
if 'slide_num' not in st.session_state:
    st.session_state.slide_num = 0
    st.session_state.gest_done = False
    st.session_state.gest_counter = 0
    st.session_state.annotations = [[]]
    st.session_state.annot_num = 0
    st.session_state.annot_start = False
    st.session_state.drawing = False
    st.session_state.pointer = False
    st.session_state.color = (0, 0, 255)
    st.session_state.gesture_text = ""

# Sidebar Controls
st.title("🖐️ Gesture-Controlled Presentation")
st.sidebar.title("🧰 Controls")
if st.sidebar.button("⏮ Previous Slide"):
    if st.session_state.slide_num > 0:
        st.session_state.slide_num -= 1
        st.session_state.annotations = [[]]
        st.session_state.annot_num = 0

if st.sidebar.button("⏭ Next Slide"):
    if st.session_state.slide_num < len(os.listdir(FRAMES_FOLDER)) - 1:
        st.session_state.slide_num += 1
        st.session_state.annotations = [[]]
        st.session_state.annot_num = 0

if st.sidebar.button("🧹 Clear Annotations"):
    st.session_state.annotations = [[]]
    st.session_state.annot_num = 0

if st.sidebar.button("🎨 Toggle Drawing"):
    st.session_state.drawing = not st.session_state.drawing

if st.sidebar.button("👆 Enable Pointer"):
    st.session_state.pointer = not st.session_state.pointer

st.sidebar.markdown("### Pen Color")
pen_color = st.sidebar.radio("Choose Color", ["Red", "Green", "Blue", "Black"])
color_map = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Black": (0, 0, 0)
}
st.session_state.color = color_map[pen_color]

# Load Slides
path_imgs = sorted(os.listdir(FRAMES_FOLDER), key=len)

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

detector = HandDetector(detectionCon=0.8, maxHands=1)
FRAME_WINDOW = st.image(
    np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), channels="BGR")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.warning("Failed to access camera.")
        break

    frame = cv2.flip(frame, 1)
    path_full_image = os.path.join(
        FRAMES_FOLDER, path_imgs[st.session_state.slide_num])
    slide_current = cv2.imread(path_full_image)
    slide_current = cv2.resize(slide_current, (WIDTH, HEIGHT))

    hands, frame = detector.findHands(frame)
    drawrect(frame, (WIDTH, 0), (GE_THRESH_X, GE_THRESH_Y),
             (0, 255, 0), 5, 'dotted')
    gesture_detected = ""

    if hands and not st.session_state.gest_done:
        hand = hands[0]
        cx, cy = hand["center"]
        lm_list = hand["lmList"]
        fingers = detector.fingersUp(hand)

        x_val = int(np.interp(lm_list[8][0], [WIDTH//2, WIDTH], [0, WIDTH]))
        y_val = int(np.interp(lm_list[8][1], [150, HEIGHT - 150], [0, HEIGHT]))
        index_fing = x_val, y_val

        if cy < GE_THRESH_Y and cx > GE_THRESH_X:
            st.session_state.annot_start = False

            if fingers == [1, 0, 0, 0, 0]:  # Prev
                gesture_detected = "Previous Slide 👈"
                if st.session_state.slide_num > 0:
                    st.session_state.slide_num -= 1
                    st.session_state.annotations = [[]]
                    st.session_state.annot_num = 0
                    st.session_state.gest_done = True

            elif fingers == [0, 0, 0, 0, 1]:  # Next
                gesture_detected = "Next Slide 👉"
                if st.session_state.slide_num < len(path_imgs) - 1:
                    st.session_state.slide_num += 1
                    st.session_state.annotations = [[]]
                    st.session_state.annot_num = 0
                    st.session_state.gest_done = True

            elif fingers == [1, 1, 1, 1, 1]:  # Clear
                gesture_detected = "Clear All 🧼"
                st.session_state.annotations = [[]]
                st.session_state.annot_num = 0
                st.session_state.gest_done = True

        elif fingers == [0, 1, 1, 0, 0]:  # Pointer
            gesture_detected = "Pointer Mode 👆"
            st.session_state.pointer = True
            cv2.circle(slide_current, index_fing, 6, (0, 0, 255), cv2.FILLED)

        elif fingers == [0, 1, 0, 0, 0]:  # Draw
            gesture_detected = "Drawing ✍️"
            if not st.session_state.annot_start:
                st.session_state.annot_start = True
                st.session_state.annot_num += 1
                st.session_state.annotations.append([])
            st.session_state.annotations[st.session_state.annot_num].append(
                index_fing)
            cv2.circle(slide_current, index_fing, 6,
                       st.session_state.color, cv2.FILLED)

        elif fingers == [0, 1, 1, 1, 0]:  # Erase Last
            gesture_detected = "Erase ✂️"
            if st.session_state.annot_num >= 0:
                st.session_state.annotations.pop(-1)
                st.session_state.annot_num -= 1
                st.session_state.gest_done = True

    if st.session_state.gest_done:
        st.session_state.gest_counter += 1
        if st.session_state.gest_counter > DELAY:
            st.session_state.gest_counter = 0
            st.session_state.gest_done = False

    for annotation in st.session_state.annotations:
        for j in range(1, len(annotation)):
            cv2.line(slide_current, annotation[j - 1],
                     annotation[j], st.session_state.color, 6)

    # Small cam overlay
    img_small = cv2.resize(frame, (WS, HS))
    h, w, _ = slide_current.shape
    slide_current[h - HS:h, w - WS:w] = img_small

    # Overlay gesture text
    if gesture_detected:
        cv2.putText(slide_current, gesture_detected, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 3)

    FRAME_WINDOW.image(slide_current, channels="BGR")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()