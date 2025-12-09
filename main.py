import itertools
from warnings import catch_warnings

import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import pyvolume
from jaxlib.utils import foreach
from pyvolume import windows
from pycaw.pycaw import AudioUtilities
#setting up which device to change the volume of
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume




# --- CONFIGURATION ---
SKIP_FRAMES = 5
CAM_WIDTH, CAM_HEIGHT = 240, 240  # Lower resolution = faster, im not sure if this actually does anything

# 1. Load the Trained Model
try:
    with open('gesture_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: gesture_model.pkl not found.")
    exit()

# 2. Setup MediaPipe
mp_hands = mp.solutions.hands
# LOWER detection confidence slightly
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
# one of the latter things I added as when working I have my hands on my face when im working like a lot of people
# this meant that it would trigger an event such as muting when im resting on my hand etc
mp_face=mp.solutions.face_detection
face = mp_face.FaceDetection(
    min_detection_confidence=0.6,
    model_selection=0
)

mp_draw = mp.solutions.drawing_utils

# 3. Setup Camera
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

# Variables to manage state
frame_count = 0
last_prediction = -1  # Store the last result to display during skipped frames
prev_time = 0

print("Gesture Controller Active! Press 'q' to quit.")
muteDelay=0 # sets how many frames to wait before next change of volume state to stop constant toggling
unMutedelay=0
intersects=False
while True:
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Flip frame
            frame = cv2.flip(frame, 1)

            #  Only process Heavy Logic every N frames
            if frame_count % SKIP_FRAMES == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                face_results=face.process(rgb_frame)
                if face_results and face_results.detections:
                    for detection in face_results.detections:
                        mp_draw.draw_detection(frame, detection)
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        # VISUALIZATION: Drawing is slow.
                        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)


                        # Normalization in reference to wrist
                        base_x = hand_lms.landmark[0].x
                        base_y = hand_lms.landmark[0].y

                        row = []
                        for lm in hand_lms.landmark:
                            row.append(lm.x - base_x)
                            row.append(lm.y - base_y)

                        # Predict
                        X = np.array([row])
                        last_prediction = model.predict(X)[0]
                        #checking if the hand is on or near the face
                        if  face_results:  # checks if face_results exists as it will not exist if it is obscured by a hand
                            for hand_coordinate in hand_lms.landmark:
                                for detection in face_results.detections:
                                    bbox=detection.location_data.relative_bounding_box
                                    # although this doesn't check the y coordiante,
                                    # if your hand is the the same x range as your face it probably shouldnt trigger anything
                                    if bbox.xmin< hand_coordinate.x < (bbox.xmin+bbox.width):
                                        intersects=True
                                        print("Intersects: ")
                                        break
                                    else:
                                        intersects=False # changes intersect flag despite it being changed at the end of the loop, this is probably my fault
                                        #for writing spaghetti code without using any functions
                                    break

                else:
                    last_prediction = -1  # Reset if no hand found



            # Update frame counter
            frame_count += 1

            # DISPLAY UI
            if intersects ==False:
                if last_prediction == 0 and muteDelay <10 and intersects==False:
                    cv2.putText(frame, "MUTE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    pyvolume.windows.mute()
                    muteDelay=100
                elif last_prediction == 1 and unMutedelay <10 :
                    cv2.putText(frame, "UNMUTE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    pyvolume.windows.mute()
                    unMutedelay=100
                elif last_prediction ==2:
                    pyvolume.windows.volume_up()
                    cv2.putText(frame, "Increasing volume", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                elif last_prediction ==3:
                    pyvolume.windows.volume_down()
                else:
                    cv2.putText(frame, "none", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            # Display FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            #Wait delay is how many frames before the state can be switched
            unMutedelay +=-1
            muteDelay +=-1


            cv2.imshow("Gesture Control", frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        intersects = False
    except:
        print("An error has occurred")