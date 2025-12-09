import cv2

import mediapipe as mp

import csv

# 1. Setup

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)

mp_draw = mp.solutions.drawing_utils

# 2. Create/Open CSV File

csv_file = open('gesture_data.csv', 'a', newline='')

writer = csv.writer(csv_file)

cap = cv2.VideoCapture(0)

print("Press 0 for fist 1 for palm 2 for index up 3 for index down,  to save frame data. Press 'ESC' to quit.")

while True:

    ret, frame = cap.read() #ret for recording or not, frame is the frame object

    if not ret: break

    # Flip for mirror view

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_lms in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Check for key press

            key = cv2.waitKey(1)

            if key == 48:  # '0' Key this is a fist

                label = 0

            elif key == 49:  # '1' Key #this is a palm

                label = 1

            elif key == 50: #2 key, index  finger pointing up
                label = 2

            elif key ==51: # 3 key, index finger pointing up
                label = 3

            else:

                label = -1

            # Save Data if key pressed

            if label != -1:

                # Normalize: Make relative to Wrist (Point 0)

                base_x = hand_lms.landmark[0].x

                base_y = hand_lms.landmark[0].y

                row = [label]

                for lm in hand_lms.landmark:
                    # Store relative X, Y

                    row.append(lm.x - base_x)

                    row.append(lm.y - base_y)

                writer.writerow(row)

                print(f"Saved sample for Class {label}")

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) == 27:  # ESC key

        break

cap.release()

csv_file.close()

cv2.destroyAllWindows()