import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)] 
color_labels = ["Blue", "Green", "Red", "Yellow", "Erase"]
current_color = colors[0]
brush_thickness = 5
eraser_thickness = 50

canvas = None

cap = cv2.VideoCapture(0)

cv2.namedWindow('Hand Gesture Drawing', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Hand Gesture Drawing', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_x, prev_y = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_index = int(hand_landmarks.landmark[8].x * w)
            y_index = int(hand_landmarks.landmark[8].y * h)
            x_middle = int(hand_landmarks.landmark[12].x * w)
            y_middle = int(hand_landmarks.landmark[12].y * h)

            fingers = []
            for i in [8, 12]:
                if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            if fingers == [1, 1]:  
                if y_index < 50:
                    if 50 < x_index < 110:
                        current_color = colors[0]
                    elif 160 < x_index < 220:
                        current_color = colors[1]
                    elif 270 < x_index < 330:
                        current_color = colors[2]
                    elif 380 < x_index < 440:
                        current_color = colors[3]
                    elif 490 < x_index < 550:
                        current_color = colors[4] 
                    prev_x, prev_y = None, None 
            elif fingers == [1, 0]:  
                if current_color == (255, 255, 255): 
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x_index, y_index), (0, 0, 0), eraser_thickness)
                    else:
                        cv2.circle(canvas, (x_index, y_index), eraser_thickness, (0, 0, 0), -1)
                else:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x_index, y_index), current_color, brush_thickness)
                    else:
                        cv2.circle(canvas, (x_index, y_index), brush_thickness, current_color, -1)
                prev_x, prev_y = x_index, y_index
            else:
                prev_x, prev_y = None, None  

    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    button_positions = [50, 160, 270, 380, 490]
    for i in range(len(colors)):
        cv2.rectangle(frame, (button_positions[i], 0), (button_positions[i] + 60, 50), colors[i], -1)
        cv2.putText(frame, color_labels[i], (button_positions[i] + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1 if i != 4 else 1)

    cv2.imshow('Hand Gesture Drawing', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
