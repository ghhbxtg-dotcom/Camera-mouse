import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

mp_draw = mp.solutions.drawing_utils

smooth = 7
prev_x, prev_y = 0, 0

click_delay = 0.45
last_click = 0

pTime = 0

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            lm = hand.landmark

            index_tip = lm[8]
            middle_tip = lm[12]
            thumb_tip = lm[4]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            cv2.circle(frame, (x, y), 10, (0,255,0), -1)

            screen_x = np.interp(x, (0,w), (0,screen_w))
            screen_y = np.interp(y, (0,h), (0,screen_h))

            curr_x = prev_x + (screen_x - prev_x) / smooth
            curr_y = prev_y + (screen_y - prev_y) / smooth

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            index_dist = np.hypot(
                index_tip.x - thumb_tip.x,
                index_tip.y - thumb_tip.y
            )

            middle_dist = np.hypot(
                middle_tip.x - thumb_tip.x,
                middle_tip.y - thumb_tip.y
            )

            if index_dist < 0.05 and time.time() - last_click > click_delay:

                pyautogui.click()

                cv2.putText(
                    frame,
                    "LKM",
                    (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,255,0),
                    3
                )

                last_click = time.time()

            elif middle_dist < 0.05 and time.time() - last_click > click_delay:

                pyautogui.rightClick()

                cv2.putText(
                    frame,
                    "PKM",
                    (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,0,255),
                    3
                )

                last_click = time.time()

    cTime = time.time()
    fps = 1/(cTime - pTime) if cTime != pTime else 0
    pTime = cTime

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )

    cv2.imshow("camera", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()