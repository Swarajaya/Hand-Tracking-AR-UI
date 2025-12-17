import cv2
import mediapipe as mp
import collections
import time


cap = cv2.VideoCapture(0)
prev_time = 0


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

history_index = collections.deque(maxlen=5)
history_thumb = collections.deque(maxlen=5)
history_middle = collections.deque(maxlen=5)


while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(0,255,255), thickness=2)
            )

            
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            thumb = hand_landmarks.landmark[4]
            wrist = hand_landmarks.landmark[0]

            
            ix, iy = int(index.x*w), int(index.y*h)
            mx, my = int(middle.x*w), int(middle.y*h)
            tx, ty = int(thumb.x*w), int(thumb.y*h)
            wx, wy = int(wrist.x*w), int(wrist.y*h)

           
            history_index.append((ix, iy))
            avg_ix = int(sum([p[0] for p in history_index])/len(history_index))
            avg_iy = int(sum([p[1] for p in history_index])/len(history_index))
            cv2.circle(img, (avg_ix, avg_iy), 10, (0,255,255), -1)

            history_middle.append((mx,my))
            avg_mx = int(sum([p[0] for p in history_middle])/len(history_middle))
            avg_my = int(sum([p[1] for p in history_middle])/len(history_middle))
            cv2.circle(img, (avg_mx, avg_my), 8, (255,0,255), -1)

            history_thumb.append((tx,ty))
            avg_tx = int(sum([p[0] for p in history_thumb])/len(history_thumb))
            avg_ty = int(sum([p[1] for p in history_thumb])/len(history_thumb))
            cv2.circle(img, (avg_tx, avg_ty), 8, (0,255,0), -1)

            
            palm = hand_landmarks.landmark[9]
            px, py = int(palm.x*w), int(palm.y*h)
            cv2.rectangle(img, (px-80, py-80), (px+80, py+80), (0,255,0), 2)
            cv2.putText(img, "PALM DATA 80%", (px-70, py-90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            
            overlay = img.copy()
            cv2.rectangle(overlay, (px-80, py-80), (px+80, py+80), (255,0,0), -1)
            img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)

      
            angle = int(index.x*360)
            cv2.line(img, (wx,wy), (avg_ix, avg_iy), (0,0,255), 2)
            cv2.putText(img, f"rotation {angle}", (px-70, py+110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

           
            cv2.putText(img, f"Index ({avg_ix},{avg_iy})", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(img, f"Middle ({avg_mx},{avg_my})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
            cv2.putText(img, f"Thumb ({avg_tx},{avg_ty})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            
            if index.y < hand_landmarks.landmark[6].y:
                cv2.putText(img, "UI ACTIVE", (10, h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    
    curr_time = time.time()
    fps = 1/(curr_time-prev_time) if prev_time>0 else 0
    prev_time = curr_time
    cv2.putText(img, f"FPS: {int(fps)}", (w-100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("Hand Tracking AR HUD", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
