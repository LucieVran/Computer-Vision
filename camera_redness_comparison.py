
import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access the camera")
    exit()

previous_time = time.perf_counter()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame")
        break

    current_time = time.perf_counter()
    frame_time = current_time - previous_time
    previous_time = current_time

    fps = 1 / frame_time if frame_time > 0 else 0

   #bdr

    b, g, r = cv2.split(frame)

    red_score = r.astype("float32") - (
        g.astype("float32") + b.astype("float32")
    ) / 2

    _, _, _, bgr_location = cv2.minMaxLoc(red_score)

    bgr_result = frame.copy()

    cv2.circle(bgr_result, bgr_location, 10, (0, 0, 255), 2)

   #hsv

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    _, max_value, _, hsv_location = cv2.minMaxLoc(red_mask)

    hsv_result = frame.copy()

    if max_value > 0:
        cv2.circle(hsv_result, hsv_location, 10, (0, 0, 255), 2)

    cv2.putText(bgr_result, "BGR ", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(hsv_result, "HSV ", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(bgr_result, f"FPS: {fps:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
    comparison = cv2.hconcat([bgr_result, hsv_result])# put both results next to each other

    cv2.imshow("BGR vs HSV comparison", comparison)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
