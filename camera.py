import time
import cv2

cap = cv2.VideoCapture(0) #connect to laptop camera

if not cap.isOpened():
    print("Cannot access the camera")
    exit()

previous_time = time.perf_counter() #first frame 

while True: # capture the frames 

    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame")  # if not captured 
        break

    current_time = time.perf_counter()

    frame_time = current_time - previous_time
    previous_time = current_time

    if frame_time > 0:
        fps = 1 / frame_time
    else:
        fps = 0                 # calculation of FPS

    milliseconds = frame_time * 1000 # frame time to ms

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #displey FPS

    cv2.putText(frame, f"Frame time: {milliseconds:.2f} ms", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  #displey frmae time 

    cv2.imshow("image", frame) #displey the image 

    if cv2.waitKey(1) & 0xFF == ord('q'): #exit command 
        break

cap.release() #release camera
cv2.destroyAllWindows() #close window 