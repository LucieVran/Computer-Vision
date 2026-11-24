import time
import cv2

cap = cv2.VideoCapture(0) #connect to laptop camera, nuber 0 is laptop, 1 is phone

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert into greyscale

    min_value, max_value, min_location, max_location = cv2.minMaxLoc(gray) #place of the brightest place

    cv2.circle(frame, max_location, 10, (0, 0, 255), 2) # making a curcle/ marking 

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #displey FPS

    cv2.putText(frame, f"Frame time: {milliseconds:.2f} ms", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  #displey frmae time 

    cv2.imshow("Brightest spot detection", frame) #displey the image 

    if cv2.waitKey(1) & 0xFF == ord('q'): #exit command 
        break

cap.release() #release camera
cv2.destroyAllWindows() #close window 