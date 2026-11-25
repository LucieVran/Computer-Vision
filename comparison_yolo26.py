import cv2
import numpy as np
import time

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

CONFIDENCE_THRESHOLD = 0.25

net_n = cv2.dnn.readNetFromONNX("yolo26n.onnx")
net_s = cv2.dnn.readNetFromONNX("yolo26s.onnx")
print("YOLO26n")
print("YOLO26s")

cap = cv2.VideoCapture(0)
selected_model = "n"

fps_frames = {"n": 0, "s": 0}
fps_total_time = {"n": 0.0, "s": 0.0}

if not cap.isOpened():
    print("Cannot open camera")
    exit()


while True:

    start_time = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        print("Cannot read camera frame")
        break

    height, width = frame.shape[:2]

# letterbox
    scale = min(640 / height, 640 / width)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(frame, (new_width, new_height))

    pad_width = 640 - new_width
    pad_height = 640 - new_height

    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top

    image = cv2.copyMakeBorder(resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114))

   #preprocesing 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    # interference
    if selected_model == "n":
        net = net_n
    else:
        net = net_s

    net.setInput(image)
    outputs = net.forward()

    detection_count = 0

    for detection in outputs[0]:

        x1, y1, x2, y2, confidence, class_id = detection

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(class_id)

        # Convert coordinates
        x1 = int((x1 - left) / scale)
        y1 = int((y1 - top) / scale)
        x2 = int((x2 - left) / scale)
        y2 = int((y2 - top) / scale)

        # Keep coordinates inside frame
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        class_name = COCO_CLASSES[class_id]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_name}: {confidence:.2f}"

        cv2.putText( frame, label, (x1, max(y1 - 10, 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2)

        detection_count += 1

    #FPS
    elapsed_time = time.perf_counter() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0

    # measure average FPS for each model over 200 frames
    fps_frames[selected_model] += 1
    fps_total_time[selected_model] += elapsed_time

    if fps_frames[selected_model] == 200:
        average_fps = 200 / fps_total_time[selected_model]

        model_name = "YOLO26n" if selected_model == "n" else "YOLO26s"

        print(f"{model_name} average FPS: {average_fps:.2f}")

        fps_frames[selected_model] = 0
        fps_total_time[selected_model] = 0.0

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display detection count
    #cv2.putText(frame,f"Detections: {detection_count}",(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)

    model_name = "YOLO26n" if selected_model == "n" else "YOLO26s"

    cv2.putText(frame,f"Model: {model_name}", (frame.shape[1] - 190, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("YOLO26n Live Detection", frame)

    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("1"):
        selected_model = "n"
        print("Switched to YOLO26n")
    elif key == ord("2"):
        selected_model = "s"
        print("Switched to YOLO26s")

cap.release()
cv2.destroyAllWindows()