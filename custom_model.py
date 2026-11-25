import cv2
import numpy as np
import time

net = cv2.dnn.readNetFromONNX("best.onnx")

class_names = ["flash_drive", "food_clip"]

confidence_threshold = 0.20

cap = cv2.VideoCapture(1) # throught phone

if not cap.isOpened():
    exit()

previous_time = time.perf_counter()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # preprocess image using letterbox
    height, width = frame.shape[:2]

    scale = min(640 / width, 640 / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(frame, (new_width, new_height))

    pad_x = (640 - new_width) // 2
    pad_y = (640 - new_height) // 2

    input_image = np.full((640, 640, 3), 114, dtype=np.uint8)

    input_image[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.astype(np.float32) / 255.0

    input_image = np.transpose(input_image, (2, 0, 1))# HWC to NCHW
    input_image = np.expand_dims(input_image, axis=0) # add batch dimension

    net.setInput(input_image)#inference
    output = net.forward()

    #model output
    predictions = output[0].T

    boxes = []
    confidences = []
    class_ids = []

    for prediction in predictions:

        scores = prediction[4:] #class score 

        # find highest scoring class
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > confidence_threshold:

            x_center, y_center, box_width, box_height = prediction[:4] # bounding box coordinates

            # convert center coordinates to top-left coordinates
            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2

            # convert coordinates from 640x640 to original frame
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            box_width = box_width / scale
            box_height = box_height / scale

            boxes.append([int(x1), int(y1), int(box_width), int(box_height)])

            confidences.append(float(confidence))
            class_ids.append(int(class_id))

    # apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.45)

    # detections
    if len(indices) > 0:

        for i in np.array(indices).flatten():

            x, y, w, h = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]

            label = f"{class_names[class_id]} {confidence:.2f}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, label, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),2)

    cv2.imshow("Custom YOLO26n", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    current_time = time.perf_counter()
    previous_time = current_time

cap.release()
cv2.destroyAllWindows()