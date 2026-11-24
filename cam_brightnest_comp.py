import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access the camera")
    exit()

ret, frame = cap.read()

if not ret:
    print("Cannot receive frame")
    cap.release()
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# double loop
def brightest_double_loop(image):

    max_value = -1
    max_location = (0, 0)

    height, width = image.shape

    for y in range(height):
        for x in range(width):

            pixel_value = int(image[y, x])

            if pixel_value > max_value:
                max_value = pixel_value
                max_location = (x, y)

    return max_value, max_location


# numpy 
def brightest_numpy(image):

    max_value = np.max(image) # maximum brightness value

    locations = np.where(image == max_value)

    y = locations[0][0]
    x = locations[1][0] #matching pixels 

    return int(max_value), (int(x), int(y))

#openCV
def brightest_opencv(image):

    # Find minimum and maximum values and their locations
    min_value, max_value, min_location, max_location = cv2.minMaxLoc(image)

    return int(max_value), max_location

#measuring the latecy 
def measure_time(function, image, iterations=20):

    function(image)
    start_time = time.perf_counter()

    for _ in range(iterations): #multiple times 
        function(image)

    end_time = time.perf_counter()
    average_time = (end_time - start_time) / iterations #average

    return average_time * 1000  #convert to ms

#results:

methods = [
    ("Double loop", brightest_double_loop),
    ("NumPy", brightest_numpy),
    ("OpenCV", brightest_opencv)
]
for name, function in methods:

    value, location = function(gray)# brightest pixel

    average_ms = measure_time(function, gray)# measure execution time
    fps = 1000 / average_ms # calculate equivalent FPS

    print("\n" + name)
    print("Maximum brightness:", value)
    print("Location:", location)
    print(f"Average execution time: {average_ms:.4f} ms")
    print(f"Equivalent FPS: {fps:.2f}")

cap.release()