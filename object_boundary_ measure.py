import time
import cv2
import numpy as np

def angle_difference(a, b):
 return abs((a - b + np.pi / 2) % np.pi - np.pi / 2) #angle difference 

def suppress_duplicate_lines(segments):

    if segments is None:
        return []

    unique_lines = []
    segments = [np.asarray(line).reshape(-1)[:4] for line in segments]# convert every segment into four coordinates

    segments = sorted(segments,key=lambda line: np.hypot(
            line[2] - line[0], line[3] - line[1]), reverse=True)

    for segment in segments:

        x1, y1, x2, y2 = segment
        dx = x2 - x1
        dy = y2 - y1

        length = np.hypot(dx, dy)

        #convert segment into rho and theta
        theta = (np.arctan2(dy, dx) + np.pi / 2) % np.pi
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)

        #consistent representation
        if theta >= np.pi / 2:
            theta -= np.pi
            rho = -rho

        duplicate = False

        for line in unique_lines:

            if (angle_difference(theta, line["theta"]) < np.deg2rad(5) and abs(rho - line["rho"]) < 25):
                line["length"] += length
                duplicate = True
                break

        if not duplicate:
            unique_lines.append({"rho": rho, "theta": theta,"length": length})

    return unique_lines

#four rectangle boundary lines
def select_rectangle_lines(lines):

    if len(lines) < 4:
        return None

    best_pair = None# group lines by their orientation
    best_score = 0

    for first in lines:

        first_angle = (first["theta"] - np.pi / 2) % np.pi
        first_group = [line for line in lines
            if angle_difference((line["theta"] - np.pi / 2) % np.pi, first_angle) < np.deg2rad(25)]

        if len(first_group) < 2:
            continue

        for second in lines:

            second_angle = (second["theta"] - np.pi / 2) % np.pi
            difference = angle_difference(first_angle, second_angle)

            if not (np.deg2rad(65) < difference < np.deg2rad(115)):#rectangle sides should be approximately perpendicular
                continue

            second_group = [line for line in lines
                if angle_difference((line["theta"] - np.pi / 2) % np.pi, second_angle) < np.deg2rad(25)]

            if len(second_group) < 2:
                continue

            # Select the two strongest lines in each group
            group1 = sorted(first_group,key=lambda line: line["length"], reverse=True)[:2]
            group2 = sorted(second_group, key=lambda line: line["length"],reverse=True)[:2]

            # avoid selecting the same lines twice
            if len({id(line) for line in group1 + group2}) < 4:
                continue

            score = sum(line["length"] for line in group1 + group2)

            if score > best_score:
                best_score = score
                best_pair = group1 + group2

    return best_pair

def draw_line(frame, line, color, thickness):

    rho = line["rho"]
    theta = line["theta"]

    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))

    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    exit()


warmup_frames = 10
measurement_frames = 200
frame_count = 0

edge_times = []
hough_times = []
selection_times = []
total_times = []

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_start = time.perf_counter()
    start = time.perf_counter()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()

    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, minLineLength=80, maxLineGap=30)
    hough_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()

    unique_lines = suppress_duplicate_lines(segments)
    selected_lines = select_rectangle_lines(unique_lines)
    selection_time = (time.perf_counter() - start) * 1000

    total_time = (time.perf_counter() - frame_start) * 1000
    fps = 1000 / total_time if total_time > 0 else 0

    if frame_count >= warmup_frames:
        edge_times.append(edge_time)
        hough_times.append(hough_time)
        selection_times.append(selection_time)
        total_times.append(total_time)

    frame_count += 1

    #draw all Hough line segments
    if segments is not None:
        for segment in segments:
            x1, y1, x2, y2 = np.asarray(segment).reshape(-1)[:4]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
   # if selected_lines is not None:
       # for line in selected_lines:
            #draw_line(frame, line, (0, 255, 0), 2)

    total_time = (time.perf_counter() - frame_start) * 1000
    fps = 1000 / total_time if total_time > 0 else 0

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Edge: {edge_time:.1f} ms", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Hough: {hough_time:.1f} ms", (20, 65),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255, 255), 2)

    cv2.imshow("Detection", frame)
    cv2.imshow("Canny", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(total_times) >= measurement_frames:
        break

cap.release()
cv2.destroyAllWindows()

if len(total_times) > 0:
    print(f"Measured frames: {len(total_times)}")
    print(f"Average edge processing time: {np.mean(edge_times):.2f} ms")
    print(f"Average Hough Transform time: {np.mean(hough_times):.2f} ms")
    print(f"Average line selection time: {np.mean(selection_times):.2f} ms")
    print(f"Average total processing time: {np.mean(total_times):.2f} ms")

    average_fps = 1000 / np.mean(total_times)
    print(f"Average processing FPS: {average_fps:.2f}")