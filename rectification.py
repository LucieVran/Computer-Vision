import time
import cv2
import numpy as np
from itertools import combinations

def angle_difference(a, b):
 return abs((a - b + np.pi / 2) % np.pi - np.pi / 2) #angle difference 

def suppress_duplicate_lines(segments):

    if segments is None:
        return []

    unique_lines = []
    segments = [np.asarray(line).reshape(-1)[:4] for line in segments]# convert every segment into four coordinates
    segments = sorted(segments,key=lambda line: np.hypot(line[2] - line[0], line[3] - line[1]), reverse=True)

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


def line_intersection(line1, line2):

    #convert both lines to homogeneous coordinates
    l1 = np.array([np.cos(line1["theta"]), np.sin(line1["theta"]), -line1["rho"]])
    l2 = np.array([np.cos(line2["theta"]),np.sin(line2["theta"]), -line2["rho"]])

    point = np.cross(l1, l2) #intesection using cross product

    if abs(point[2]) < 1e-8:# avoid division by zero for parallel lines
        return None

    x = point[0] / point[2]
    y = point[1] / point[2]

    if not np.isfinite(x) or not np.isfinite(y):
        return None

    return np.array([x, y], dtype=np.float32)

def select_rectangle_lines(lines, frame_shape): #four rectangle boundary lines

    if len(lines) < 4:
        return []

    frame_height, frame_width = frame_shape[:2]
    frame_area = frame_width * frame_height

    best_pair = []
    best_score = -1

    #limit the number of candidates
    lines = sorted(lines, key=lambda line: line["length"], reverse=True)[:18]

    for four_lines in combinations(lines, 4):   # test every combination of four detected lines
        pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]

        for pair1_indices, pair2_indices in pairings:

            group1 = [four_lines[pair1_indices[0]], four_lines[pair1_indices[1]]]
            group2 = [four_lines[pair2_indices[0]], four_lines[pair2_indices[1]]]

            angle1 = group1[0]["theta"] #orientation of each line
            angle2 = group1[1]["theta"]
            angle3 = group2[0]["theta"]
            angle4 = group2[1]["theta"]

        
            if angle_difference(angle1, angle2) > np.deg2rad(15): #opposite sides should be approximately parallel
                continue
            if angle_difference(angle3, angle4) > np.deg2rad(15):
                continue
           
            difference = angle_difference(angle1, angle3) #two groups should be approximately perpendicular

            if not (np.deg2rad(65) < difference < np.deg2rad(115)):
                continue

            corners = []

            for line1 in group1:
                for line2 in group2:

                    point = line_intersection(line1, line2)
                    if point is not None:
                        corners.append(point)

            if len(corners) != 4:
                continue

            corners = order_corners(corners)

            # All corners should be inside the camera image
            #if np.any(corners[:, 0] < 0) or np.any(corners[:, 0] >= frame_width):
                #continue
            #if np.any(corners[:, 1] < 0) or np.any(corners[:, 1] >= frame_height):
                #continue

            if not cv2.isContourConvex(corners.astype(np.float32)):# reject crossed or non-convex quadrilaterals
                continue

            area = abs(cv2.contourArea(corners))# calculate quadrilateral area
            area_ratio = area / frame_area

            if not (0.005 < area_ratio < 0.98):#reject very small or almost full-frame quadrilaterals
                continue

            #estimate the width and height
            width_top = np.linalg.norm(corners[1] - corners[0])
            width_bottom = np.linalg.norm(corners[2] - corners[3])

            height_left = np.linalg.norm(corners[3] - corners[0])
            height_right = np.linalg.norm(corners[2] - corners[1])

            width = (width_top + width_bottom) / 2
            height = (height_left + height_right) / 2

            if width < 30 or height < 30:
                continue

            ratio = width / height

            portrait_ratio = 210 / 297
            landscape_ratio = 297 / 210

            ratio_error = min(abs(np.log(ratio / portrait_ratio)), abs(np.log(ratio / landscape_ratio)))

            if ratio_error > 1.5: #reject proportion far from a4
                continue

            # Score the candidate: longer lines, larger area and A4-like proportions are preferred
            line_strength = sum(line["length"] for line in four_lines)

            score = ( line_strength * np.sqrt(area_ratio) * np.exp(-2 * ratio_error))

            if score > best_score:
                best_score = score
                best_pair = list(group1 + group2)

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

def calculate_intersections(selected_lines):# calculate intersections between four selected lines

    if selected_lines is None or len(selected_lines) != 4:
        return []

    intersections = []

    for i, j in combinations(range(4), 2):  #generate all possible pairs of four lines

        line1 = selected_lines[i]
        line2 = selected_lines[j]

        #convert lines into homogeneous coordinates
        l1 = np.array([np.cos(line1["theta"]), np.sin(line1["theta"]), -line1["rho"]])
        l2 = np.array([np.cos(line2["theta"]), np.sin(line2["theta"]), -line2["rho"]])

        point = np.cross(l1, l2)# cross product gives the intersection

        if abs(point[2]) < 1e-8:# check for parallel or almost parallel lines
            continue

        #convert homogeneous coordinates to pixel coordinates
        x = point[0] / point[2]
        y = point[1] / point[2]

        if np.isfinite(x) and np.isfinite(y):
            intersections.append((x, y))

    return intersections

def calculate_rectangle_corners(selected_lines):

    if selected_lines is None or len(selected_lines) != 4:
        return []

    group1 = [selected_lines[0], selected_lines[1]] #separate lines into two orientation groups
    group2 = [selected_lines[2], selected_lines[3]]

    corners = []

    #calculate intersections between the two groups
    for line1 in group1:
        for line2 in group2:

            l1 = np.array([np.cos(line1["theta"]), np.sin(line1["theta"]), -line1["rho"]])
            l2 = np.array([np.cos(line2["theta"]), np.sin(line2["theta"]), -line2["rho"]])

            point = np.cross(l1, l2)

            if abs(point[2]) < 1e-8:
                continue

            x = point[0] / point[2]
            y = point[1] / point[2]

            if np.isfinite(x) and np.isfinite(y):
                corners.append((x, y))

    return corners

def order_corners(corners):

    points = np.array(corners, dtype=np.float32)
    centre = np.mean(points, axis=0) #calculate the centre of the four points

    # Calculate the angle of each point around the centre
    angles = np.arctan2(
        points[:, 1] - centre[1],
        points[:, 0] - centre[0] )


    ordered = points[np.argsort(angles)] # sort points clockwise
    start = np.argmin(ordered[:, 0] + ordered[:, 1])# find the top-left corner (smallest x + y)
    ordered = np.roll(ordered, -start, axis=0) # rotate the array to start with top-left

    return ordered


def perspective_transform(frame, corners):
    width = 600
    height = 848

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(frame, matrix, (width, height))

    return warped
    

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    exit()

#corner smoothing
previous_corners = None
alpha = 0.15

missing_frames = 0
max_missing_frames = 5

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

    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=80, maxLineGap=30)
    hough_time = (time.perf_counter() - start) * 1000

    #suppres and select 
    start = time.perf_counter()
    unique_lines = suppress_duplicate_lines(segments)
    selected_lines = select_rectangle_lines(unique_lines, frame.shape)

    corners = calculate_rectangle_corners(selected_lines)
    
    if len(corners) == 4:
        corners = order_corners(corners)
        current_corners = np.array(corners, dtype=np.float32)

        # First detection: initialise previous corners
        if previous_corners is None:
            previous_corners = current_corners
        else:
            previous_corners = (alpha * current_corners + (1 - alpha) * previous_corners)

        corners = previous_corners.copy()
        missing_frames = 0

    else:
        missing_frames += 1

        if previous_corners is not None and missing_frames <= max_missing_frames:#keep previous corners briefly if detection is lost
            corners = previous_corners.copy()

        else:
            previous_corners = None
            corners = []
        
    if len(corners) == 4:
        warped = perspective_transform(frame, corners)
        cv2.imshow("Rectified", warped)

    for i, point in enumerate(corners):

        x, y = point  # Extract x and y coordinates

        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:

            cv2.circle(frame, (int(x), int(y)), 7, (0, 255, 255), -1)

            cv2.putText(frame, str(i + 1), (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
    selection_time = (time.perf_counter() - start) * 1000

    if segments is not None:
        for segment in segments:
            x1, y1, x2, y2 = np.asarray(segment).reshape(-1)[:4]

            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


    # Draw the four selected lines in 
    #if selected_lines:
       # for line in selected_lines:
         #   draw_line(frame, line, (0,255, 0), 2)

    # calculate FPS and display measurements
    total_time = (time.perf_counter() - frame_start) * 1000
    fps = 1000 / total_time if total_time > 0 else 0

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Edge: {edge_time:.1f} ms", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Hough: {hough_time:.1f} ms", (20, 65),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255, 255), 2)

    cv2.imshow("Detection", frame)
    cv2.imshow("Canny", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()