import time
import cv2
import numpy as np

CANNY_LOW = 50
CANNY_HIGH = 150
RANSAC_ITERATIONS = 500
RANSAC_THRESHOLD = 3

SCALE = 1
K = 2

def capture_frame(cap):
    start = time.perf_counter()
    ret, frame = cap.read()
    capture_time = time.perf_counter() - start

    if not ret:
        return None, capture_time

    return frame, capture_time



def detect_edges(frame, threshold1=100, threshold2=200):

    start = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #change colors to grey
    edges = cv2.Canny(gray, threshold1, threshold2) #openCV canny function 

    canny_time = time.perf_counter() - start

    return edges, canny_time


def get_edge_points(edges): #extract coordinates x and y 

    ys, xs = np.where(edges > 0)
    points = np.column_stack((xs, ys))
    return points

def display_results(frame, edges, fps, capture_avg, canny_avg, edge_count):

    cv2.putText(frame, f"FPS: {fps:.2f}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Edge points: {edge_count}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    cv2.imshow("Canny Edges", edges)

def calculate_line(point1, point2): #normalized line calculation 

    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1        #direction between two points 
    dy = y2 - y1

    a = dy          # normal perpedicular vector
    b = -dx

    length = np.sqrt(a**2 + b**2)   # normalize the coefficients

    if length == 0:
        return None

    a = a / length
    b = b / length

    c = -(a * x1 + b * y1) # make sure the line pass the firt point 

    return np.array([a, b, c])

def ransac_line(points, iterations=500, threshold=3): 

    best_line = None
    best_inliers = None
    best_count = 0

    number_of_points = len(points)

    if number_of_points < 2: # minimum of two points 
        return None, None

    rng = np.random.default_rng() #random number generator 

    #RANSAC repetation 
    for i in range(iterations):

        indices = rng.choice(number_of_points, 2, replace=False) # randomly select two points 

        point1 = points[indices[0]]
        point2 = points[indices[1]]

        line = calculate_line(point1, point2) # calculate the line

        if line is None:
            continue

        a, b, c = line

        # distance from every point to the line
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)

        inliers = distances < threshold
        inlier_count = np.sum(inliers) #count inliners 

        if inlier_count > best_count: #best line keeper 

            best_count = inlier_count
            best_line = line
            best_inliers = points[inliers]

    return best_line, best_inliers

def refine_line(inliers):

    if inliers is None or len(inliers) < 2: # do we have enought points 
        return None

    points = np.asarray(inliers, dtype=float) # convert points to floating-point numbers

    centroid = np.mean(points, axis=0) # the center of the points 
    centered_points = points - centroid  # move all points relative to the centre

    _, _, vh = np.linalg.svd(centered_points, full_matrices=False)    # Perform Singular Value Decomposition
    # find the direction in which our points are distributed the mos
    # The first vector gives the main direction of the points
    direction = vh[0]

    # Convert direction into line coefficients
    a = -direction[1]
    b = direction[0]
    length = np.sqrt(a**2 + b**2)  # normalize the coefficients

    a = a / length
    b = b / length
    c = -(a * centroid[0] + b * centroid[1])    # calculate the final coefficient
    return np.array([a, b, c])

def draw_line(frame, line, color=(0, 0, 255), thickness=2):

    if line is None:
        return

    a, b, c = line
    height, width = frame.shape[:2]

    # calculate the line's intersections with the left and right image borders
    if abs(b) > 1e-6:

        x1 = 0
        y1 = int(-c / b)
        x2 = width - 1
        y2 = int(-(a * x2 + c) / b)

    # special case of vertical line 
    elif abs(a) > 1e-6:

        x1 = int(-c / a)
        y1 = 0
        x2 = int(-c / a)
        y2 = height - 1

    else:
        return

    cv2.line(frame, (x1, y1), (x2, y2), color, thickness) #draw the line 

#main loop 
def run_camera():

    cap = cv2.VideoCapture(0)

    frame_count = 0
    total_start = time.perf_counter()
    total_capture_time = 0
    total_canny_time = 0

    try:
        while True:

            frame, capture_time = capture_frame(cap) #capture 

            # canny
            edges, canny_time = detect_edges(
                frame,
                threshold1=100,
                threshold2=200
            )

            points = get_edge_points(edges) # get the coordinates of the arrey 

            frame_count += 1 #update the measurements 

            total_capture_time += capture_time
            total_canny_time += canny_time

            elapsed_time = time.perf_counter() - total_start

            fps = frame_count / elapsed_time #average FPS

            capture_avg = total_capture_time / frame_count
            canny_avg = total_canny_time / frame_count #average processing time 

            display_results(frame, edges, fps, capture_avg, canny_avg, len(points))

            # print measurements every 150 frames
            if frame_count % 100 == 0:

                print(f"\nFrames processed: {frame_count}")
                print(f"Average FPS: {fps:.2f}")
                print(f"Average capture: {capture_avg * 1000:.2f} ms")
                print(f"Average Canny: {canny_avg * 1000:.2f} ms")
                print(f"Number of edge points: {len(points)}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

def run_ransac():

    cap = cv2.VideoCapture(0)

    frame_count = 0
    total_start = time.perf_counter()

    total_capture_time = 0
    total_canny_time = 0
    total_ransac_time = 0

    try:
        while True:
          
            frame, capture_time = capture_frame(cap)  # capture frame

            if SCALE != 1.0:
                frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

            if frame is None:
                break

            # Canny
            edges, canny_time = detect_edges(frame,threshold1=CANNY_LOW, threshold2=CANNY_HIGH)

            points = get_edge_points(edges) # edge coordinates 
            points = points[::K]
            ransac_start = time.perf_counter() #ransac

            best_line, best_inliers = ransac_line(points, iterations=RANSAC_ITERATIONS, threshold=RANSAC_THRESHOLD)
            
            refined_line = refine_line(best_inliers) 
            if best_line is not None and refined_line is not None:

                print("RANSAC line:", best_line)
                print("Refined line:", refined_line)

                difference = np.linalg.norm(best_line - refined_line)
                print("Coefficient difference:", difference)

            if best_line is not None:
                draw_line(frame, best_line, color=(0, 0, 255))
            if refined_line is not None:
                draw_line(frame, refined_line, color=(0, 255, 0))

            ransac_time = time.perf_counter() - ransac_start

            # update measurements
            frame_count += 1

            total_capture_time += capture_time
            total_canny_time += canny_time
            total_ransac_time += ransac_time

            elapsed_time = time.perf_counter() - total_start

            fps = frame_count / elapsed_time

            capture_avg = total_capture_time / frame_count
            canny_avg = total_canny_time / frame_count
            ransac_avg = total_ransac_time / frame_count

            # print measurements
            if frame_count % 30 == 0:

                inlier_count = (len(best_inliers) if best_inliers is not None else 0)
                inlier_ratio = (inlier_count / len(points) if len(points) > 0 else 0)

                print(f"\nFrame: {frame_count}")
                print(f"FPS: {fps:.2f}")
                print(f"Capture: {capture_avg * 1000:.2f} ms")
                print(f"Canny: {canny_avg * 1000:.2f} ms")
                print(f"RANSAC: {ransac_avg * 1000:.2f} ms")
                print(f"Edge points: {len(points)}")
                print(f"Inliers: {inlier_count}")
                print(f"Inlier ratio: {inlier_ratio:.3f}")

            # Display
            cv2.putText(frame, f"Average FPS: {fps:.2f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Camera - RANSAC", frame)
            cv2.imshow("Canny Edges", edges)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    mode = input("choose: ")
    if mode == "1":
        run_camera()
    elif mode == "2":
        run_ransac()
    else:
        print("invalid")