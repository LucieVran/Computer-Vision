import time
import cv2
import numpy as np

CANNY_LOW = 50
CANNY_HIGH = 150
RANSAC_ITERATIONS = 500
RANSAC_THRESHOLD = 3
SCALE = 1.0
K = 2

def capture_frame(cap):
    start = time.perf_counter()
    ret, frame = cap.read()
    capture_time = time.perf_counter() - start

    if not ret:
        return None, capture_time

    return frame, capture_time


def detect_edges(frame):
    start = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    canny_time = time.perf_counter() - start

    return edges, canny_time

def get_edge_points(edges):
    ys, xs = np.where(edges > 0)
    return np.column_stack((xs, ys))

def calculate_line(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    dx = x2 - x1
    dy = y2 - y1

    a = dy
    b = -dx

    length = np.sqrt(a**2 + b**2)

    if length == 0:
        return None

    a /= length
    b /= length
    c = -(a * x1 + b * y1)

    return np.array([a, b, c])


def ransac_line(points, iterations=500, threshold=3):
    best_line = None
    best_inliers = None
    best_count = 0

    number_of_points = len(points)

    if number_of_points < 2:
        return None, None

    rng = np.random.default_rng()

    for _ in range(iterations):
        indices = rng.choice(number_of_points, 2, replace=False)

        point1 = points[indices[0]]
        point2 = points[indices[1]]

        line = calculate_line(point1, point2)

        if line is None:
            continue

        a, b, c = line

        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        inliers = distances < threshold
        inlier_count = np.sum(inliers)

        if inlier_count > best_count:
            best_count = inlier_count
            best_line = line
            best_inliers = points[inliers]

    return best_line, best_inliers


def refine_line(inliers):
    if inliers is None or len(inliers) < 2:
        return None

    points = np.asarray(inliers, dtype=float)

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
    direction = vh[0]

    a = -direction[1]
    b = direction[0]

    length = np.sqrt(a**2 + b**2)
    a /= length
    b /= length

    c = -(a * centroid[0] + b * centroid[1])

    return np.array([a, b, c])


def draw_line(frame, line, color=(0, 0, 255), thickness=2):
    if line is None:
        return

    a, b, c = line
    height, width = frame.shape[:2]

    if abs(b) > 1e-6:
        x1, y1 = 0, int(-c / b)
        x2 = width - 1
        y2 = int(-(a * x2 + c) / b)

    elif abs(a) > 1e-6:
        x1 = x2 = int(-c / a)
        y1, y2 = 0, height - 1

    else:
        return

    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def run_camera():
    cap = cv2.VideoCapture(0)

    frame_count = 0
    total_start = time.perf_counter()
    total_capture_time = 0
    total_canny_time = 0

    try:
        while True:
            frame, capture_time = capture_frame(cap)

            if frame is None:
                break

            edges, canny_time = detect_edges(frame)
            points = get_edge_points(edges)

            frame_count += 1
            total_capture_time += capture_time
            total_canny_time += canny_time

            elapsed_time = time.perf_counter() - total_start
            fps = frame_count / elapsed_time

            capture_avg = total_capture_time / frame_count
            canny_avg = total_canny_time / frame_count

            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Camera", frame)
            cv2.imshow("Canny Edges", edges)

            if frame_count % 100 == 0:
                print(f"\nFrames processed: {frame_count}")
                print(f"Average FPS: {fps:.2f}")
                print(f"Average capture: {capture_avg * 1000:.2f} ms")
                print(f"Average Canny: {canny_avg * 1000:.2f} ms")
                print(f"Edge points: {len(points)}")

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
    total_detection_time = 0

    try:
        while True:
            frame, capture_time = capture_frame(cap)

            if frame is None:
                break

            if SCALE != 1.0:
                frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

            edges, canny_time = detect_edges(frame)

            all_points = get_edge_points(edges)
            points = all_points[::K]

            detection_start = time.perf_counter()

            best_line, best_inliers = ransac_line(points, RANSAC_ITERATIONS, RANSAC_THRESHOLD)
            refined_line = refine_line(best_inliers)

            detection_time = time.perf_counter() - detection_start

            draw_line(frame, best_line, color=(0, 0, 255))
            draw_line(frame, refined_line, color=(0, 255, 0))

            frame_count += 1
            total_capture_time += capture_time
            total_canny_time += canny_time
            total_detection_time += detection_time

            elapsed_time = time.perf_counter() - total_start
            fps = frame_count / elapsed_time

            capture_avg = total_capture_time / frame_count
            canny_avg = total_canny_time / frame_count
            detection_avg = total_detection_time / frame_count

            inlier_count = len(best_inliers) if best_inliers is not None else 0
            inlier_ratio = inlier_count / len(points) if len(points) > 0 else 0

            cv2.putText(frame, f"Average FPS: {fps:.2f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Camera - RANSAC", frame)
            cv2.imshow("Canny Edges", edges)

            if frame_count % 30 == 0:
                print(f"\nFrame: {frame_count}")
                print(f"FPS: {fps:.2f}")
                print(f"Capture: {capture_avg * 1000:.2f} ms")
                print(f"Canny: {canny_avg * 1000:.2f} ms")
                print(f"RANSAC + refinement: {detection_avg * 1000:.2f} ms")
                print(f"Edge points: {len(all_points)}")
                print(f"Points used: {len(points)}")
                print(f"Inliers: {inlier_count}")
                print(f"Inlier ratio: {inlier_ratio:.3f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = input("Choose: ")
    if mode == "1":
        run_camera()
    elif mode == "2":
        run_ransac()
    else:
        print("invalide")
