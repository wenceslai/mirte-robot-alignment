import cv2
import numpy as np

from utils import plot_lines

def remove_noise(image: np.ndarray) -> np.ndarray:
    image = cv2.GaussianBlur(image, (5, 5), 15)
    image = cv2.medianBlur(image, 5)
    return image

def detect_edges(image: np.ndarray) -> np.ndarray:
    low_threshold = 50
    high_threshold = 150

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized_image = cv2.Canny(gray_image, low_threshold, high_threshold)
    return binarized_image

def find_lines(binarized_image: np.ndarray) -> np.ndarray:
    threshold = 200
    rho_step = np.pi / 180

    lines = cv2.HoughLines(binarized_image, 1, rho_step, threshold=threshold)

    if lines is None:
        print("No lines detected")
        return []
    else:
        return [line[0] for line in lines]

    
def cluster_lines(lines: list[tuple[float, float]]) -> list[tuple[float, float]]:
    K = 3

    lines_array = np.array(lines).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(lines_array.astype(np.float32), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Group the lines by their cluster
    clustered_lines = [[] for _ in range(K)]
    for i, label in enumerate(labels.flatten()):
        clustered_lines[label].append(lines[i])
    
    # Calculate the mean (rho, theta) for each cluster
    centroid_lines = []
    for cluster in clustered_lines:
        if cluster:
            mean_rho = np.mean([line[0] for line in cluster])
            mean_theta = np.mean([line[1] for line in cluster])
            centroid_lines.append((mean_rho, mean_theta))
    
    return centroid_lines

def is_alligned(centroid_lines: tuple[float, float]) -> bool:
    expected_angles = [(-60, 10), (0, 5), (60, 10)]

    # Define the expected angles and their tolerances
    thetas = [np.degrees(line[1]) - 90 for line in centroid_lines]
    sorted_thetas = sorted(thetas)

    for theta, (expected, tolerance) in zip(sorted_thetas, expected_angles):
        if abs(theta - expected) > tolerance:
            return False
    
    return True


if __name__ == "__main__":
    image = cv2.imread("test-images/rails_aligned.jpg", cv2.IMREAD_UNCHANGED)

    cv2.imwrite("debug-images/original.png", image)

    image = remove_noise(image)
    cv2.imwrite("debug-images/noise-removal.png", image)

    binarized_image = detect_edges(image)
    cv2.imwrite("debug-images/edge-detection.png", binarized_image)

    lines = find_lines(binarized_image)
    print(len(lines), " lines found")
    plot_lines(binarized_image, lines, "hough-transform")

    centroid_lines = cluster_lines(lines)
    plot_lines(binarized_image, centroid_lines, "clustered-lines", line_weight=10)
    
    print([np.degrees(line[1]) - 90 for line in centroid_lines])

    print(is_alligned(centroid_lines))