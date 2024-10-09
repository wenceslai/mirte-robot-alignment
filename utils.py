import cv2
import numpy as np

def plot_lines(image: np.ndarray, lines: list[tuple[float, float]], title="lines", line_weight=2):
    lines_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = int(a * rho)
        y0 = int(b * rho)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), line_weight)
    cv2.imwrite(f"debug-images/{title}.png", lines_image)
