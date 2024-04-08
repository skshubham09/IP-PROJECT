import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def process_image(image_path):
    # Load the kidney image
    image = cv2.imread(image_path)

    # Manually provide contour points
    contour_points = np.array([[30, 80], [130, 150], [240, 90], [100, 30]])

    # Use cubic spline method for contour detection
    tck, _ = splprep([contour_points[:, 1], contour_points[:, 0]], s=0)

    # Evaluate spline
    u = np.linspace(0, 1, 100)
    x, y = splev(u, tck)

    # Create a new contour with x, y coordinates
    contour = np.column_stack((y, x)).astype(int)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask for the contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Rotate the image to 0 degrees
    angle = 0  # You can change this angle if needed
    rows, cols = gray.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(gray, M, (cols, rows))

    # Remove background
    result = cv2.bitwise_and(rotated_image, rotated_image, mask=mask)

    return result 