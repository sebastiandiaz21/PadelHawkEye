import cv2
import numpy as np

class BallDetector:
    """Detect the padel ball in frames."""

    def __init__(self, lower_color=(30, 150, 100), upper_color=(50, 255, 255)):
        """Initialize detector with color range in HSV."""
        self.lower_color = np.array(lower_color, dtype=np.uint8)
        self.upper_color = np.array(upper_color, dtype=np.uint8)

    def detect(self, frame):
        """Return the center of the detected ball in pixel coordinates."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius < 2:
            return None
        return int(x), int(y)
