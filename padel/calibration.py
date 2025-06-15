import cv2
import numpy as np

class CourtCalibrator:
    """Detect court lines and compute homography."""

    def __init__(self, reference_points=None, real_width=20.0, real_height=10.0):
        """Initialize the calibrator.

        Args:
            reference_points: Optional list of four points [(x, y), ...] in
                image coordinates corresponding to court corners.
            real_width: Width of the court in meters.
            real_height: Height of the court in meters.
        """
        self.reference_points = reference_points
        self.real_width = real_width
        self.real_height = real_height
        self.homography = None

    def calibrate(self, frame):
        """Calibrate using the provided frame."""
        if self.reference_points is None:
            # Detect lines via Hough transform (simplified example).
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                    minLineLength=100, maxLineGap=10)
            if lines is None:
                return False
            # Placeholder: Use image corners as approximate reference.
            h, w = frame.shape[:2]
            self.reference_points = [(0, 0), (w, 0), (w, h), (0, h)]
        pts_src = np.array(self.reference_points, dtype=np.float32)
        pts_dst = np.array([[0, 0], [self.real_width, 0],
                            [self.real_width, self.real_height],
                            [0, self.real_height]], dtype=np.float32)
        self.homography, _ = cv2.findHomography(pts_src, pts_dst)
        return self.homography is not None

    def pixel_to_world(self, point):
        """Map pixel coordinates to world coordinates using homography."""
        if self.homography is None:
            raise ValueError("Calibrator has not been fitted")
        px = np.array([[point[0], point[1], 1]], dtype=np.float32).T
        world = self.homography @ px
        world /= world[2, 0]
        return float(world[0, 0]), float(world[1, 0])
