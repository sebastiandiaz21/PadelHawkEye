import numpy as np

class TrajectoryPredictor:
    """Predict trajectory using recent positions."""

    def __init__(self, maxlen=10):
        self.maxlen = maxlen
        self.points = []

    def add_point(self, point):
        if point is None:
            return
        self.points.append(point)
        if len(self.points) > self.maxlen:
            self.points.pop(0)

    def predict(self):
        if len(self.points) < 3:
            return None
        pts = np.array(self.points)
        t = np.arange(len(pts))
        coeffs_x = np.polyfit(t, pts[:, 0], 2)
        coeffs_y = np.polyfit(t, pts[:, 1], 2)
        next_t = len(pts)
        pred_x = np.polyval(coeffs_x, next_t)
        pred_y = np.polyval(coeffs_y, next_t)
        return int(pred_x), int(pred_y)
