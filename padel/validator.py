import cv2
import numpy as np

class FrameValidator:
    """Check if the current frame matches the calibrated view."""

    def __init__(self, reference_frame=None, threshold=10):
        self.reference_hash = None
        self.threshold = threshold
        if reference_frame is not None:
            self.reference_hash = self._phash(reference_frame)

    def _phash(self, frame):
        resized = cv2.resize(frame, (32, 32))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(np.float32(gray))
        dct_low = dct[:8, :8]
        med = np.median(dct_low)
        return (dct_low > med).astype(int)

    def is_valid(self, frame):
        if self.reference_hash is None:
            self.reference_hash = self._phash(frame)
            return True
        current_hash = self._phash(frame)
        diff = np.sum(current_hash != self.reference_hash)
        return diff < self.threshold
