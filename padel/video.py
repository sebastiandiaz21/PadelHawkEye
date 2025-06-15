import cv2

class VideoSource:
    """Manage video input from file or camera."""

    def __init__(self, path: str = 0):
        """Initialize the video source.

        Args:
            path: Path to video file or camera index/URL.
        """
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {path}")

    def read(self):
        """Read a frame from the video source."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
