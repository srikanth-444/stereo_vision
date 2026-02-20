import cv2

class CameraUSB:
    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = timestamp_ms / 1000.0
        return gray_frame, timestamp

    def release(self):
        self.cap.release()