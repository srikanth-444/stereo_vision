import cv2
import time

class VideoPlayback:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        
    
    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from video.")
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp = time.time()
        return gray_frame, timestamp
    
    def release(self):
        self.cap.release()