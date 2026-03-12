from .frame import Frame
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
class FrameManager():
    def __init__(self, camera_map):
        self.camera_map=camera_map
        self.keyframes=[]
        self.index=0
        
    def capture_frames(self):
        current_frames={}
        for camera_id, camera in self.camera_map.items():
            frame,timestamp=camera.get_frame()
            if frame is not None:
                frame=Frame(self.index,camera,frame, timestamp)
                self.index+=1
                current_frames[camera_id]=frame
        return current_frames
        
            
    def set_keyframe(self, frame):
        frame.keyframe=True
        self.keyframes.append(frame)
        for id,landmark in enumerate(frame.landmarks):
            if landmark is None:
                continue
            landmark.add_observation(frame,id)
    
    def get_last_keyframe(self):
        return self.keyframes[-1]
    
    def get_aged_keyframe(self,):
        return self.keyframes[-5]
    
    def find_closest_keyframe(self,frame,N=5):
        sorted_neighbors = sorted(frame.covisible.items(),key=lambda x: -x[1]) 
        best_neighbors = [kf for kf, weight in sorted_neighbors[:N]]
        if len(best_neighbors)==0:
            best_neighbors=self.keyframes[-5:]
        return best_neighbors
    
    def get_len_keyframes(self):
        return len(self.keyframes)
        
        