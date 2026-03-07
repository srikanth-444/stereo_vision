from .frame import Frame
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
class FrameManager():
    def __init__(self, camera_map):
        self.camera_map=camera_map
        self.keyframe_map={}
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
        self.keyframe_map[frame.id]=frame
    
    def get_last_keyframe(self):
        my_list = sorted(list(self.keyframe_map.keys()))
        return self.keyframe_map[my_list[-1]]
    
    def find_closest_keyframe(self):
        my_list = sorted(list(self.keyframe_map.keys()))
        return(my_list[-5:])
    
    def get_len_keyframes(self):
        return len(list(self.keyframe_map))
        
        