import numpy as np
class Frame():
    def __init__(self,id,camera_id, frame, time_stamp, keyframe=False):
        self.id=id
        self.camera_id=camera_id
        self.frame=frame
        self.timestamp=time_stamp
        self.keyframe=keyframe
        self.landmark_ids=[]
        self.bag_of_words=[]
        self.T=np.array([])


        