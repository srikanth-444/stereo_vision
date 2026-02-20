import numpy as np
from .landmarks import Landmark


class LandmarkManager:
    def __init__(self):
        self.landmarks=[]
        self.id_counter=0
        self.landmark_map = {}
        self.active_ids = set()
        self.poistion_3d={}
       

    def add_landmark(self,position,image_points,descriptor,frame_id):
        landmark=Landmark()
        landmark.update(position,image_points,frame_id)
        position=np.floor(position/0.01).astype(int)
        self.poistion_3d[position[0],position[1],position[2]]=landmark
        landmark.descriptor=descriptor
        landmark.id=self.id_counter
        self.landmark_map[landmark.id] = landmark
        self.active_ids.add(landmark.id)
        self.id_counter+=1
        self.landmarks.append(landmark)

    def update_landmark_position(self,id,position):
        landmark=self.get_landmark_by_id(id)
        if landmark:
            landmark.position=position

    def get_active_landmarks(self):
        return [self.landmark_map[id] for id in self.active_ids] 

    def deactivate_landmark(self,id):
        landmark=self.get_landmark_by_id(id)
        if landmark:
            landmark.active=False
        else:
            print(f"Landmark with ID {id} not found.")

    def get_landmark_by_id(self,id):
        if id in self.landmark_map:
            return self.landmark_map[id]
        return None

    def num_landmarks(self):
        return len(self.landmarks)
    
    def check_closest_point(self,point,threshold=0.1):
        if len(self.poistion_3d) == 0:
            return False
        point =np.floor(point/threshold).astype(int)

        try:
            landmark=self.poistion_3d[point[0],point[1],point[2]]
            if landmark is not None:
                landmark.active=True
                return True
            else:
                return False
        except KeyError:
            return False

    def num_of_active_landmarks(self):
        return len(self.get_active_landmarks())
    
    def update_active_landmark(self,):
        self.active_ids = {
        id for id in self.active_ids
        if self.get_landmark_by_id(id).active
    }
    