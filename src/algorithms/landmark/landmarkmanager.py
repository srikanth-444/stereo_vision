import numpy as np
from .landmarks import Landmark
import logging


class LandmarkManager:
    def __init__(self):
        self.id_counter=0
        self.landmark_map = {}
        self.active_ids = set()
        self.poistion_3d={}
        
        
       

    def add_landmark(self,position,frame,feature_id):
        landmark=Landmark(self.id_counter, position,frame,feature_id)
        self.landmark_map[landmark.id] = landmark
        self.active_ids.add(landmark.id)
        self.id_counter+=1
        return landmark

    def get_active_landmarks(self):
        return [self.landmark_map[id] for id in self.active_ids] 

    def deactivate_landmark(self,id):
        landmark=self.get_landmark_by_id(id)
        if landmark:
            landmark.active=False
        else:
            logging.debug(f"Landmark with ID {id} not found.")

    def get_landmark_by_id(self,id):
        if id in self.landmark_map:
            return self.landmark_map[id]
        return None

    
    def check_closest_point(self,id,image_point,des,point,threshold=0.1):
        if len(self.poistion_3d) == 0:
            return False
        point =np.floor(point/threshold).astype(int)

        try:
            landmark=self.poistion_3d[point[0],point[1],point[2]]
            if landmark is not None:
                landmark.active=True
                landmark.add_observation(id, image_point,des)
                return True
            else:
                return False
        except KeyError:
            return False

    def num_of_active_landmarks(self):
        return len(self.get_active_landmarks())
    
    def update_active_landmark(self,landmarks):
        self.active_ids = {
        landmark.id for landmark in landmarks
        if landmark.active
    }
    
    def remove_bad_landmarks(self,landmarks):
        land_ids=[lm.id for lm in landmarks]
        
        for id in land_ids:
            if id not in self.landmark_map:
                continue
            landmark=self.get_landmark_by_id(id)
            landmark.active=False
            if landmark.get_found_ratio()<0.25 or landmark.tracked<=2:
                landmark.is_bad=True
                for frame,idx in landmark.observations.items():
                    frame.landmarks[idx]=None


    def merge_landamrks(self, landmarks):
        if len(landmarks) < 2:
            return
        survivor = max(landmarks, key=lambda lm: len(lm.observations))

        for landmark in landmarks:
            if landmark is survivor:
                continue
            landmark.is_bad=True
            for frame, idx in landmark.observations.items():
                if frame in survivor.observations:
                    continue
                survivor.observations[frame] = idx
                frame.landmarks[idx] = survivor
                survivor.tracked += 1

                    
         


    