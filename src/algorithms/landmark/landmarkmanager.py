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
                del(self.landmark_map[id])
                # print(landmark.id)
                for frame ,idx in landmark.observations.items():
                    frame.landmarks[idx]=None

    def merge_landamrks(self, landmarks):
        if len(landmarks) < 2:
            return
        landmarks = list(landmarks)
        
        ids=[lm.id for lm in landmarks]
        # print(f"ids{ids}")
        survivor_id = min(ids)
        # print(f"survivor {survivor_id}")
        survivor = self.get_landmark_by_id(survivor_id)
        for id in ids:
            if id != survivor_id:
                # print(id)
                landmark=self.get_landmark_by_id(id)
                if landmark is None:
                    continue
                for frame in landmark.observations.keys():
                    # print(frame.id)
                    idx=frame.keypoint_landmarks_association[landmark]
                    frame.keypoint_landmarks_association[survivor]=idx
                    frame.landmarks_keypoint_association[idx]=survivor
                    survivor.observations[frame]=frame.image_points[idx]
                    survivor.tracked=survivor.tracked+1
                    del(frame.keypoint_landmarks_association[landmark])
                del(self.landmark_map[id])


    