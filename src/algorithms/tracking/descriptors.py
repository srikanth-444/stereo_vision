import numpy as np
from .pnp import PNPSolver
from .tracker import Tracker
import logging 

class DescriptorTracker(Tracker):
    def __init__(self,reprojection_error,confidence,iterationsCount,landmark_manager,camera) -> None:
        
        self.pnp_solver=PNPSolver(reprojection_error,confidence,iterationsCount,camera)
        self.landmark_manager=landmark_manager
    
    def track_landmarks(self,landmarks, T, frames):
        logging.debug(f"length of landmarks {len(landmarks)}")

        if len(landmarks) < 6:
            return None, None
        for lm in landmarks:
            lm.active=False
            lm.nvisible=lm.nvisible+1
        for frame in frames:
            matched_objects, matched_images, matched_des=frame.projection_match(landmarks,T)
            logging.debug(f"length of matched points {len(matched_objects)}")
            if len(matched_objects) < 20:
                return None, None
            rvec_new, tvec_new, inliers = self.pnp_solver.estimate_pose_pnp(matched_objects,matched_images,frame.intrinsic)
            if inliers is None:
                return None,None
            frame.set_world_to_camere(rvec_new,tvec_new)
            
            inliers = inliers.flatten()
        
            logging.debug(f"length of inliers {len(inliers)}")
            
            for i in inliers:
                landmarks[i].active=True
                landmarks[i].add_observation(id, matched_images[i],matched_des[i])
        return rvec_new, tvec_new
    
        



