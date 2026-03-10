import numpy as np
from .pnp import PNPSolver
from .tracker import Tracker
import logging 
import time
import threading
import queue

class DescriptorTracker(Tracker):
    def __init__(self,reprojection_error,confidence,iterationsCount,landmark_manager,camera) -> None:
        
        self.pnp_solver=PNPSolver(reprojection_error,confidence,iterationsCount,camera)
        self.landmark_manager=landmark_manager
        self.tracker_logger=logging.getLogger('Tracker')
        # self.thread=threading.Thread(target=self.add_observations,daemon=True)
        # self.queue = queue.Queue(maxsize=300)
        # self.thread.start()
    
    def track_landmarks(self,landmarks, T, frames):
        self.tracker_logger.debug(f"length of landmarks {len(landmarks)}")

        if len(landmarks) < 6:
            return None, None
        start_time=time.time()
        for lm in landmarks:
            lm.active=False
            lm.nvisible=lm.nvisible+1
        visibility_addition=(time.time()-start_time)*1000
        self.tracker_logger.debug(f"visibility {visibility_addition}ms |")
        for frame in frames:
            start_time=time.time()
            matched_objects, matched_images, matched_des=frame.projection_match(landmarks,T)
            matching_time=(time.time()-start_time)*1000
            self.tracker_logger.debug(f"length of matched points {len(matched_objects)}")
            if len(matched_objects) < 8:
                return None, None
            start_time=time.time()
            rvec_new, tvec_new, inliers = self.pnp_solver.estimate_pose_pnp(matched_objects,matched_images,frame.camera.intrinsic)
            pose_estimation_time=(time.time()-start_time)*1000

            self.tracker_logger.debug(f"matching {matching_time}ms | pose_estimation_time {pose_estimation_time}ms")
            if inliers is None:
                return None,None
            frame.set_world_to_camere(rvec_new,tvec_new)
            
            inliers = inliers.flatten()
        
            self.tracker_logger.debug(f"length of inliers {len(inliers)}")
            start_time=start_time
            
            for i in inliers:
                landmarks[i].active=True
                # landmarks[i].tracked=landmarks[i].tracked+1
                # self.queue.put((landmarks[i],frames[0].id, matched_images[i],matched_des[i]))
                # landmarks[i].add_observation(frame.id, matched_images[i],matched_des[i])
            inliers_addition=(time.time()-start_time)*1000
            self.tracker_logger.debug(f"inlers_Addition {inliers_addition}ms")
        return rvec_new, tvec_new
    

    # def add_observations(self):
    #     while True:
    #         landmark,frame_id,matched_image,matched_des=self.queue.get()
    #         landmark.add_observation(frame_id, matched_image,matched_des)
    #         self.queue.task_done()

    
        



