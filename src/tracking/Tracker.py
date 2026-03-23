import numpy as np
from .pnp import PNPSolver
import logging 
import time
import threading
import queue

class Tracker():
    def __init__(self,reprojection_error,confidence,iterationsCount,camera,optimizer) -> None:
        
        self.pnp_solver=PNPSolver(reprojection_error,confidence,iterationsCount,camera)
        self.tracker_logger=logging.getLogger('Tracker    ')
        self.optimizer=optimizer

    
    def track_landmarks(self,landmarks, T, frames):
        self.tracker_logger.debug(f"length of landmarks {len(landmarks)}")
        for frame in frames:
            start_time=time.time()
            q,t=T[0],T[1]
            frame.setCameraWorldPose(q,t)
            landmarks=frame.getVisibleLandamrks(landmarks)
            self.tracker_logger.debug(f"landmarks that are visible {len(landmarks)}")
            matched_objects, matched_images=frame.projectionMatch(landmarks)
            matching_time=(time.time()-start_time)*1000
            self.tracker_logger.debug(f"length of matched points {len(matched_objects)}")
            if len(matched_objects) < 8:
                matched_objects, matched_images=frame.match(landmarks)
                self.tracker_logger.debug(f"length of matched points {len(matched_objects)}")
            start_time=time.time()
            self.optimizer.optimizePose(frame)
            pose_estimation_time=(time.time()-start_time)*1000
            self.tracker_logger.debug(f"matching {int(matching_time)}ms | pose_estimation_time {int(pose_estimation_time)}ms")

    
        



