import numpy as np
from .pnp import PNPSolver
import logging 
import time
import threading
import queue

class Tracker():
    def __init__(self,config_load,optimizer,camera,atlas) -> None:
        reprojection_error=config_load.get('reprojection_error',{})
        confidence =config_load.get('confidence',{})
        iterationsCount=config_load.get('iterationsCount',{})
        camera=camera
        self.pnp_solver=PNPSolver(reprojection_error,confidence,iterationsCount,camera)
        self.tracker_logger=logging.getLogger('Tracker    ')
        self.optimizer=optimizer
        self.atlas=atlas


    def get_closest_landmarks(self,type, frame):
        landmarks = []
        if type:
            landmarks=self.currentMap.getLastKeyFrame().getLandmarks()
            landmarks=frame.getVisibleLandmarks(landmarks)
        else:
            landmarks=self.currentMap.getLocalMap(self.currentMap.getLastKeyFrame())
            landmarks=frame.getVisibleLandmarks(landmarks)
        return landmarks
    
    def track_landmarks(self,T, frames):
        self.currentMap=self.atlas.getActiveMap()
        for frame in frames:
            start_time=time.time()
            q,t=T[0],T[1]
            frame.setCameraWorldPose(q,t)
            landmarks=self.get_closest_landmarks(True,frame)
            self.tracker_logger.debug(f"landmarks that are visible {len(landmarks)}")
            matched_objects, matched_images=frame.projectionMatch(landmarks)
            matching_time=(time.time()-start_time)*1000
            self.tracker_logger.debug(f"length of matched points {len(matched_objects)}")
            if len(matched_objects) < 20:
                matched_objects, matched_images=frame.match(landmarks)
            #     self.tracker_logger.debug(f"length of matched points {len(matched_objects)}")
            # if len(matched_objects) < 20:
            #     return False
            
            start_time=time.time()
            self.optimizer.optimizePose(frame)
            landmarks=self.get_closest_landmarks(False,frame)
            self.tracker_logger.debug(f"landmarks that are visible {len(landmarks)}")
            matched_objects, matched_images=frame.projectionMatch(landmarks)
            self.tracker_logger.debug(f"length of matched points {len(matched_objects)}")
            self.optimizer.optimizePose(frame)
            pose_estimation_time=(time.time()-start_time)*1000
            self.tracker_logger.debug(f"matching {int(matching_time)}ms | pose_estimation_time {int(pose_estimation_time)}ms")
            return True
    
        



