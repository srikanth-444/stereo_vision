import numpy as np
from .pnp import PNPSolver
import cv2
from scipy.spatial.transform import Rotation as R_scipy
import logging 
import time
import threading
import queue

class Tracker():
    def __init__(self,config_load,optimizer,camera,atlas,motion_model) -> None:
        reprojection_error=config_load.get('reprojection_error',{})
        confidence =config_load.get('confidence',{})
        iterationsCount=config_load.get('iterationsCount',{})
        camera=camera
        self.pnp_solver=PNPSolver(reprojection_error,confidence,iterationsCount,camera)
        self.tracker_logger=logging.getLogger('Tracker    ')
        self.optimizer=optimizer
        self.atlas=atlas
        self.tracking_state="good"
        self.motion_model=motion_model
        q=np.array([0,0,0,1],dtype=np.float32)
        t=np.array([0,0,0],dtype=np.float32).reshape(3,1)
        self.T=(q,t)
        self.path=[]
        self.map_initialized=False

    def PoseFromMotionModel(self,):
        
        if len(self.path)>2 and self.tracking_state=="good":
                    qprev,tprev=self.path[-2]
                    qcurr,tcurr=self.path[-1]
                    self.T=self.motion_model(qprev,tprev,qcurr,tcurr)
        

    def get_closest_landmarks(self,type, frame):
        landmarks = []
        if type:
            landmarks=self.currentMap.getLastKeyFrame().getLandmarks()
            landmarks=frame.getVisibleLandmarks(landmarks)
        else:
            landmarks=self.currentMap.getLocalMap(self.currentMap.getLastKeyFrame())
            landmarks=frame.getVisibleLandmarks(landmarks)
        return landmarks
    
    def rvec_to_quaternion(self,rvec,tvec):
        R, _ = cv2.Rodrigues(rvec)
        R_wc = R.T
        t_wc = -R_wc @ tvec
        q_wc = R_scipy.from_matrix(R_wc).as_quat()
        return q_wc, t_wc
    
    def track(self, left_frame, right_frame=None):
        
        self.currentMap=self.atlas.getActiveMap()
        start_time=time.time()
        self.PoseFromMotionModel()
        q,t=self.T[0],self.T[1]
        left_frame.setCameraWorldPose(q,t)  
        if(not self.map_initialized):
            self.map_initialized=True
            return True
        landmarks=self.get_closest_landmarks(True,left_frame) 
        matched_objects, matched_images=left_frame.projectionMatch(landmarks,50)
        if right_frame is not None:
            right_frame.setCameraWorldPose(q,t)
            landmarks=right_frame.getVisibleLandmarks(landmarks) 
            matched_objects, matched_images=right_frame.projectionMatch(landmarks,50)
        self.tracker_logger.debug(f"landmarks that are visible {len(landmarks)}")
        self.tracker_logger.debug(f"length of matched points {len(matched_objects)}")
        matching_time=(time.time()-start_time)*1000
        if len(matched_objects) < 20:
            matched_objects, matched_images=left_frame.match(landmarks)
            self.tracker_logger.debug(f"length of matched points descriptor based {len(matched_objects)}")
        if len(matched_objects) < 20:
            self.tracking_state="lost"
            
        start_time=time.time()
        self.tracker_logger.debug(f"calling optimize pose")
        if not self.optimizer.optimizePose(left_frame):
             self.tracking_state="lost"
             return True
        landmarks=self.get_closest_landmarks(False,left_frame)
        matched_objects, matched_images=left_frame.projectionMatch(landmarks,10)
        if right_frame is not None:
            landmarks=right_frame.getVisibleLandmarks(landmarks) 
            matched_objects, matched_images=right_frame.projectionMatch(landmarks,10)
        self.tracker_logger.debug(f"landmarks that are visible {len(landmarks)}")  
        self.tracker_logger.debug(f"length of matched points {len(matched_objects)}")
        self.optimizer.optimizePose(left_frame)
        pose_estimation_time=(time.time()-start_time)*1000
        self.tracker_logger.debug(f"matching {int(matching_time)}ms | pose_estimation_time {int(pose_estimation_time)}ms")   
        no_tracked_landmarks=len(left_frame.getTrackedPoints())
        self.tracker_logger.debug(f"no of tracked landmarks{no_tracked_landmarks}")
        c1=no_tracked_landmarks<left_frame.nVisible*0.9
        c2=(left_frame.id-self.currentMap.getLastKeyFrame().id)>6
        mframe=self.currentMap.getLastKeyFrame()
        T=mframe.worldPose
        c3=np.linalg.norm(T[1]-self.T[1])>0.1
        # c4=no_tracked_landmarks<20
        self.T=left_frame.worldPose
        self.path.append(self.T)         
        return (c1 and c2)  or self.tracking_state=="lost"


