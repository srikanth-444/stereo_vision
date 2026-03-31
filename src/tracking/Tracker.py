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
    
    def track(self, frames):
        self.currentMap=self.atlas.getActiveMap()
        for frame in frames:
            start_time=time.time()
            self.PoseFromMotionModel()
            q,t=self.T[0],self.T[1]
            frame.setCameraWorldPose(q,t)
            try:
                landmarks=self.get_closest_landmarks(True,frame)
            except ValueError:
                 return True
            self.tracker_logger.debug(f"landmarks that are visible {len(landmarks)}")
            matched_objects, matched_images=frame.projectionMatch(landmarks,50)
            matching_time=(time.time()-start_time)*1000
            self.tracker_logger.debug(f"length of matched points {len(matched_objects)}")
            if len(matched_objects) < 20:
                matched_objects, matched_images=frame.match(landmarks)
                self.tracker_logger.debug(f"length of matched points descriptor based {len(matched_objects)}")
                
            if len(matched_objects) < 20:
                self.tracking_state="lost"
            # rvec,tvec,inliers=self.pnp_solver.estimate_pose_pnp(matched_objects,matched_images,frame.intrinsic)
            # q,t=self.rvec_to_quaternion(rvec,tvec)
            # frame.setCameraWorldPose(q,t)

            # print(self.tracking_state)
            start_time=time.time()
            self.optimizer.optimizePose(frame)
            landmarks=self.get_closest_landmarks(False,frame)
            self.tracker_logger.debug(f"landmarks that are visible {len(landmarks)}")
            matched_objects, matched_images=frame.projectionMatch(landmarks,10)
            self.tracker_logger.debug(f"length of matched points {len(matched_objects)}")
            self.optimizer.optimizePose(frame)
            pose_estimation_time=(time.time()-start_time)*1000
            self.tracker_logger.debug(f"matching {int(matching_time)}ms | pose_estimation_time {int(pose_estimation_time)}ms")
            
            no_tracked_landmarks=len(frame.getTrackedPoints())
            self.tracker_logger.debug(f"no of tracked landmarks{no_tracked_landmarks}")
            c1=no_tracked_landmarks<frame.nVisible*0.25
            c2=(frame.id-self.currentMap.getLastKeyFrame().id)>6
            mframe=self.currentMap.getLastKeyFrame()
            T=mframe.worldPose
            c3=np.linalg.norm(T[1]-self.T[1])>0.1
            # c4=no_tracked_landmarks<20
            self.T=frame.worldPose
            self.path.append(self.T)
            return (c1 and c2) and c3 or self.tracking_state=="lost"
    
        



