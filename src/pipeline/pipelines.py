import numpy as np
import cv2
import time

from scipy.spatial.transform import Rotation as R
import logging
import threading
import objgraph
import gc
import cProfile
from ..atlas import Frame
from concurrent.futures import ThreadPoolExecutor

class Pipeline():

    def __init__(self, atlas, tracker,depth_estimator,motion_model,visualizer,camera_map,no_workers=4):
        self.atlas=atlas
        self.tracker=tracker
        
        
        self.depth_estimator=depth_estimator
        q=np.array([1,0,0,0],dtype=np.float32)
        t=np.array([0,0,0],dtype=np.float32).reshape(3,1)
        self.T=(q,t)
        self.path=[]
        
        self.current_frame=None
        self.visualizer=visualizer
        self.camera_map=camera_map
        self.motion_model=motion_model
        self.performance_logger=logging.getLogger("Performance")
        self.pipeline_logger=logging.getLogger("Pipeline   ")
        self.index=0

    def getFrames(self,):
        current_frames={}
        for camera_id, camera in self.camera_map.items():
            frame,timeStamp=camera.get_frame()
            if frame is not None:
                frame=Frame(self.index,frame,int(timeStamp),camera.intrinsic,camera.extrinsic,camera.feature_extractor)
                self.index+=1
                current_frames[camera_id]=frame
        return current_frames
        
    def process_frame(self, frames):
        self.pipeline_logger.debug(f"Number of frames: {len(frames)}")

        def process_single_frame(frame):
            start_time = time.time()
            frame.extractFeatures()
            feature_extraction_time = int((time.time() - start_time) * 1000)
            self.performance_logger.debug(f"feature_extraction {feature_extraction_time}ms")
        threads = []

        for frame in frames:
            t = threading.Thread(target=process_single_frame, args=(frame,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        
    
    def estimate_depth(self, left_frame, right_frame):
        if  self.depth_estimator.__class__.__name__ == "Stereo":

            #get un associated points
            start_time=time.time()
            kp_l=left_frame.getNotAssociatedPoints()
            des_l=left_frame.getNotAssociatedDescriptors()
            kp_r=right_frame.getNotAssociatedPoints()
            des_r=right_frame.getNotAssociatedDescriptors()
            left_not_associated_points=left_frame.getNotAssociatedIndices()
            point_time=int((time.time()-start_time)*1000)

            #matching stereo
            start_time=time.time()
            matches = self.depth_estimator.stereo_match(kp_l, kp_r, des_l, des_r)
            matches = sorted(matches, key=lambda x: x.distance)
            ptsL = np.float32([kp_l[m.queryIdx] for m in matches]).reshape(-1,2)
            ptsR = np.float32([kp_r[m.trainIdx] for m in matches]).reshape(-1,2)
            filteredkeypoint_ids=np.array([left_not_associated_points[m.queryIdx]for m in matches])
            stereo_match_time=int((time.time()-start_time)*1000)

            #triangulating
            start_time=time.time()
            pts_2d=[ptsL,ptsR]
            pts_3d,reprojection_errors=self.depth_estimator.get_depth(pts_2d)
            pts_3d=pts_3d[reprojection_errors<3]
            filteredkeypoint_ids=filteredkeypoint_ids[reprojection_errors<3]
            trinagulation_time=int((time.time()-start_time)*1000)
            self.performance_logger.debug(f"not ass points {point_time}ms | matching {stereo_match_time}ms | traingulation_time {trinagulation_time}ms")
            return pts_3d,filteredkeypoint_ids
        else:
            return None, None, None,None
            
    def create_new_landmarks(self, pts_3d,frame,keypoint_idx):
        if pts_3d is not None:
                landmarks=self.atlas.createLandmarks(pts_3d,frame,keypoint_idx)
                self.pipeline_logger.debug(f"no new landmarks {len(landmarks)}")
                if self.atlas.getLengthKeyFrame()>1:
                    self.merging_mappoints(frame, landmarks)
                    frame.updateCovisibility()
                start_time=time.time()
                keyframe = self.atlas.getAgedFrame(2)
                if keyframe is not None:
                    landmarks=keyframe.getLandmarks()
                    self.atlas.removeBadLandmarks(landmarks)
                self.performance_logger.debug(f"landmark removeal {int((time.time()-start_time)*1000)}")
                
        
    def merging_mappoints(self,frame,newlandmarks):
        frames=self.atlas.getClosestKeyFrames(frame,5)
        for frame in frames:
            frame.match(newlandmarks)
            self.atlas.mergeLandmarks(frame.mergers)
        landmarks = self.atlas.getLocalMap(frame)
        start_time=time.time()
        frame.match(landmarks)
        self.performance_logger.debug(f"projection matching {int((time.time()-start_time)*1000)}ms")
        start_time=time.time()
        self.pipeline_logger.debug(f"no of mergers detected {len(frame.mergers)}")
        self.atlas.mergeLandmarks(frame.mergers)
        self.performance_logger.debug(f"landmark merging {int((time.time()-start_time)*1000)}")
       
    def get_closest_landmarks(self):
        landmarks = []
        landmarks=self.atlas.getLastKeyFrame().getLandmarks()
        return landmarks


    def run(self,):
        previous_frame = None
        
        for i in range(0,10000):
            i=i+1
            self.pipeline_logger.debug(f"current iteration {i}")

            #capture frames
            start_time=time.time()
            frames=self.getFrames()
            if len(frames)==0:
                self.pipeline_logger.critical("No frames received from source. Shutting down pipeline.")
                exit(0)  
            left_frame=frames[0]
            right_frame=frames[1]
            capture_time=int((time.time()-start_time)*1000)

            #process frame
            start_time=time.time()
            self.process_frame(frames.values())
            process_time=int((time.time()-start_time)*1000)

            #tracking
            get_landmark_time=0
            pose_time=0
            tracking_time=0
            if previous_frame is not None :

                # predict T from motion model
                start_time=time.time()
                if len(self.path)>2:
                    qprev,tprev=self.path[-2]
                    qcurr,tcurr=self.path[-1]
                    self.T=self.motion_model(qprev,tprev,qcurr,tcurr)
                pose_time=int((time.time()-start_time)*1000)

                # Get landmarks to track
                start_time=time.time()
                landmarks=self.get_closest_landmarks()
                get_landmark_time=int((time.time()-start_time)*1000)

                # track landmarks
                start_time=time.time()
                self.tracker.track_landmarks(landmarks,self.T,[left_frame])
                self.T=left_frame.worldPose
                tracking_time= int((time.time()-start_time)*1000)     

            if previous_frame is None:
                c1=True
                c2=True
                c3=True
                c4=True
            else:
                no_tracked_landmarks=len(left_frame.getTrackedPoints())
                self.pipeline_logger.debug(f"no of tracked landmarks{no_tracked_landmarks}")
                c1=no_tracked_landmarks<left_frame.nVisible*0.6
                c2=(left_frame.id-self.atlas.getLastKeyFrame().id)>6
                frame=self.atlas.getLastKeyFrame()
                T=frame.worldPose
                c3=np.linalg.norm(T[1]-self.T[1])>0.1
                c4=no_tracked_landmarks<20
            depth_time=0
            landmarks_time=0
            keyframe_time=0
            self.pipeline_logger.debug(f"c1:{c1} ,c2:{c2} ,c3:{c3} ,c4:{c4}")
            
            # creating landmarks
            if (c1 and c2) and c3 or c4:
                # triangulate points
                start_time=time.time()   
                pts_3d,keypoint_idx=self.estimate_depth(left_frame,right_frame)
                depth_time=int((time.time()-start_time)*1000)

                # adding landmarks
                start_time=time.time() 
                self.atlas.setKeyframe(left_frame)

                self.pipeline_logger.debug(f"new keyframe id {left_frame.id}")
                previous_frame=left_frame
                keyframe_time=int((time.time()-start_time)*1000) 
                start_time=time.time()
                self.create_new_landmarks(pts_3d,left_frame,keypoint_idx)
                landmarks_time=int((time.time()-start_time)*1000) 
            self.pipeline_logger.debug(f"length of keyframes {self.atlas.getLengthKeyFrame()}")
            self.path.append(self.T)

            #gui update
            start_time=time.time()
            self.visualizer.visualize_as_point_cloud(None)
            self.visualizer.visualize_pipeline(left_frame)
            gui_time=int((time.time()-start_time)*1000)
            self.performance_logger.info(f"capture {capture_time}ms | Process {process_time}ms | close lm {get_landmark_time}ms | motion_model {pose_time}ms | tracking {tracking_time}ms | depth {depth_time}ms | keyframe {keyframe_time}ms | creat lm {landmarks_time}ms | gui {gui_time}ms")
            

def pipeline_factory( atlas, tracker,depth_estimator,motion_model,visualizer,camera_map=None):
    return Pipeline( atlas, tracker,depth_estimator,motion_model,visualizer,camera_map)