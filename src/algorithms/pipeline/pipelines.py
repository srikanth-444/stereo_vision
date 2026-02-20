import numpy as np
import cv2
from ..loopclosure import Loop_closure
import time
import gtsam

class Pipeline():

    def __init__(self, landmark_manager, tracker, feature_extractor,frame_manager,depth_estimator,min_num_landmarks=100,visualizer=None):
        self.landmark_manager=landmark_manager
        self.frame_manager=frame_manager
        self.tracker=tracker
        self.feature_extractor=feature_extractor
        self.min_num_landmarks=min_num_landmarks
        self.depth_estimator=depth_estimator
        self.T=np.eye(4)
        self.path=[]
        self.path.append(self.T)
        self.current_frame=None
        self.visualizer=visualizer
        
        
        
    

    def process_frame(self, frames):
        print(f"Number of frames: {len(frames)}")
        kps={}
        des_s={}
        for frame in frames:
                kp, des=self.feature_extractor.extract_features(frame.frame)
                if frame.keyframe:
                    frame.bag_of_words=des
                kps[frame.camera_id]=kp
                des_s[frame.camera_id]=des

        return kps, des_s
    
    def estimate_depth(self, kps, des_s,camera_1,camera_2):
        if  self.depth_estimator.__class__.__name__ == "Stereo":
            start_time=round(time.time()*1000)
            matches = self.feature_extractor.bf.match(des_s[camera_1], des_s[camera_2])
            matches = sorted(matches, key=lambda x: x.distance)
            ptsL = np.float32([kps[camera_1][m.queryIdx].pt for m in matches]).reshape(-1,2)
            desL = np.array([des_s[camera_1][m.queryIdx] for m in matches])
            ptsR = np.float32([kps[camera_2][m.trainIdx].pt for m in matches]).reshape(-1,2)
            pts_2d=[ptsL,ptsR]
            pts_3d,reprojection_errors=self.depth_estimator.get_depth(pts_2d)
            pts_3d=pts_3d[reprojection_errors<0.5]
            ptsL=ptsL[reprojection_errors<0.5]
            desL=desL[reprojection_errors<0.5]
            
            return pts_3d, ptsL, desL

        if self.depth_estimator.__class__.__name__ == "rgbd":
            kp=kps[camera_1]
            pts_2d=[kp.pt for kp in kp]
            pts_3d=self.depth_estimator.get_depth(pts_2d)
            pts_3d_hom=np.hstack((pts_3d,np.ones((pts_3d.shape[0],1))))
            pts_3d_hom=(self.T@pts_3d_hom.T).T
            pts_3d=pts_3d_hom[:,:3]
            return pts_3d, pts_2d, des_s[camera_1]
        else:
            return None, None, None
            
    def create_new_landmarks(self, pts_3d, pts_2d, des, id):
        if pts_3d is not None:
                pts_3d_hom=np.hstack((pts_3d,np.ones((pts_3d.shape[0],1))))
                pts_3d_hom=(self.T@pts_3d_hom.T).T
                pts_3d=pts_3d_hom[:,:3]
                for pt_3d, pt, des in zip(pts_3d, pts_2d, des):
                    if self.landmark_manager.check_closest_point(pt_3d,threshold=1):
                        continue
                    self.landmark_manager.add_landmark(pt_3d, pt, des, id)
                    current_frame=self.frame_manager.frames[id]
                    current_frame.landmark_ids.append(self.landmark_manager.id_counter-1)
                return pts_3d

    def run(self,):
        previous_frame = None
        i=0
        
        while True:
            
            i=i+1
            frames=self.frame_manager.capture_frames()
            if i>10000 or len(frames)==0:
                # self.visualizer.visualize_as_point_cloud(self.path)
                exit(0)
            current_frame=[frame for frame in frames if frame.camera_id==0][0]
            kps,des=self.process_frame(frames)

            
            inlier_indices=None
            if previous_frame is not None and len(landmarks)>4:
                frame_ids=self.frame_manager.find_closest_keyframe(self.T, threshold=1)
                landmarks=[]
                for frame_id in frame_ids:
                    keyframe=self.frame_manager.frames[frame_id]
                    land_ids=keyframe.landmark_ids
                    for id in land_ids:
                        landmark=self.landmark_manager.get_landmark_by_id(id)
                        landmarks.append(landmark)
                if len(landmarks)==0:
                    landmarks=self.landmark_manager.get_active_landmarks()
                
                
                rvec,tvec,inlier_indices=self.tracker.track_landmarks(kps[current_frame.camera_id],des[current_frame.camera_id],landmarks)
                if rvec is None or tvec is None:
                    print("Tracking failed, skipping frame.")
                    previous_frame=current_frame
                    continue
                self.landmark_manager.update_active_landmark()
                R,_=cv2.Rodrigues(rvec)
                T=np.eye(4)
                T[:3,:3]=R.T
                T[:3,3]=-R.T@tvec.flatten()
                T=T @ np.linalg.inv(self.frame_manager.camera_map[current_frame.camera_id].get_extrinsic())
                self.T=T
                self.path.append(self.T)
                
            landmarks=self.landmark_manager.get_active_landmarks() 
            print(f"Number of active landmarks: {len(landmarks)}")   
            current_frame.T=self.T
            if len(landmarks)<self.min_num_landmarks:      
                pts_3d, pts, des=self.estimate_depth(kps, des,camera_1=current_frame.camera_id, camera_2=frames[0].camera_id if len(frames)>1 else None)
                if inlier_indices is not None:
                    mask = []
                    for i in range(len(pts)):
                        # Check if this specific pixel was already used by the tracker
                        if i not in inlier_indices:
                            mask.append(True)
                        else:
                            mask.append(False)
                    print(f"Number of new landmarks: {len(pts_3d)}")
                    pts_3d=self.create_new_landmarks(pts_3d[mask], pts[mask], des[mask], current_frame.id)
                else:
                    pts_3d=self.create_new_landmarks(pts_3d, pts, des, current_frame.id)
                self.frame_manager.set_keyframe(current_frame.id,threshold=0.1)

            previous_frame=current_frame
            self.visualizer.visualize_pipeline()
            self.visualizer.visualize_as_point_cloud(self.T)   
            
            

def pipeline_factory( landmark_manager, tracker, feature_extractor,camera,depth_estimator,min_num_landmarks=100,visualizer=None):
    return Pipeline(landmark_manager, tracker, feature_extractor,camera,depth_estimator,min_num_landmarks,visualizer)