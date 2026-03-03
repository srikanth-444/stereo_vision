import numpy as np
import cv2
from ..loopclosure import Loop_closure
import time
import gtsam
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

class Pipeline():

    def __init__(self, landmark_manager, tracker, feature_extractor,frame_manager,depth_estimator,min_num_landmarks=100,visualizer=None,camera_map=None):
        self.landmark_manager=landmark_manager
        self.frame_manager=frame_manager
        self.tracker=tracker
        self.feature_extractor=feature_extractor
        self.min_num_landmarks=min_num_landmarks
        self.depth_estimator=depth_estimator
        self.T=np.eye(4)
        self.path=[]
        
        self.current_frame=None
        self.visualizer=visualizer
        self.camera_map=camera_map
        
    def process_frame(self, frames):
        print(f"Number of frames: {len(frames)}")
        kps={}
        des_s={}
        for frame in frames:
                kp, des=self.feature_extractor.extract_features(frame.frame)
                frame.set_keypoints(kp)
                frame.set_descriptors(des)
                kps[frame.camera_id]=kp
                des_s[frame.camera_id]=des

        return kps, des_s
    
    def estimate_depth(self, left_frame, right_frame):
        if  self.depth_estimator.__class__.__name__ == "Stereo":
            kp_l=left_frame.get_not_associated_kps()
            des_l=left_frame.get_not_associated_des()
            kp_r=right_frame.get_not_associated_kps()
            des_r=right_frame.get_not_associated_des()
            left_not_associated_points=list(left_frame.not_associated_points)
            matches = self.depth_estimator.stereo_match(kp_l, kp_r, des_l, des_r)
            matches = sorted(matches, key=lambda x: x.distance)
            ptsL = np.float32([kp_l[m.queryIdx].pt for m in matches]).reshape(-1,2)
            desL = np.array([des_l[m.queryIdx] for m in matches])
            ptsR = np.float32([kp_r[m.trainIdx].pt for m in matches]).reshape(-1,2)
            filteredkeypoint_ids=np.array([left_not_associated_points[m.queryIdx]for m in matches])
            pts_2d=[ptsL,ptsR]
            pts_3d,reprojection_errors=self.depth_estimator.get_depth(pts_2d)
            depth_bins=(pts_3d[:,2]/2.5).astype(int)
            pts_3d=pts_3d[reprojection_errors<3]
            ptsL=ptsL[reprojection_errors<3]
            desL=desL[reprojection_errors<3]
            filteredkeypoint_ids=filteredkeypoint_ids[reprojection_errors<3]
            
            return pts_3d, ptsL, desL,depth_bins,filteredkeypoint_ids
        else:
            return None, None, None,None
            
    def create_new_landmarks(self, pts_3d, pts_2d, des_s, id,camera_center,keypoint_idx):
        if pts_3d is not None:
                pts_3d_hom=np.hstack((pts_3d,np.ones((pts_3d.shape[0],1))))
                pts_3d_hom=(self.T@pts_3d_hom.T).T
                pts_3d=pts_3d_hom[:,:3]
                for pt_3d, pt, des, idx in zip(pts_3d, pts_2d, des_s,keypoint_idx):

                    # if self.landmark_manager.check_closest_point(id,pt,des,pt_3d,threshold=0.1):
                    #     continue
                    landmark=self.landmark_manager.add_landmark(pt_3d, pt, des, id)
                    # landmark.set_normal(camera_center)
                    # landmark.set_reference_depth(camera_center)
                    self.frame_manager.frames[id].keypoint_landmarks_association[idx]=landmark.id
                if self.frame_manager.get_len_keyframes()>3:
                    frame_id = self.frame_manager.find_closest_keyframe()[0]
                    landmark_ids = set()
                    keyframe = self.frame_manager.frames[frame_id]
                    landmark_ids.update(keyframe.get_land_ids())
                    self.landmark_manager.remove_bad_landmarks(landmark_ids)
                return pts_3d
       
    def compute_trajectory(self,rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)

        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ tvec.flatten()
        extrinsic= self.camera_map[0].get_extrinsic()
        return T@np.linalg.inv(extrinsic)
    
    def get_closest_landmarks(self):
        landmark_ids = set()
        # Always add active landmarks
        active_ids = self.frame_manager.get_last_keyframe().get_land_ids()
        landmark_ids.update(active_ids)

        

        # Convert back to landmark objects
        landmarks = [
            self.landmark_manager.get_landmark_by_id(lm_id)
            for lm_id in landmark_ids
        ]
        landmarks=[lm for lm in landmarks if lm is not None]
        if len(landmarks)<100:
            frame_ids = self.frame_manager.find_closest_keyframe()
            for frame_id in frame_ids:
                keyframe = self.frame_manager.frames[frame_id]
                landmark_ids.update(keyframe.get_land_ids())
        landmarks = [
            self.landmark_manager.get_landmark_by_id(lm_id)
            for lm_id in landmark_ids
        ]
        landmarks=[lm for lm in landmarks if lm is not None]
        return landmarks

    def predict_next_pose(self,T_prev, T_curr,dt):
        # Extract rotation and translation
        R_prev = T_prev[:3,:3]
        t_prev = T_prev[:3,3]
        
        R_curr = T_curr[:3,:3]
        t_curr = T_curr[:3,3]
        
        # Linear velocity
     
        v = (t_curr - t_prev) / dt
        
        # Angular velocity
        R_delta = R.from_matrix(R_prev.T @ R_curr)
        omega = R_delta.as_rotvec() / dt  # rad/s
        
        # Predict next translation
        t_next = t_curr + v * dt
        
        # Predict next rotation
        R_next = R.from_matrix(R_curr) * R.from_rotvec(omega * dt)
        T_next = np.eye(4)
        T_next[:3,:3] = R_next.as_matrix()
        T_next[:3,3] = t_next
        
        return T_next

    def run(self,):
        previous_frame = None
        for i in range(0,10000):
            
            i=i+1
            
            frames=self.frame_manager.capture_frames()
            if len(frames)==0:
                exit(0)
            current_frame=[frame for frame in frames if frame.camera_id==0][0]
            start_time=time.time()
            self.process_frame(frames)
            print(f"process {(time.time()-start_time)*1000}ms")
            start_time=time.time()
            len_landmarks=np.inf
            no_tracked_landmarks=0
            if previous_frame is not None :
                landmarks=self.get_closest_landmarks()
                len_landmarks=len(landmarks)
                if len(self.path)>2:
                    self.T=self.predict_next_pose(self.path[-2],self.path[-1],0.05)
                print(f"time to get close landmarks {(time.time()-start_time)*1000}ms")
                start_time=time.time()
                rvec,tvec=self.tracker.track_landmarks(landmarks,self.T,[current_frame])
                if rvec is None or tvec is None:
                    print("Tracking failed, skipping frame.")
                    previous_frame=current_frame
                    continue
                self.landmark_manager.update_active_landmark(landmarks)
                no_tracked_landmarks=self.landmark_manager.num_of_active_landmarks()
                # self.landmark_manager.active_ids=set()
                print(f"no of tracked landmarks{no_tracked_landmarks}")
                self.T=self.compute_trajectory(rvec, tvec)  
            print(f"tracking {(time.time()-start_time)*1000}ms")
            
            
            if no_tracked_landmarks<len_landmarks*0.3:
                start_time=time.time()   
                pts_3d, pts, des, depth_bins, keypoint_idx=self.estimate_depth(current_frame,[frame for frame in frames if frame.camera_id==1][0])
                print(f"depth estimation {(time.time()-start_time)*1000}ms") 
                start_time=time.time() 
                pts_3d=self.create_new_landmarks(pts_3d, pts, des, current_frame.id, current_frame.get_camera_center(),keypoint_idx)
                self.frame_manager.set_keyframe(current_frame.id)
                print(f"landmarks addition {(time.time()-start_time)*1000}ms")   
            print(f" length of keyframes {self.frame_manager.get_len_keyframes()}")

            previous_frame=current_frame
            self.path.append(self.T)
            self.visualizer.visualize_pipeline(current_frame)
            self.visualizer.visualize_as_point_cloud(self.T)  
            
            
            

def pipeline_factory( landmark_manager, tracker, feature_extractor,camera,depth_estimator,min_num_landmarks=100,visualizer=None,camera_map=None):
    return Pipeline(landmark_manager, tracker, feature_extractor,camera,depth_estimator,min_num_landmarks,visualizer,camera_map)