import numpy as np
import cv2
from collections import defaultdict

class Frame():
    def __init__(self,id,camera,frame, time_stamp, keyframe=False,):
        self.id=id
        self.camera=camera
        self.frame=frame
        self.timestamp=time_stamp
        self.keyframe=keyframe
        self.camera_center=None


        
        # self.keypoint_landmarks_association={}
        # self.landmarks_keypoint_association={}
        self.T=np.array([])
        self.keypoints=[]
        self.landmarks=[]
        self.descriptors=[]
        self.covisible = {}
        # self.height, self.width = 480, 752 
        # self.debug_img = np.zeros((self.height,self.width, 3), dtype=np.uint8)
        

    def get_camera_center(self,):
        if len(self.T)==0:
            self.camera_center=np.array([0,0,0])
        else:
            T=np.linalg.inv(self.T)
            self.camera_center= T[:3,3]
        return self.camera_center

    def set_world_to_camere(self,rvec,tvec):
        R, _ = cv2.Rodrigues(rvec)  # Rotation: world → camera
        t = tvec.flatten()           # Translation vector
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        self.T = T

    def set_keypoints(self, keypoints):
        self.keypoints=np.array(keypoints)
        self.image_points=self.keypoints[:,0:2]
        self.landmarks=[None]*len(self.keypoints)
        self.not_associated_points=set(range(self.image_points.shape[0]))

    def drop_not_associated_points(self,):
        if not self.not_associated_points:
            return  # nothing to drop

        keep_indices = sorted(set(range(self.image_points.shape[0])) - self.not_associated_points)

        # Filter keypoints
        self.keypoints = self.keypoints[keep_indices]

        # Filter image_points
        self.image_points = self.image_points[keep_indices]

        # Filter descriptors
        if len(self.descriptors) > 0:
            self.descriptors = self.descriptors[keep_indices]

        # Reset not_associated_points
        self.not_associated_points = set()
        

    def set_descriptors(self,decriptors):
        self.descriptors=np.array(decriptors, dtype=np.uint8)

    def get_not_associated_kps(self,):
        return self.image_points[list(self.not_associated_points)]
    
    def get_not_associated_des(self,):
        return self.descriptors[list(self.not_associated_points)]
    
    def get_landmarks(self,):
        return [lm for lm in self.landmarks if lm is not None]
    
    def get_tracked_points(self,):
        associated_idx = list(set(range(len(self.image_points))) - self.not_associated_points)
        return self.image_points[associated_idx]
    
    def update_covisibility(self,):
        landmarks=self.get_landmarks()
        counter = {}
        for lm in landmarks:
            for kf in lm.observations.keys():
                if kf == self:
                    continue
                counter[kf] = counter.get(kf, 0) + 1
        for kf, weight in counter.items():

            if weight < 15:
                continue

            self.covisible[kf] = weight
            kf.covisible[self] = weight
    
    def project_landmarks(self, pts_3d, T=None):

        ones = np.ones((pts_3d.shape[0], 1))
        pts3d_hom = np.hstack((pts_3d, ones))  # (N,4)

        # Transform to camera frame
        if T is None:
            pts_cam = self.T @ pts3d_hom.T   
        else:
            pts_cam = np.linalg.inv(T@self.camera.extrinsic) @ pts3d_hom.T          # (4,N)

        # Z = pts_cam[2, :]                      # depth values

        # Project to image
        pts_2d = self.camera.intrinsic @ pts_cam[:3, :]    # (3,N)
        pts_2d = pts_2d / pts_2d[2, :]         # normalize
        pts_2d=pts_2d[:2,:]

        # # Depth binning
        # depth_bins = (Z / 2.5).astype(int).reshape(1, -1)

        # # Stack depth bin
        # pts_2d_with_bins = np.vstack((pts_2d, depth_bins))

        return pts_2d.T
  

    def projection_match(self,landmarks,T=None):
        
        matched_idx=set()
        object_points = np.array([lm.position for lm in landmarks])
        landmark_desc = np.array([lm.descriptor for lm in landmarks],dtype=np.uint8)
        projected_points =self.project_landmarks(object_points,T)
        final_matches=[]
        global_matches=[]
        matches = self.camera.feature_extractor.bf.knnMatch(landmark_desc, self.descriptors, k=2)
        for i, m_list in enumerate(matches):
            if len(m_list) < 2: continue
            m, n = m_list[0], m_list[1]

            if m.distance < 0.9* n.distance and m.distance < 40:
                if m.trainIdx in matched_idx:
                    continue
                global_matches.append(m)
                matched_idx.add(m.trainIdx)
                matched_kp_pt = self.image_points[m.trainIdx]
                proj_pt = projected_points[m.queryIdx]
                dist_sq = np.sum((matched_kp_pt - proj_pt)**2)
                if dist_sq < 100: # 10 pixel radius
                    m.queryIdx = i # landmark index
                    final_matches.append(m)
                     
        if len(final_matches)<20:
            final_matches=global_matches
        for m in final_matches:
            self.landmarks[m.trainIdx]=landmarks[m.queryIdx]
        matched_objects=np.array([object_points[m.queryIdx] for m in final_matches])
        matched_images=np.array([self.image_points[m.trainIdx] for m in final_matches])
        matched_indices = {m.trainIdx for m in final_matches}  # indices of matched keypoints
        self.not_associated_points -= matched_indices        # remove them
        matched_des =[self.descriptors[m.trainIdx] for m in final_matches]
        

        # self.plot_points(projected_points,(0,0,255))
        # self.plot_points(self.image_points,(255,0,0))
        # for m in final_matches:
        #     pt1 = tuple(projected_points[m.queryIdx][:2].astype(int))
        #     pt2 = tuple(self.image_points[m.trainIdx].astype(int))
        #     cv2.line(self.debug_img, pt1, pt2, (0, 255, 0), 1)
       
        # cv2.imshow("Projection Debug", self.debug_img)
        # cv2.waitKey(1)  # <-- Blocking! Waits until any key is pressed
        return matched_objects, matched_images, matched_des
    
    def projection_match_merger(self,landmarks):
        merging_land = []
        object_points = np.array([lm.position for lm in landmarks])
        landmark_desc = np.array([lm.descriptor for lm in landmarks],dtype=np.uint8)
        projected_points =self.project_landmarks(object_points)
        matches = self.camera.feature_extractor.bf.knnMatch(landmark_desc, self.descriptors, k=2)
        for i, m_list in enumerate(matches):
            if len(m_list) < 2: continue
            m, n = m_list[0], m_list[1]
            if m.distance < 0.7 * n.distance and m.distance < 40:
                matched_kp_pt = self.image_points[m.trainIdx]
                proj_pt = projected_points[m.queryIdx]
                dist_sq = np.sum((matched_kp_pt - proj_pt)**2)
                if dist_sq < 100:
                    m.queryIdx = i
                    if self.landmarks[m.trainIdx]==None:
                        self.landmarks[m.trainIdx]=landmarks[m.queryIdx]
                    else:
                        merging_land.append([landmarks[m.queryIdx],self.landmarks[m.trainIdx]])
                    
        return merging_land
    def plot_points(self,points,color):
        # Draw projected landmarks in red
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < self.width and 0 <= y <self.height:
                cv2.circle(self.debug_img, (x, y), 3, color, -1)


        