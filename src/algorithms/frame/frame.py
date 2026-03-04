import numpy as np
from scipy.spatial import KDTree
import cv2

class Frame():
    def __init__(self,id,camera_id, frame, time_stamp,intrinsic,extrinsic,keyframe=False,):
        self.id=id
        self.camera_id=camera_id
        self.frame=frame
        self.timestamp=time_stamp
        self.keyframe=keyframe
        self.intrinsic=intrinsic
        self.extrinsic=extrinsic

        self.camera_center=None
        self.keypoint_landmarks_association={}
        self.all_descriptors=[]
        self.bag_of_words=[]
        self.T=np.array([])
        self.keypoints=[]
        self.tree=[]
        self.height, self.width = 480, 752 
        self.debug_img = np.zeros((self.height,self.width, 3), dtype=np.uint8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        

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
        self.image_points=np.array([kp.pt for kp in self.keypoints])
        self.not_associated_points=set(range(len(self.image_points)))
        self.tree = KDTree(self.image_points)

    def set_descriptors(self,decriptors):
        self.all_descriptors=decriptors

    def get_not_associated_kps(self,):
        return self.keypoints[list(self.not_associated_points)]
    
    def get_not_associated_des(self,):
        return self.all_descriptors[list(self.not_associated_points)]
    
    def get_land_ids(self,):
        return self.keypoint_landmarks_association.values()
    
    def get_tracked_points(self,):
        return self.image_points[list(self.keypoint_landmarks_association.keys())]
    
    def project_landmarks(self, pts_3d, T=None):

        ones = np.ones((pts_3d.shape[0], 1))
        pts3d_hom = np.hstack((pts_3d, ones))  # (N,4)

        # Transform to camera frame
        if T is None:
            pts_cam = self.T @ pts3d_hom.T   
        else:
            pts_cam = np.linalg.inv(T@self.extrinsic) @ pts3d_hom.T          # (4,N)

        # Z = pts_cam[2, :]                      # depth values

        # Project to image
        pts_2d = self.intrinsic @ pts_cam[:3, :]    # (3,N)
        pts_2d = pts_2d / pts_2d[2, :]         # normalize
        pts_2d=pts_2d[:2,:]

        # # Depth binning
        # depth_bins = (Z / 2.5).astype(int).reshape(1, -1)

        # # Stack depth bin
        # pts_2d_with_bins = np.vstack((pts_2d, depth_bins))

        return pts_2d.T
    def global_matches(self,landmark_desc,land_ids):
        final_matches=[]
        matches = self.bf.knnMatch(landmark_desc, self.all_descriptors,k=2)

        for m, n in matches:
            # Lowe's ratio test
            if m.distance < 0.9 * n.distance and m.distance < 40:
                    final_matches.append(m)
                    self.keypoint_landmarks_association[m.trainIdx]=land_ids[m.queryIdx] 

        return final_matches

    def projection_match(self,landmarks,T=None):
        
        
        object_points = np.array([lm.position for lm in landmarks])
        landmark_desc = np.array([lm.descriptor for lm in landmarks],dtype=np.uint8)
        land_ids=[lm.id for lm in landmarks]
        projected_points =self.project_landmarks(object_points,T)
        final_matches=[]
        matches = self.bf.knnMatch(landmark_desc, self.all_descriptors, k=2)
        for i, m_list in enumerate(matches):
            if len(m_list) < 2: continue
            m, n = m_list[0], m_list[1]

            # 3. Ratio & Distance Thresholds
            if m.distance < 0.9 * n.distance and m.distance < 40:
                
                # 4. THE GEOMETRIC MASK (Replaces the KD-Tree)
                # Check if the matched keypoint is actually near where we projected it
                matched_kp_pt = self.image_points[m.trainIdx]
                proj_pt = projected_points[i]
                
                # L2 distance check: sqrt((x1-x2)^2 + (y1-y2)^2) < 10
                # Faster version: squared distance < 100
                dist_sq = np.sum((matched_kp_pt - proj_pt)**2)
                
                if dist_sq < 100: # 10 pixel radius
                    m.queryIdx = i # landmark index
                    final_matches.append(m)
                    self.keypoint_landmarks_association[m.trainIdx] = land_ids[i] 
        # if len(final_matches)<20:
        # final_matches=self.global_matches(landmark_desc,land_ids)
        matched_objects=np.array([object_points[m.queryIdx] for m in final_matches])
        matched_images=np.array([self.image_points[m.trainIdx] for m in final_matches])
        matched_indices = {m.trainIdx for m in final_matches}  # indices of matched keypoints
        self.not_associated_points -= matched_indices        # remove them
        matched_des =[self.all_descriptors[m.trainIdx] for m in final_matches]
        

        # self.plot_points(projected_points,(0,0,255))
        # self.plot_points(self.image_points,(255,0,0))
        # for m in final_matches:
        #     pt1 = tuple(projected_points[m.queryIdx][:2].astype(int))
        #     pt2 = tuple(self.image_points[m.trainIdx].astype(int))
        #     cv2.line(self.debug_img, pt1, pt2, (0, 255, 0), 1)
       
        # cv2.imshow("Projection Debug", self.debug_img)
        # cv2.waitKey(1)  # <-- Blocking! Waits until any key is pressed
        return matched_objects, matched_images, matched_des
    
    def plot_points(self,points,color):
        # Draw projected landmarks in red
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < self.width and 0 <= y <self.height:
                cv2.circle(self.debug_img, (x, y), 3, color, -1)


        