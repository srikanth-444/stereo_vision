import cv2
import numpy as np
from .pnp import PNPSolver
from .tracker import Tracker
from scipy.spatial import KDTree
import matplotlib.pyplot as plt 
from collections import defaultdict

class DescriptorTracker(Tracker):
    def __init__(self,reprojection_error,confidence,iterationsCount,landmark_manager,camera) -> None:
        
        self.pnp_solver=PNPSolver(reprojection_error,confidence,iterationsCount,camera)
        self.landmark_manager=landmark_manager
    
    
    def track_landmarks(self,landmarks, T, frames):
        print(f"length of landmarks {len(landmarks)}")

        if len(landmarks) < 6:
            return None, None
        for lm in landmarks:
            lm.active=False
            lm.nvisible=lm.nvisible+1
        for frame in frames:
            matched_objects, matched_images, matched_des=frame.projection_match(landmarks,T)
            print(f"length of matched points {len(matched_objects)}")
            if len(matched_objects) < 20:
                return None, None
            rvec_new, tvec_new, inliers = self.pnp_solver.estimate_pose_pnp(matched_objects,matched_images,frame.intrinsic)
            if inliers is None:
                return None,None
            frame.set_world_to_camere(rvec_new,tvec_new)
            
            inliers = inliers.flatten()
        
            # print(f"length of inliers {len(inliers)}")
            
            for i in inliers:
                landmarks[i].active=True
                landmarks[i].add_observation(id, matched_images[i],matched_des[i])
        return rvec_new, tvec_new
    
    # def track_from_previous_frame(self,kp, des, intrinsic, distortion, landmarks, id):
    #     for lm in landmarks:
    #         lm.active=False
    #     object_points = np.array([lm.position for lm in landmarks])
    #     landmark_des = np.array([lm.descriptor for lm in landmarks],dtype=np.uint8)
    #     image_points = np.array([k.pt for k in kp])
    #     matches = self.feature_extractor.bf.match(landmark_des,des)
    #     matches = sorted(matches, key=lambda x: x.distance)
    #     matched_land_idx={}
    #     matched_objects=[]
    #     matched_image_idx={}
    #     matched_images=[]
    #     for m in matches:
    #         matched_objects.append(object_points[m.queryIdx])
    #         matched_land_idx[len(matched_objects)-1]=m.queryIdx
    #         matched_images.append(image_points[m.trainIdx])
    #         matched_image_idx[len(matched_images)-1]=m.trainIdx
    #     obj_p    = np.array(matched_objects)
    #     img_p    = np.array(matched_images) 
    #     rvec_new, tvec_new, inliers =self.pnp_solver.estimate_pose_pnp(obj_p,img_p)
    #     if inliers is None:
    #         return None,None,None,None
    #     projected_points, _ = cv2.projectPoints(object_points,rvec_new,tvec_new,intrinsic,distortion)
    #     projected_points = projected_points.reshape(-1, 2)
    #     matched_objects,matched_images,matched_land_idx,matched_image_idx=self.projection_match(image_points,projected_points,landmark_des, des, object_points)
        
    #     inliers = inliers.flatten()
    #     print(f"length of inliers {len(inliers)}")
    #     inlier_index=set()
    #     tracked_landmarks_ids=set()
    #     for i in  range(len(matched_objects)):
    #         landmarks[matched_land_idx[i]].active=True
    #         landmarks[matched_land_idx[i]].image_points=image_points[matched_image_idx[i]]
    #         landmarks[matched_land_idx[i]].tracked=landmarks[matched_land_idx[i]].tracked+1
    #         landmarks[matched_land_idx[i]].add_observation(id, image_points[matched_image_idx[i]],des[matched_image_idx[i]])
    #         inlier_index.add(matched_image_idx[i])
    #         tracked_landmarks_ids.add(landmarks[matched_land_idx[i]].id)
    #     return rvec_new, tvec_new, inlier_index, tracked_landmarks_ids

    #     # if inliers is None:
    #     #     return None,None,None
    #     # inliers = inliers.flatten()
    #     # print(f"length of inliers {len(inliers)}")
    #     # inlier_index=set()
    #     # for i in  inliers:
    #     #     landmarks[matched_land_idx[i]].active=True
    #     #     landmarks[matched_land_idx[i]].image_points=image_points[matched_image_idx[i]]
    #     #     inlier_index.add(matched_image_idx[i])
    #     # return rvec_new, tvec_new, inlier_index
        



