import numpy as np
import cv2

class Landmark:
    def __init__(self,id, position,frame, feature_id):
        self.position=position
        self.descriptor=frame.descriptors[feature_id]
        self.id=id
        self.frame=frame
        self.active=True
        self.observations={}
        self.observations[self.frame]=feature_id
        self.confidence=None
        self.all_descriptors=[self.descriptor]
        self.normal_vector = None
        self.nvisible=1
        self.tracked=1
        self.is_bad=False

    def set_landmark_as_bad(self):
        self.is_bad=True

    def set_normal(self,camera_center):
        normal_vector=self.position-camera_center
        self.normal_vector=normal_vector/np.linalg.norm(normal_vector)

    def set_reference_depth(self,camera_center):
        self.ref_depth=np.linalg.norm(self.position-camera_center)
    
    def is_landmark_visible(self,camera_center,dist_ratio_thresh=2.0, angle_thresh_deg=60):
        vec_to_cam = camera_center - self.position
        current_dist = np.linalg.norm(vec_to_cam)
        unit_vec_to_cam = vec_to_cam / current_dist

        if current_dist > self.reference_dist * dist_ratio_thresh or \
           current_dist < self.reference_dist / dist_ratio_thresh:
            return False

        cos_theta = np.dot(unit_vec_to_cam, self.normal_vector)
        if cos_theta < np.cos(np.radians(angle_thresh_deg)):
            return False
        return True
        
    def add_observation(self,frame, feature_id): 
        self.tracked=self.tracked+1 
        self.observations[frame]=feature_id
        self.all_descriptors.append(frame.descriptors[feature_id])
        if len(self.observations)>5:
            self.update_descriptor()
    
    def strip_landmark(self,):
        self.descriptor = None
        self.all_descriptors = []
        # self.position = None
        self.active=False

    def update_descriptor(self):
        if len(self.observations) < 2:
            return
        all_descs = np.array(self.all_descriptors,dtype=np.uint8)
        dist_matrix = cv2.batchDistance(
                                            all_descs,
                                            all_descs,
                                            dtype=cv2.CV_32S,
                                            normType=cv2.NORM_HAMMING,
                                            K=all_descs.shape[0]
                                        )[0]
        median_distances = np.median(dist_matrix, axis=1)
        best_idx = np.argmin(median_distances)
        self.descriptor = all_descs[best_idx]

    def increase_visible(self):
        self.nvisible += 1

    def get_found_ratio(self):
        return self.tracked / self.nvisible if self.nvisible > 0 else 0
    
    

