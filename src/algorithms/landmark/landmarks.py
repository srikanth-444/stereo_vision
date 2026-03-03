import numpy as np
import cv2

class Landmark:
    def __init__(self,id, position, image_points,descriptor,frame_id):
        self.position=position
        self.image_points=image_points
        self.descriptor=descriptor
        self.id=id
        self.frame_id=frame_id
        self.active=True

        self.observations=[{'frame_id':self.frame_id, 'image_point':self.image_points}]
        self.confidence=None
        self.all_descriptors=[descriptor]
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

        # 1. Distance Check: Is it too far or too close compared to the original?
        # Default: Must be between 0.5x and 2.0x of the original distance
        if current_dist > self.reference_dist * dist_ratio_thresh or \
           current_dist < self.reference_dist / dist_ratio_thresh:
            return False

        # 2. View Angle Check: Are we looking at the 'face' of the point?
        # Uses dot product to ensure we aren't viewing from a sharp side angle
        cos_theta = np.dot(unit_vec_to_cam, self.normal_vector)
        if cos_theta < np.cos(np.radians(angle_thresh_deg)):
            return False
        return True
        
    def add_observation(self,id, image_point,descriptor):  
        observation={'frame_id':id, 'image_point':image_point}
        self.observations.append(observation)
        self.all_descriptors.append(descriptor)
        self.update_descriptor()
        self.tracked=self.tracked+1

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
