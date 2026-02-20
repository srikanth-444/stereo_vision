import numpy as np
import cv2
class PNPSolver:
    def __init__(self,reprojection_error,confidence,iterationsCount,camera):
        self.reprojection_error=reprojection_error
        self.confidence=confidence
        self.iterationsCount=iterationsCount
        self.camera=camera

    def estimate_pose_pnp(self,object_points,image_points):
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        k=self.camera.intrinsic
        dist=self.camera.distortion
        # print(self.reprojection_error)
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, k,dist, flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=self.reprojection_error, confidence=self.confidence, iterationsCount=self.iterationsCount)

        if retval:
            reproj = cv2.projectPoints(object_points, rvec, tvec, k,dist)[0].squeeze()
            reproj_err = np.linalg.norm(reproj - image_points, axis=1)
        return rvec, tvec, inliers
                    