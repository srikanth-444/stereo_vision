import numpy as np
import cv2
class PNPSolver:
    def __init__(self,reprojection_error,confidence,iterationsCount,camera):
        self.reprojection_error=reprojection_error
        self.confidence=confidence
        self.iterationsCount=iterationsCount
        self.camera=camera

    def estimate_pose_pnp(self,object_points,image_points,intrinsic):
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        k=intrinsic
        dist=np.zeros(4)
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, k,dist, flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=self.reprojection_error, confidence=self.confidence, iterationsCount=self.iterationsCount)

        if retval:
            reproj = cv2.projectPoints(object_points, rvec, tvec, k,dist)[0].squeeze()
            reproj_err = np.linalg.norm(reproj - image_points, axis=1)
        return rvec, tvec, inliers
                    
    def estimate_pose_pnp_3d(self, Q, P):
        """
        P, Q: (N,3) matched 3D points
        Returns: R (3x3), t (3,)
        """
        assert P.shape == Q.shape

        # 1. centroids
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)

        # 2. center
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q

        # 3. covariance
        H = P_centered.T @ Q_centered

        # 4. SVD
        U, S, Vt = np.linalg.svd(H)

        # 5. rotation
        R = Vt.T @ U.T

        # reflection correction
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        # 6. translation
        t = centroid_Q - R @ centroid_P

        return R, t