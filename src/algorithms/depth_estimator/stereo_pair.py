import numpy as np
import cv2
from scipy.optimize import least_squares

class Stereo :
    def __init__(self, left_camera,right_camera):
        self.left_camera=left_camera
        self.right_camera=right_camera

        self.l_K=left_camera.get_intrinsic()
        self.r_K=right_camera.get_intrinsic()

        if self.left_camera.get_extrinsic() is None or self.right_camera.get_extrinsic() is None:
            raise ValueError("Both cameras must have extrinsic parameters defined.")
        if (self.left_camera.get_extrinsic()==self.right_camera.get_extrinsic()).all():
            raise ValueError("Extrinsic parameters of left and right cameras cannot be the same.")

        self.l_extrinsic=np.linalg.inv(self.left_camera.get_extrinsic())
        self.r_extrinsic=np.linalg.inv(self.right_camera.get_extrinsic())
        
        self.P= [
            self.l_K @ self.l_extrinsic[:3, :],
            self.r_K @ self.r_extrinsic[:3, :]
        ]
        np.seterr(divide='ignore', invalid='ignore')
    



    def triangulate_points(self, pts1, pts2):
        """
        Triangulate 3D points from matched 2D points in stereo images.
        pts1, pts2: Nx2 arrays of matched points in left and right images.
        Returns Nx3 array of 3D points.
        """
        P1 = self.P[0]
        P2 = self.P[1]
        pts3d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T).T
        pts3d = pts3d_hom[:, :3] / pts3d_hom[:, 3][:, np.newaxis]
        return pts3d
    
    

    def nonlinear_triangulation(self,pts_2d, X0):
        """
        pts_2d: list of 2D points in each camera
        P: list of 3x4 camera projection matrices
        X0: initial 3D point (from linear triangulation)
        """
        X0=X0.flatten()
        
        res = least_squares(reprojection_residual, X0, args=(pts_2d, self.P),max_nfev=100,ftol=1e-6, xtol=1e-6, gtol=1e-6)

        residuals = reprojection_residual(res.x, pts_2d, self.P)
        # print(residuals.shape)
        X0_reshaped = res.x.reshape(-1, 3)
        return X0_reshaped, residuals
    
    def get_depth(self,pts_2d,Non_linear=False):
        """
        pts_2d : list of shape [2N,2] 
        """
        #split pts_2d to [N,2],[N,2]
       
        pts1=pts_2d[0]
        pts2=pts_2d[1]
        pts3d= self.triangulate_points(pts1,pts2)
        if Non_linear==True:
            self.nonlinear_triangulation(pts_2d,pts3d)
         # Homogeneous coordinates (N,4)
        ones = np.ones((pts3d.shape[0], 1))
        pts3d_hom = np.hstack((pts3d, ones))

        P1 = self.P[0]  # (3,4)
        P2 = self.P[1]

        # Project all points at once
        proj_left = (P1 @ pts3d_hom.T).T    # (N,3)
        proj_right = (P2 @ pts3d_hom.T).T

        proj_left = proj_left[:, :2] / proj_left[:, 2:3]
        proj_right = proj_right[:, :2] / proj_right[:, 2:3]

        # Vectorized reprojection error
        error_left = np.linalg.norm(proj_left - pts1, axis=1)
        error_right = np.linalg.norm(proj_right - pts2, axis=1)

        reprojection_errors = (error_left + error_right) * 0.5

        return pts3d, reprojection_errors
def reprojection_residual(X_flat, pts_2d_list, P_list):
        """
        X_flat: flattened 3D points, shape (num_points*3,)
        pts_2d_list: list of per-camera 2D points, each shape (num_points, 2)
        P_list: list of 3x4 projection matrices, length = num_cameras
        """
        num_points = X_flat.shape[0] // 3
        X = X_flat.reshape((num_points, 3))  # (num_points, 3)
        
        residuals = []

        for i in range(num_points):
            X_h = np.hstack([X[i], 1.0])
            reproj_sum = np.zeros(2)
            
            for cam_idx, P in enumerate(P_list):
                proj = P @ X_h
                proj_2d = proj[:2] / proj[2]
                reproj_sum += (proj_2d - pts_2d_list[cam_idx][i])
            
            reproj_avg = reproj_sum / len(P_list)
            
            # combine dx, dy into a single scalar
            residual_scalar = np.linalg.norm(reproj_avg)
            residuals.append(residual_scalar)
        
        return np.array(residuals)  # 1D array of length = num_points * num_cameras * 2

