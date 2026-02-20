import numpy as np
import cv2
from scipy.optimize import least_squares
# import matplotlib.pyplot as plt

class Camera:
    def __init__(self, intrinsic, distortion, extrinsic, interface):
        fx= intrinsic.get('fx')
        fy= intrinsic.get('fy')
        cx= intrinsic.get('cx')
        cy= intrinsic.get('cy')

        T=np.array(extrinsic).reshape(4,4)
        
        distortion=np.array(distortion.get('distortion_coefficients',[]),dtype=np.float32)
        self.intrinsic=np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
        self.distortion=distortion
        self.extrinsic=T
        self.interface=interface
        

    def get_intrinsic(self,):
        return self.intrinsic
    def get_extrinsic(self,):
        return self.extrinsic
    def set_extrinsic(self, extrinsic: np.ndarray):
        """
        Set camera extrinsic matrix (world -> camera transform).

        Args:
            extrinsic: 4x4 rigid transform matrix
        """
        if not isinstance(extrinsic, np.ndarray):
            raise TypeError("extrinsic must be a NumPy array")

        if extrinsic.shape != (4, 4):
            raise ValueError("extrinsic must have shape (4, 4)")

        # Check last row is [0, 0, 0, 1]
        if not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("extrinsic must be a rigid transform (last row must be [0,0,0,1])")

        self.extrinsic = extrinsic
        
    def compute_projection_matrix(self,):
        return self.intrinsic @ self.extrinsic
    def update_projection_matrix(self,):
        self.projection_matrix=self.compute_projection_matrix()
    
    def project_points(self, points_3d):
        """
        Reproject 3D points into 2D image coordinates.

        Args:
            points_3d: Nx3 array of 3D points
            projection: 3x4 camera projection matrix

        Returns:
            Nx2 array of 2D image points
        """
        if not isinstance(points_3d, np.ndarray):
            raise TypeError("points_3d must be a NumPy array")

        if points_3d.ndim != 2 or points_3d.shape[1] != 3:
            raise ValueError("points_3d must have shape (N, 3)")

        if self.projection_matrix.shape != (3, 4):
            raise ValueError("projection matrix must be 3x4")
        # Convert to homogeneous coordinates (N x 4)
        ones = np.ones((points_3d.shape[0], 1))
        points_h = np.hstack([points_3d, ones])
        points_2d = (self.projection_matrix @ points_h.T).T
        points_2d[:,0] /=points_2d[:,2]
        points_2d[:,1] /=points_2d[:,2]
        return points_2d[:,:2]
    
    def world_to_camera(self,points_3d):

        if points_3d.ndim != 2 or points_3d.shape[1] != 3:
            raise ValueError("points_3d must have shape (N, 3)")
        ones = np.ones((points_3d.shape[0], 1))
        points_h = np.hstack([points_3d, ones])
        
        transformed_points= (self.extrinsic@points_h.T).T

        return transformed_points[:,:3]
    
    def camera_to_world(self, points_3d):
        if points_3d.ndim != 2 or points_3d.shape[1] != 3:
            raise ValueError("points_3d must have shape (N, 3)")
        ones = np.ones((points_3d.shape[0], 1))
        points_h = np.hstack([points_3d, ones])

        transformed_points=(np.linalg.inv(self.extrinsic)@points_h.T).T

        return transformed_points[:,:3]
    
    def get_frame(self,):
        frame,timestamp = self.interface.read_frame()
        frame=self.undistortframe(frame)
        return frame, timestamp
    
    def undistortframe(self, frame):
        if self.distortion is None or len(self.distortion) == 0:
            return frame
        h, w = frame.shape[:2]
        new_intrinsic, _ = cv2.getOptimalNewCameraMatrix(self.intrinsic, self.distortion, (w,h), 1)
        undistorted_frame = cv2.undistort(frame, self.intrinsic, self.distortion, None, new_intrinsic)
        return undistorted_frame

        
        
# class stereo_camera :
#     def __init__(self, left_camera,right_camera):
#         self.left_camera=left_camera
#         self.right_camera=right_camera

#         self.l_K=left_camera.get_intrinsic()
#         self.r_K=right_camera.get_intrinsic()

#         T = self.right_camera.get_extrinsic()@np.linalg.inv(self.left_camera.get_extrinsic())
#         self.R=T[:3,:3]
#         self.t=T[:3, 3]
#         np.seterr(divide='ignore', invalid='ignore')

#     def triangulate_points(self, pts1, pts2, R=None, t=None):
#         """
#         Triangulate 3D points from matched 2D points in stereo images.
#         pts1, pts2: Nx2 arrays of matched points in left and right images.
#         Returns Nx3 array of 3D points.
#         """
#         if R==None:
#             R=self.R
#         if t==None:
#             t=self.t
        
#         P1 = self.l_K @ np.hstack((np.eye(3), np.zeros((3, 1))))
#         P2 = self.r_K @ np.hstack((R, t.reshape(3, 1)))
#         pts3d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T).T
#         pts3d = pts3d_hom[:, :3] / pts3d_hom[:, 3][:, np.newaxis]
#         for pts in pts3d:
#             if pts[2]<0:
#                 pts[2]=0
#             if pts[2]>10:
#                 pts[2]=10

#         reprojection_errors = []
#         for i in range(len(pts3d)):
#             point_3d_hom = np.hstack((pts3d[i], 1))

#             proj_left = P1 @ point_3d_hom
#             proj_left /= proj_left[2]
#             error_left = np.linalg.norm(proj_left[:2] - pts1[i])

#             proj_right = P2 @ point_3d_hom
#             proj_right /= proj_right[2]
#             error_right = np.linalg.norm(proj_right[:2] - pts2[i])

#             reprojection_errors.append((error_left + error_right) / 2)

#         return pts3d, reprojection_errors
    
    

#     def nonlinear_triangulation(self,pts_2d, X0):
#         """
#         pts_2d: list of 2D points in each camera
#         P: list of 3x4 camera projection matrices
#         X0: initial 3D point (from linear triangulation)
#         """
#         X0=X0.flatten()
#         P= [
#             self.l_K @ np.hstack((np.eye(3), np.zeros((3, 1)))),
#             self.r_K @ np.hstack((self.R, self.t.reshape(3, 1)))
#         ]
#         res = least_squares(reprojection_residual, X0, args=(pts_2d, P),max_nfev=100,ftol=1e-6, xtol=1e-6, gtol=1e-6)

#         residuals = reprojection_residual(res.x, pts_2d, P)
#         # print(residuals.shape)
#         X0_reshaped = res.x.reshape(-1, 3)
#         return X0_reshaped, residuals

#     def plot_camera(self,x, z, yaw, size=0.1, color='r'):
#         # Triangle points in camera frame (forward along Z)
#         tri = np.array([[0, -20], [-10, 10], [10, 10]]) * size
#         # Rotate triangle according to yaw
#         R = np.array([[np.cos(yaw), -np.sin(yaw)],
#                     [np.sin(yaw),  np.cos(yaw)]])
#         tri_rot = tri @ R.T
#         # Translate to camera position
#         tri_rot[:,0] += x
#         tri_rot[:,1] += z
#         plt.fill(tri_rot[:,0], tri_rot[:,1], color=color)
    
#     def skew(self,t):
#         """
#         t: 3x1 vector
#         returns: 3x3 skew symmetric matrix
#         """
#         t = t.flatten()
#         return np.array([
#             [0, -t[2], t[1]],
#             [t[2], 0, -t[0]],
#             [-t[1], t[0], 0]
#         ])

    

# def reprojection_residual(X_flat, pts_2d_list, P_list):
#         """
#         X_flat: flattened 3D points, shape (num_points*3,)
#         pts_2d_list: list of per-camera 2D points, each shape (num_points, 2)
#         P_list: list of 3x4 projection matrices, length = num_cameras
#         """
#         num_points = X_flat.shape[0] // 3
#         X = X_flat.reshape((num_points, 3))  # (num_points, 3)
        
#         residuals = []

#         for i in range(num_points):
#             X_h = np.hstack([X[i], 1.0])
#             reproj_sum = np.zeros(2)
            
#             for cam_idx, P in enumerate(P_list):
#                 proj = P @ X_h
#                 proj_2d = proj[:2] / proj[2]
#                 reproj_sum += (proj_2d - pts_2d_list[cam_idx][i])
            
#             reproj_avg = reproj_sum / len(P_list)
            
#             # combine dx, dy into a single scalar
#             residual_scalar = np.linalg.norm(reproj_avg)
#             residuals.append(residual_scalar)
        
#         return np.array(residuals)  # 1D array of length = num_points * num_cameras * 2

# def reprojection_ba_residual(X_flat, pts_2d_list):
#     """
#     X_flat: flattened 3D points, shape (num_points*3,)
#     pts_2d_list: list of per-camera 2D points, each shape (num_points, 2)
#     P_list: list of 3x4 projection matrices, length = num_cameras
#     """
#     num_points = X_flat.shape[0] // 9
#     X = X_flat.reshape((num_points, 9))  # (num_points, 3)
#     X_points = X[:, :3]
#     R_vecs = X[:, 3:6]
#     t_vecs = X[:, 6:9]
#     residuals = []
#     for i in range(num_points):
#         X_h = np.hstack([X_points[i], 1.0])
#         reproj_sum = np.zeros(2)
        
#         R, _ = cv2.Rodrigues(R_vecs[i])
#         t = t_vecs[i].reshape(3, 1)
#         P = np.hstack((R, t))
        
#         proj = P @ X_h
#         proj_2d = proj[:2] / proj[2]
#         reproj_sum += (proj_2d - pts_2d_list[i])
#         # print(reproj_sum)
#         reproj_avg = reproj_sum
        
#         # # combine dx, dy into a single scalar
#         # residual_scalar = np.linalg.norm(reproj_avg)
#         residuals.append(reproj_avg)

# def anms(keypoints, num_keep=500):
#     # sort by response
#     keypoints = sorted(keypoints, key=lambda x: -x.response)
#     radii = [float('inf')] * len(keypoints)

#     for i in range(len(keypoints)):
#         xi, yi = keypoints[i].pt
#         for j in range(i):
#             xj, yj = keypoints[j].pt
#             dist = (xi - xj)**2 + (yi - yj)**2
#             if keypoints[j].response > keypoints[i].response:
#                 if dist < radii[i]:
#                     radii[i] = dist

#     # sort by suppression radius
#     sorted_idx = sorted(range(len(keypoints)), key=lambda i: -radii[i])
#     keep_idx = sorted_idx[:num_keep]
#     return keep_idx