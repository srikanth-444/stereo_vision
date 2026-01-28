import numpy as np
import cv2
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


class stereo_camera:
    def __init__(self, l_K,r_K, l_dist, r_dist,r,t):
        self.l_K = np.load(l_K)
        self.r_K = np.load(r_K)
        self.l_dist = np.load(l_dist)
        self.r_dist = np.load(r_dist)
        self.R = np.load(r)
        self.t = np.load(t)/1000.0
        np.seterr(divide='ignore', invalid='ignore')

    def triangulate_points(self, pts1, pts2, R=None, t=None):
        """
        Triangulate 3D points from matched 2D points in stereo images.
        pts1, pts2: Nx2 arrays of matched points in left and right images.
        Returns Nx3 array of 3D points.
        """
        if R==None:
            R=self.R
        if t==None:
            t=self.t
        
        P1 = self.l_K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.r_K @ np.hstack((R, t.reshape(3, 1)))
        pts3d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T).T
        pts3d = pts3d_hom[:, :3] / pts3d_hom[:, 3][:, np.newaxis]
        for pts in pts3d:
            if pts[2]<0:
                pts[2]=0
            if pts[2]>10:
                pts[2]=10

        reprojection_errors = []
        for i in range(len(pts3d)):
            point_3d_hom = np.hstack((pts3d[i], 1))

            proj_left = P1 @ point_3d_hom
            proj_left /= proj_left[2]
            error_left = np.linalg.norm(proj_left[:2] - pts1[i])

            proj_right = P2 @ point_3d_hom
            proj_right /= proj_right[2]
            error_right = np.linalg.norm(proj_right[:2] - pts2[i])

            reprojection_errors.append((error_left + error_right) / 2)

        return pts3d, reprojection_errors
    
    

    def nonlinear_triangulation(self,pts_2d, X0):
        """
        pts_2d: list of 2D points in each camera
        P: list of 3x4 camera projection matrices
        X0: initial 3D point (from linear triangulation)
        """
        X0=X0.flatten()
        P= [
            self.l_K @ np.hstack((np.eye(3), np.zeros((3, 1)))),
            self.r_K @ np.hstack((self.R, self.t.reshape(3, 1)))
        ]
        res = least_squares(reprojection_residual, X0, args=(pts_2d, P),max_nfev=100,ftol=1e-6, xtol=1e-6, gtol=1e-6)

        residuals = reprojection_residual(res.x, pts_2d, P)
        # print(residuals.shape)
        X0_reshaped = res.x.reshape(-1, 3)
        return X0_reshaped, residuals

    def plot_camera(self,x, z, yaw, size=0.1, color='r'):
        # Triangle points in camera frame (forward along Z)
        tri = np.array([[0, -20], [-10, 10], [10, 10]]) * size
        # Rotate triangle according to yaw
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
        tri_rot = tri @ R.T
        # Translate to camera position
        tri_rot[:,0] += x
        tri_rot[:,1] += z
        plt.fill(tri_rot[:,0], tri_rot[:,1], color=color)
    
    def skew(self,t):
        """
        t: 3x1 vector
        returns: 3x3 skew symmetric matrix
        """
        t = t.flatten()
        return np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ])
    def bundle_adjustment(self, landmarks_list):
        mark_list=[]
        im_points_list=[]
        for mark in landmarks_list:
            for obs in mark.observations:
                m=np.hstack([mark.position.reshape(3,), obs['rotation'], obs['translation']])
                mark_list.append(m)
                im_points_list.append(obs['image_point'])
        mark_list=np.array(mark_list)
        im_points_list=np.array(im_points_list) 
        res = least_squares(reprojection_ba_residual, np.array(mark_list).flatten(), args=(im_points_list,))
        X_reshaped = res.x.reshape(-1, 9)
        for i, mark in enumerate(landmarks_list):
            mark.position=X_reshaped[i,:3]
    

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

def reprojection_ba_residual(X_flat, pts_2d_list):
    """
    X_flat: flattened 3D points, shape (num_points*3,)
    pts_2d_list: list of per-camera 2D points, each shape (num_points, 2)
    P_list: list of 3x4 projection matrices, length = num_cameras
    """
    num_points = X_flat.shape[0] // 9
    X = X_flat.reshape((num_points, 9))  # (num_points, 3)
    X_points = X[:, :3]
    R_vecs = X[:, 3:6]
    t_vecs = X[:, 6:9]
    residuals = []
    for i in range(num_points):
        X_h = np.hstack([X_points[i], 1.0])
        reproj_sum = np.zeros(2)
        
        R, _ = cv2.Rodrigues(R_vecs[i])
        t = t_vecs[i].reshape(3, 1)
        P = np.hstack((R, t))
        
        proj = P @ X_h
        proj_2d = proj[:2] / proj[2]
        reproj_sum += (proj_2d - pts_2d_list[i])
        # print(reproj_sum)
        reproj_avg = reproj_sum
        
        # # combine dx, dy into a single scalar
        # residual_scalar = np.linalg.norm(reproj_avg)
        residuals.append(reproj_avg)

def anms(keypoints, num_keep=500):
    # sort by response
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    radii = [float('inf')] * len(keypoints)

    for i in range(len(keypoints)):
        xi, yi = keypoints[i].pt
        for j in range(i):
            xj, yj = keypoints[j].pt
            dist = (xi - xj)**2 + (yi - yj)**2
            if keypoints[j].response > keypoints[i].response:
                if dist < radii[i]:
                    radii[i] = dist

    # sort by suppression radius
    sorted_idx = sorted(range(len(keypoints)), key=lambda i: -radii[i])
    keep_idx = sorted_idx[:num_keep]
    return keep_idx