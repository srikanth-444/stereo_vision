import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial import KDTree

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
        self.l_dist=self.left_camera.distortion
        self.r_dist=self.right_camera.distortion
        
        self.P= [
            self.l_K @ self.l_extrinsic[:3, :],
            self.r_K @ self.r_extrinsic[:3, :]
        ]
        self.height, self.width = 480, 752 
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.debug_img = np.zeros((self.height,self.width, 3), dtype=np.uint8)

         # 1. Calculate Relative Transform for Rectification
        # R_rel = R_r * R_l.T | T_rel = R_r * (t_l - t_r)
        R_l, t_l = self.l_extrinsic[:3, :3], self.l_extrinsic[:3, 3]
        R_r, t_r = self.r_extrinsic[:3, :3], self.r_extrinsic[:3, 3]
        
        R_rel = R_r @ R_l.T
        T_rel = t_r - R_rel @ t_l

        # 2. Compute Rectification Matrices
        self.R1, self.R2, self.P1, self.P2, _, _, _ = cv2.stereoRectify(
            self.l_K, self.l_dist, self.r_K, self.r_dist, 
            (self.width, self.height), R_rel, T_rel, 
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        np.seterr(divide='ignore', invalid='ignore')
    
    def rectify_pts(self, pts, camera='left'):
        """Helper to rectify keypoints without changing the image."""
        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        if camera == 'left':
            rect = cv2.undistortPoints(pts, self.l_K, self.l_dist, R=self.R1, P=self.P1)
        else:
            rect = cv2.undistortPoints(pts, self.r_K, self.r_dist, R=self.R2, P=self.P2)
        return rect.reshape(-1, 2)

    def stereo_match(self, kps_l, kps_r, des_l, des_r, epipolar_threshold=3.0):
        """
        1. Rectifies keypoints
        2. Matches descriptors
        3. Filters matches based on Y-alignment (Epipolar Constraint)
        """
        # Get raw coordinates
        pts_l_raw = np.array([kp.pt for kp in kps_l])
        pts_r_raw = np.array([kp.pt for kp in kps_r])

        # Step 1: Rectify keypoints
        rect_l = self.rectify_pts(pts_l_raw, 'left')
        rect_r = self.rectify_pts(pts_r_raw, 'right')
        # self.plot_points(rect_l,(0,0,255))
        # self.plot_points(rect_r,(255,0,0))
        
        # cv2.imshow("stereo Debug", self.debug_img)
        # cv2.waitKey(0)
        row_dict={}
        final_matches=[]
        unique_index=set()
        for i,pt in enumerate(rect_r):
            round_y=max(0,min(round(pt[1]),self.height))
            try:
                row_dict[round_y].add(i)
            except KeyError:
                row_dict[round_y]=set()
                row_dict[round_y].add(i)
        for i,pt in enumerate(rect_l):
            round_y=max(0,min(round(pt[1]),self.height))
            candidate_indices=[]
            for y in range(round_y-2,round_y+3):
                if y in row_dict:
                    candidate_indices.extend(list(row_dict[y]))
            if len(candidate_indices)==0:
                continue
            candiate_des=des_r[candidate_indices]
            matches=self.bf.knnMatch(des_l[i].reshape(1, -1), candiate_des, k=2)
            if len(matches) == 0 or len(matches[0]) < 2:
                continue
            m, n = matches[0]
            # print(f"matches ratio {m.distance/n.distance} m.distance{m.distance}")
            if m.distance < 0.8 * n.distance and m.distance < 40:
                global_train_idx = candidate_indices[m.trainIdx]

                if global_train_idx not in unique_index:
                    unique_index.add(global_train_idx)
                    m.trainIdx=candidate_indices[m.trainIdx]
                    m.queryIdx=i
                    final_matches.append(m) 
        # print(f"stereo matches {len(final_matches)}")
        return final_matches
    
    # def stereo_match(self,des_l,des_r):
    #     # Ensure numpy arrays
    #     des_l = np.array(des_l, dtype=np.uint8)
    #     des_r = np.array(des_r, dtype=np.uint8)

    #     matches_knn = self.bf.knnMatch(des_l, des_r, k=2)

    #     final_matches = []
    #     for knn in matches_knn:
    #         if len(knn) < 2:
    #             continue
    #         m, n = knn
    #         if m.distance < 0.7 * n.distance and m.distance < 50:
    #             final_matches.append(m)

    #     return final_matches
        
    def plot_points(self,points,color):
        # Draw projected landmarks in red
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < self.width and 0 <= y <self.height:
                cv2.circle(self.debug_img, (x, y), 3, color, -1)

    def triangulate_points(self, pts1, pts2):
        """
        Triangulate 3D points from matched 2D points in stereo images.
        pts1, pts2: Nx2 arrays of matched points in left and right images.
        Returns Nx3 array of 3D points.
        """
        # pts1=self.rectify_pts(pts1,'left')
        # pts2=self.rectify_pts(pts2,'right')
        # self.plot_points(pts1,(0,0,255))
        # self.plot_points(pts2,(255,0,0))
        # for i in range(len(pts1)):
        #     pt1 = tuple(map(int, pts1[i]))
        #     pt2 = tuple(map(int, pts2[i]))
        #     cv2.line(self.debug_img, pt1, pt2, (0, 255, 0), 1)
       
        # cv2.imshow("stereo Debug", self.debug_img)
        # cv2.waitKey(0)
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

        for pt in pts3d:
            if pt[2]>=20:
                pt[2]=20
            if pt[2]<=0:
                pt[2]=0
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

