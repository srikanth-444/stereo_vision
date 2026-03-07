import numpy as np
import cv2
from scipy.optimize import least_squares
# import matplotlib.pyplot as plt

class Camera:
    def __init__(self, intrinsic, distortion, extrinsic, interface,w,h,feature_extractor):
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
        self.new_intrinsic, _ = cv2.getOptimalNewCameraMatrix(self.intrinsic, self.distortion, (w,h), 1)
        self.feature_extractor=feature_extractor
        

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
        undistorted_frame = cv2.undistort(frame, self.intrinsic, self.distortion, None, self.new_intrinsic)
        return undistorted_frame
