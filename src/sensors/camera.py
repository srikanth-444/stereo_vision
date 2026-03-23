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

        T=np.array(extrinsic,dtype=np.float32).reshape(4,4)
        distortion=np.array(distortion.get('distortion_coefficients',[]),dtype=np.float32)
        self.intrinsic=np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]],dtype=np.float32)
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
        if not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("extrinsic must be a rigid transform (last row must be [0,0,0,1])")

        self.extrinsic = extrinsic
        

    def get_frame(self,):
        frame,timestamp = self.interface.read_frame()
        # frame=self.undistortframe(frame)
        return frame, timestamp
    
    def undistortframe(self, frame):
        if self.distortion is None or len(self.distortion) == 0:
            return frame
        undistorted_frame = cv2.undistort(frame, self.intrinsic, self.distortion, None, self.new_intrinsic)
        return undistorted_frame
