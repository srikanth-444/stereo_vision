import cv2
import numpy as np
import sys

# Add the folder containing the .so, not the file itself
sys.path.insert(0, "/home/srikanth/stereo_vision/src/algorithms/feature_extraction/ORBExtractor")

# Now import the module by its name (the name used in PYBIND11_MODULE)
from ORBExtractor import ORBExtractor

class ORB():
    def __init__(self,nfeatures=1000,scaleFactor=1.2,nlevels=8,fastInitial=20, fastFinal=7):
        
        self.nfeatures = nfeatures
        self.orb= ORBExtractor(nfeatures,scaleFactor,nlevels,fastInitial,fastFinal)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract_features(self, frame):
        kps, des = self.orb(frame)
        return kps, des
    
       