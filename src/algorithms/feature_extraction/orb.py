import cv2
import numpy as np
from .feature_extractor import FeatureExtractor
from .Orb_slam_extractor.build.orb_slam_features import ORBExtractor

class ORB(FeatureExtractor):
    def __init__(self,nfeatures=1000,scaleFactor=1.2,nlevels=8,edgeThreshold=31,firstLevel=0,WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE,patchSize=31,fastThreshold=20,anms=500):
        
        self.nfeatures = nfeatures
        self.orb= ORBExtractor(nfeatures,scaleFactor,nlevels,fastThreshold,7)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract_features(self, frame):
        kps, des = self.orb(frame)
        return kps, des
    
       