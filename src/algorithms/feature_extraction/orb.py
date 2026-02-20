import cv2
import numpy as np
from .feature_extractor import FeatureExtractor
from sklearn.cluster import MiniBatchKMeans
class ORB(FeatureExtractor):
    def __init__(self,nfeatures=2000,scaleFactor=1.2,nlevels=8,edgeThreshold=31,firstLevel=0,WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE,patchSize=31,fastThreshold=20,anms=500):
        
        self.orb = cv2.ORB_create(nfeatures=2000,scaleFactor=1.2,nlevels=8,edgeThreshold=31,firstLevel=0,WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE,patchSize=31,fastThreshold=20)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.MIN_TRACKED = 50
        self.FB_MAX_DIST = 0.8
        self.LK_PARAMS = dict(winSize=(31,31), maxLevel=8,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.anms_num_keep=anms

    def extract_features(self,frame):
        frame = self.clahe.apply(frame)
        kp, des = self.orb.detectAndCompute(frame, None)
        # keep_idx=self.anms(kp)
        # kp = [kp[i] for i in keep_idx]
        # des = des[keep_idx]
        return kp, des
    
    
    def anms(self,keypoints):
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
        keep_idx = sorted_idx[:self.anms_num_keep]
        return keep_idx
        
    