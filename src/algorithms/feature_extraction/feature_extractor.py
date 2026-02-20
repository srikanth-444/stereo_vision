from abc import ABC, abstractmethod
import cv2
class FeatureExtractor(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def extract_features(self, frame):
        pass
def feature_extractor_factory(config_load):
    if config_load.get('type',{}) in ['orb','ORB']:
        from .orb import ORB
        nfeatures=config_load.get('nfeatures',2000)
        scaleFactor=config_load.get('scaleFactor',1.2)
        nlevels=config_load.get('nlevels',8)
        edgeThreshold=config_load.get('edgeThreshold',31)
        firstLevel=config_load.get('firstLevel',0)
        patchSize=config_load.get('patchSize',31)
        fastThreshold=config_load.get('fastThreshold',20)
        WTA_K=config_load.get('WTA_K',2)
        return ORB(nfeatures=nfeatures,scaleFactor=scaleFactor,nlevels=nlevels,edgeThreshold=edgeThreshold,firstLevel=firstLevel,WTA_K=WTA_K,scoreType=cv2.ORB_HARRIS_SCORE,patchSize=patchSize,fastThreshold=fastThreshold)
    raise ValueError(f"Unknown feature extractor type: {config_load.get('type')}")