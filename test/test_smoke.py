import sys
import os
import numpy as np

# Add the folder containing ORBExtractor.so to path
feature_path = "/home/srikanth/stereo_vision/build/src/algorithms/feature_extraction"
atlas_path= "/home/srikanth/stereo_vision/build/src/algorithms/maper"

if feature_path not in sys.path:
    sys.path.insert(0, feature_path)
if atlas_path not in sys.path:
    sys.path.insert(0, atlas_path)

from FeatureExtractor import ORBExtractor,FeatureExtractor
from Atlas import Frame, Landmark


def test_object_creation():
    # Create an ORBExtractor object
    orb = ORBExtractor()
    assert isinstance(orb, FeatureExtractor)
    

def test_frame_creation():
    orb = ORBExtractor()
    dummy_image = np.zeros((480, 640), dtype=np.uint8)
    time_stamp=1
    intrinsic=np.zeros((3,3),dtype=np.float32, order='F')
    extrinsic=np.zeros((4,4),dtype=np.float32, order='F')
    frame=Frame(1,dummy_image,time_stamp,intrinsic,extrinsic,orb)
    assert frame.id == 1, f"Expected id 1, got {frame.id}"
    assert np.array_equal(frame.image, dummy_image), "Image does not match"
    assert frame.timeStamp == time_stamp, f"Expected timestamp {time_stamp}, got {frame.timeStamp}"
    assert np.allclose(frame.intrinsic, intrinsic), "Intrinsic matrix does not match"
    assert np.allclose(frame.extrinsic, extrinsic), "Extrinsic matrix does not match"

def test_landmark_creation():
    orb = ORBExtractor()
    dummy_image = np.zeros((480, 640), dtype=np.uint8)
    time_stamp=1
    intrinsic=np.zeros((3,3),dtype=np.float32, order='F')
    extrinsic=np.zeros((4,4),dtype=np.float32, order='F')
    frame=Frame(1,dummy_image,time_stamp,intrinsic,extrinsic,orb)
    position= np.array([0,0,0],dtype=np.float32)
    frame.cameraCenter=np.array([0,0,0],dtype=np.float32)
    landmark=Landmark(1,position,frame,8)
    assert landmark.id==1, f"Expected id 1, got {landmark.id}"
    assert landmark.point3D==position, f"Expected 3d point {position} but got {landmark.point3D}"
    assert landmark.frame.id==frame.id, f"Expected frame id {frame.id} but got {landmark.frame.id}"
    assert landmark.featureId==8, f"Expected featureId 8 but got {landmark.featureId}" 
 

  

if __name__ == "__main__":
    test_object_creation()
    Frame_creation_test()