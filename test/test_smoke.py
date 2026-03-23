import sys
import os
import numpy as np

# Add the folder containing ORBExtractor.so to path
feature_path = "/home/srikanth/stereo_vision/src/feature_extractor"
atlas_path= "/home/srikanth/stereo_vision/src/atlas"

if feature_path not in sys.path:
    sys.path.insert(0, feature_path)
if atlas_path not in sys.path:
    sys.path.insert(0, atlas_path)

from FeatureExtractor import ORBExtractor,FeatureExtractor
from Atlas import Frame, Landmark

def create_checkerboard(h, w, square_size=64):
    # Create a base pattern
    base = np.indices((h // square_size, w // square_size)).sum(axis=0) % 2
    # Scale it up to the full image size
    img = (base.repeat(square_size, axis=0).repeat(square_size, axis=1) * 255).astype(np.uint8)
    return img


def test_object_creation():
    # Create an ORBExtractor object
    orb = ORBExtractor()
    assert isinstance(orb, FeatureExtractor)
    

def test_frame_creation():
    orb = ORBExtractor()
    dummy_image = create_checkerboard(480, 640, square_size=64)
    time_stamp=np.int64(1.4036365797635556e+18)
    intrinsic=np.array([[280, 0, 360],[0, 280, 360],[0, 0, 1]],dtype=np.float32)
    extrinsic=np.zeros((4,4),dtype=np.float32, order='F')
    frame=Frame(1,dummy_image,time_stamp,intrinsic,extrinsic,orb)
    assert frame.id == 1, f"Expected id 1, got {frame.id}"
    assert np.array_equal(frame.image, dummy_image), "Image does not match"
    assert frame.timeStamp == time_stamp, f"Expected timestamp {time_stamp}, got {frame.timeStamp}"
    assert np.allclose(frame.intrinsic, intrinsic), "Intrinsic matrix does not match"
    assert np.allclose(frame.extrinsic, extrinsic), "Extrinsic matrix does not match"

def test_landmark_creation():
    orb = ORBExtractor()
    dummy_image = create_checkerboard(480, 640, square_size=64)
    time_stamp=1
    intrinsic=np.zeros((3,3),dtype=np.float32, order='F')
    extrinsic=np.zeros((4,4),dtype=np.float32, order='F')
    frame=Frame(1,dummy_image,time_stamp,intrinsic,extrinsic,orb)
    frame.extractFeatures()
    position= np.array([0.0,0.0,0.0],dtype=np.float32, order='F')
    frame.cameraCenter= np.array([0.0,0.0,0.0],dtype=np.float32, order='F')
    landmark=Landmark(1,position,frame,8)
    assert landmark.id==1, f"Expected id 1, got {landmark.id}"
    assert np.allclose(landmark.point3D,position), f"Expected 3d point {position} but got {landmark.point3D}"

 

  

if __name__ == "__main__":
    test_object_creation()
    test_frame_creation()
    test_landmark_creation()