import sys
import os
import numpy as np

# from src.sensors import camera

# Add the folder containing ORBExtractor.so to path
feature_path = "/home/srikanth/stereo_vision/src/feature_extractor"
atlas_path= "/home/srikanth/stereo_vision/src/atlas"
interface_path="/home/srikanth/stereo_vision/src/interfaces"
camera_path="/home/srikanth/stereo_vision/src/sensors"
if feature_path not in sys.path:
    sys.path.insert(0, feature_path)
if atlas_path not in sys.path:
    sys.path.insert(0, atlas_path)
if interface_path not in sys.path:
    sys.path.insert(0, interface_path)
if camera_path not in sys.path:
    sys.path.insert(0, camera_path)

from FeatureExtractor import ORBExtractor,FeatureExtractor
from Atlas import Frame, Landmark
from Interface import CameraCSVSource,CameraInterface
from Sensors import Camera

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
    dist_coeff=np.array([0,0,0,0],dtype=np.float32)
    frame=Frame(1,dummy_image,time_stamp,intrinsic,extrinsic,dist_coeff,orb)
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
    dist_coeff=np.array([0,0,0,0],dtype=np.float32)
    frame=Frame(1,dummy_image,time_stamp,intrinsic,extrinsic,dist_coeff,orb)
    frame.extractFeatures()
    position= np.array([0.0,0.0,0.0],dtype=np.float32, order='F')
    frame.cameraCenter= np.array([0.0,0.0,0.0],dtype=np.float32, order='F')
    landmark=Landmark(1,position,frame,8)
    assert landmark.id==1, f"Expected id 1, got {landmark.id}"
    assert np.allclose(landmark.point3D,position), f"Expected 3d point {position} but got {landmark.point3D}"

def test_interface_creation():
    camera_interface=CameraCSVSource('/home/srikanth/machine_hall/machine_hall/MH_01_easy/MH_01_easy/mav0/cam1/data.csv',5)
    # imu_interface=IMUCSVSource('test_data/imu_data.csv')
    assert isinstance(camera_interface, CameraInterface), "Camera interface is not an instance of CameraCSVSource"
    image,timestamp=camera_interface.read_frame()
    assert image is not None, "Failed to read image from camera interface"
    assert timestamp is not None, "Failed to read timestamp from camera interface"
    print(timestamp)
    # assert isinstance(imu_interface, IMUInterface), "IMU interface is not an instance of IMUCSVSource"

def test_camera_creation():
    source = CameraCSVSource('/home/srikanth/machine_hall/machine_hall/MH_01_easy/MH_01_easy/mav0/cam1/data.csv', 5)
    extractor = ORBExtractor()
    camera=Camera(np.array([[457.587, 0, 379.999],[0, 456.134, 255.238],[0, 0, 1]],dtype=np.float32), np.array([-2.8368366e-01,  7.4512839e-02, -1.0473000e-04, -3.5559071e-05],dtype=np.float32), np.array([[ 0.01486554,-0.9998809,0.0041403,-0.02164015],[ 0.99955726,0.01496721,0.02571553,-0.06467699],[-0.02577444,0.00375619,0.99966073,0.00981073],[ 0.,0.,0.,1.]],dtype=np.float32, order='F'), source, extractor, 640, 480)
    frame=camera.get_frame(1)
    assert frame is not None, "Failed to get frame from camera"
    assert isinstance(frame, Frame), "Returned object is not an instance of Frame"
  

if __name__ == "__main__":
    test_object_creation()
    test_frame_creation()
    test_landmark_creation()