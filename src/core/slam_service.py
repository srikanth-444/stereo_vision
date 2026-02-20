
from ..config import load_config
from ..algorithms.feature_extraction import feature_extractor_factory
from ..algorithms.landmark import LandmarkManager
from ..algorithms.tracking import tracker_factory
from ..algorithms.pipeline import pipeline_factory
from ..algorithms.depth_estimator import Stereo
from ..sensors import Camera
from ..interfaces import interface_factory
from .. utils.visualize import Visualize
from ..algorithms.frame import FrameManager

from multiprocessing import Process, Manager

def start_service():
    print("SLAM service started")
    print("Loading configuration...")
    config=load_config('config.yaml')

    sensor_config=config.get('sensors',{})
    print("Identifying sensors...")
    cameras_config=sensor_config.get('cameras',{})
    camera_map={}
    for camera_config in cameras_config:
        intrinsic=camera_config.get('intrinsic',{})
        extrinsic= camera_config.get('extrinsic',{})
        distortion= camera_config.get('distortion',{})
        interface_config= camera_config.get('interface',{})
        interface=interface_factory(interface_config)
        camera=Camera(intrinsic=intrinsic, extrinsic=extrinsic,distortion=distortion, interface=interface)
        camera_map[camera_config.get('ID',{})]=camera
        print(f"Initialized camera with ID {camera_config.get('ID',{})}")    

    pipeline_config=config.get('pipe_line',{})
    print("Initializing pipeline...")

    landmark_manager=LandmarkManager()
    frame_manager=FrameManager(camera_map=camera_map)
    visual_odometry_config=pipeline_config.get('visual_odometry',{})
    feature_extractor_config=visual_odometry_config.get('feature_extractor',{})
    depth_estimator_config=visual_odometry_config.get('depth_estimator',{})
    tracker_config=visual_odometry_config.get('tracker',{})
    min_num_landmarks=visual_odometry_config.get('min_num_landmarks',100)

    feature_extractor=feature_extractor_factory(feature_extractor_config)
    tracker=tracker_factory(tracker_config,feature_extractor,camera_map[tracker_config.get('camera_id',0)])
    
    if depth_estimator_config.get('type')=='stereo':
        left_camera=camera_map[depth_estimator_config.get('left_camera_id',{})]
        right_camera=camera_map[depth_estimator_config.get('right_camera_id',{})]
        depth_estimator=Stereo(left_camera,right_camera)
    visualize=Visualize(frame_manager, landmark_manager)
    pipeline=pipeline_factory(landmark_manager, tracker, feature_extractor,frame_manager,depth_estimator,min_num_landmarks,visualize)
    
    print("Starting SLAM pipeline...")
    pipeline.run()
    visualize.run()


if __name__ == "__main__":
    start_service()



    

    

    

    


    
    