
from ..config import load_config
from ..feature_extractor import feature_extractor_factory
from ..atlas.Atlas import Atlas
from ..tracking import Tracker
from ..pipeline import pipeline_factory
from ..depth_estimator import Stereo
from ..sensors import Camera
from ..interfaces import interface_factory
from ..visualize import Visualize
from ..optimizer.Optimizer import Optimizer
from ..motion_model.motion_model import ConstantVelocity 

import logging


def start_service():
    logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d | %(name)s | %(levelname)-s | %(message)s',
    datefmt='%H:%M:%S'
)
    logging.info("SLAM service started")
    logging.info("Loading configuration...")
    config=load_config('config.yaml')

    sensor_config=config.get('sensors',{})
    logging.info("Identifying sensors...")
    cameras_config=sensor_config.get('cameras',{})
    camera_map={}

    pipeline_config=config.get('pipe_line',{})
    visual_odometry_config=pipeline_config.get('visual_odometry',{})
    
    for camera_config in cameras_config:
        intrinsic=camera_config.get('intrinsic',{})
        extrinsic= camera_config.get('extrinsic',{})
        distortion= camera_config.get('distortion_coefficients',[])
        interface_config= camera_config.get('interface',{})
        interface=interface_factory(interface_config)
        [w,h]=camera_config.get('resolution',[0,0])
        feature_extractor_config=visual_odometry_config.get('feature_extractor',{})
        feature_extractor=feature_extractor_factory(feature_extractor_config)
        camera=Camera(intrinsic=intrinsic, extrinsic=extrinsic,distortion=distortion, interface=interface,w=w,h=h,feature_extractor=feature_extractor)
        camera_map[camera_config.get('ID',{})]=camera
        logging.info(f"Initialized camera with ID {camera_config.get('ID',{})}")    

    motion_model=ConstantVelocity()
    logging.info("Initializing pipeline...")
    atlas=Atlas()
    depth_estimator_config=visual_odometry_config.get('depth_estimator',{})
    tracker_config=visual_odometry_config.get('tracker',{})
    optimizer=Optimizer(False)
    tracker=Tracker(tracker_config,optimizer,camera_map[tracker_config.get('camera_id',0)],atlas)
    
    if depth_estimator_config.get('type')=='stereo':
        left_camera=camera_map[depth_estimator_config.get('left_camera_id',{})]
        right_camera=camera_map[depth_estimator_config.get('right_camera_id',{})]
        depth_estimator=Stereo.Stereo(left_camera.intrinsic,left_camera.extrinsic,left_camera.distortion,
                                      right_camera.intrinsic,right_camera.extrinsic,right_camera.distortion,left_camera.W,left_camera.H)
    visualizer=Visualize(atlas)
    pipeline=pipeline_factory(atlas,tracker,depth_estimator,motion_model,visualizer,camera_map)
    
    logging.info("Starting SLAM pipeline...")
    pipeline.run()
    


if __name__ == "__main__":
    start_service()



    

    

    

    


    
    