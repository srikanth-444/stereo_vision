
from ..config import load_config
from ..feature_extractor import feature_extractor_factory
from ..atlas.Atlas import Atlas
from ..tracking import Tracker
from ..pipeline import pipeline_factory
from ..depth_estimator import Stereo
from ..interfaces import camera_interface_factory, imu_interface_factory
from ..interfaces.Interface import CameraInterface
from ..sensors.Sensors import Camera, IMU
from ..visualize.Visualize import Visualize
from ..optimizer.Optimizer import Optimizer
from ..motion_model.motion_model import ConstantVelocity 
import numpy as np
import logging


def start_service():
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(name)s | %(levelname)-s | %(message)s',
    datefmt='%H:%M:%S'
)
    logging.info("SLAM service started")
    logging.info("Loading configuration...")
    config=load_config('config.yaml')

    sensor_config=config.get('sensors',{})
    logging.info("Identifying sensors...")
    cameras_config=sensor_config.get('cameras',{})
    imu_config=sensor_config.get('imu',{})
    camera_map={}

    pipeline_config=config.get('pipe_line',{})
    visual_odometry_config=pipeline_config.get('visual_odometry',{})
    
    for camera_config in cameras_config:
        intrinsic=camera_config.get('intrinsic',{})
        intrinsic=np.array([[intrinsic.get('fx',0), 0, intrinsic.get('cx',0)],
                            [0, intrinsic.get('fy',0), intrinsic.get('cy',0)],
                            [0, 0, 1]],dtype=np.float32)
        extrinsic= camera_config.get('extrinsic',{})
        extrinsic=np.array(extrinsic,dtype=np.float32).reshape(4, 4)
        distortion= camera_config.get('distortion_coefficients',[])
        distortion=np.array(distortion,dtype=np.float32)
        interface_config= camera_config.get('interface',{})
        interface=camera_interface_factory(interface_config,2)
        [w,h]=camera_config.get('resolution',[0,0])
        feature_extractor_config=visual_odometry_config.get('feature_extractor',{})
        feature_extractor=feature_extractor_factory(feature_extractor_config)
        camera=Camera(intrinsic,distortion,extrinsic,interface,feature_extractor, w, h)
        camera_map[camera_config.get('ID',{})]=camera
        logging.info(f"Initialized camera with ID {camera_config.get('ID',{})}")    
    gyro_noise_density=imu_config.get('noise',{}).get('gyro_noise_density',0)
    accel_noise_density=imu_config.get('noise',{}).get('accel_noise_density',0)
    gyro_bias_rw=imu_config.get('noise',{}).get('gyro_bias_rw',0)
    accel_bias_rw=imu_config.get('noise',{}).get('accel_bias_rw',0)
    rate_hz=imu_config.get('rate_hz',0)
    imu=IMU(gyro_noise_density, gyro_bias_rw, accel_noise_density, accel_bias_rw, rate_hz, imu_interface_factory(imu_config.get('interface',{})))

    motion_model=ConstantVelocity()
    logging.info("Initializing pipeline...")
    
    depth_estimator_config=visual_odometry_config.get('depth_estimator',{})
    tracker_config=visual_odometry_config.get('tracker',{})
    optimizer=Optimizer(False)
    atlas=Atlas(optimizer)
    tracker=Tracker(tracker_config,optimizer,camera_map[tracker_config.get('camera_id',0)],atlas,motion_model)
    
    if depth_estimator_config.get('type')=='stereo':
        left_camera=camera_map[depth_estimator_config.get('left_camera_id',{})]
        right_camera=camera_map[depth_estimator_config.get('right_camera_id',{})]
        depth_estimator=Stereo.Stereo(left_camera.get_intrinsic(),left_camera.get_extrinsic(),np.array([0,0,0,0]),
                                      right_camera.get_intrinsic(),right_camera.get_extrinsic(),np.array([0,0,0,0]),left_camera.get_width(),left_camera.get_height())
    visualizer=Visualize(atlas)
    pipeline=pipeline_factory(atlas,tracker,depth_estimator,visualizer,camera_map,optimizer)
    logging.info("Starting SLAM pipeline...")
    pipeline.run()
    


if __name__ == "__main__":
    start_service()
