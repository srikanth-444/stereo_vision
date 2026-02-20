
from abc import ABC, abstractmethod

class Sensor(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_frame(self):
        pass

def sensor_factory(config_load):
    if config_load['sensor']['type']=='camera':
        intrinsic=config_load['camera']['intrinsic']
        distortion=config_load['camera']['distortion']
        extrinsic=config_load['camera']['extrinsic']
        from .camera import Camera
        return Camera(intrinsic,distortion,extrinsic)
    raise ValueError(f"Unknown sensor type: {config_load['sensor']['type']}")