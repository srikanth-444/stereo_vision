from abc import ABC, abstractmethod

class interface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def read_frame(self):
        pass
    @abstractmethod
    def release(self):
        pass

def interface_factory(config):
    if config.get('type',{})=='video_playback':
        from .video_playback import VideoPlayback
        return VideoPlayback(config.get('path',{}))
    if config.get('type',{})=='csv_reader':
        from .csv_reader import CSVReader
        return CSVReader(config.get('path',{}))
    if config.get('type',{})=='camera_usb':
        from .camera_usb import CameraUSB
        camera_idx = config.get('path',{})
        width = config.get('width',{})
        height = config.get('height',{})
        fps = config.get('fps',{})
        return CameraUSB(camera_idx, width, height, fps)
    raise ValueError(f"Unknown interface type: {config['interface']['type']}")
    