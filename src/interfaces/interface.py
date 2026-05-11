def camera_interface_factory(config, queue_size):

    if config.get('type',{})=='csv_reader':
        from ..interfaces.Interface import CameraCSVSource   
        return CameraCSVSource(config.get('path',{}), queue_size)
    raise ValueError(f"Unknown interface type: {config['interface']['type']}")
    
def imu_interface_factory(config):
    if config.get('type',{})=='csv_reader':
        from ..interfaces.Interface import ImuCSVSource  
        return ImuCSVSource(config.get('path',{}))
    raise ValueError(f"Unknown interface type: {config['interface']['type']}")