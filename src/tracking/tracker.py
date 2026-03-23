
def tracker_factory(config_load,optimizer,camera):
    reprojection_error=config_load.get('reprojection_error',{})
    confidence =config_load.get('confidence',{})
    iterationsCount=config_load.get('iterationsCount',{})
    camera=camera
    if config_load.get('type',{})=='descriptor':
        from .Tracker import DescriptorTracker
        return DescriptorTracker(reprojection_error,confidence,iterationsCount,camera,optimizer)
    if config_load.get('type',{})=='optical':
        from .optical import OpticalTracker
        FB_MAX_DIST=config_load.get('FB_MAX_DIST')
        win_size=config_load.get('win_size')
        max_level=config_load.get('max_level')
        return OpticalTracker(reprojection_error,confidence,iterationsCount,camera,FB_MAX_DIST,winSize=(win_size,win_size),maxLevel=max_level)
    raise ValueError(f"Unknown tracker type: {config_load.get('type',{})}")

    