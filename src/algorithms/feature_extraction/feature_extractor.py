def feature_extractor_factory(config_load):
    if config_load.get('type',{}) in ['orb','ORB']:
        from .orb import ORB
        nfeatures=config_load.get('nfeatures',2000)
        scaleFactor=config_load.get('scaleFactor',1.2)
        nlevels=config_load.get('nlevels',8)
        fastInitial=config_load.get('fastInitial',20)
        fastFinal=config_load.get('fastFinal',7)
        return ORB(nfeatures=nfeatures,scaleFactor=scaleFactor,nlevels=nlevels,fastInitial=fastInitial,fastFinal=fastFinal)
    raise ValueError(f"Unknown feature extractor type: {config_load.get('type')}")