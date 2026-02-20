class Registory:
    def __init__(self):
        self.sensors={}
        self.pipeline={}
        self.interfaces={}

    def create_sensors(self,config_load):
        from sensors.sensors import sensor_factory
        for sensor_name,sensor_config in config_load.items():
            self.sensors[sensor_name]=sensor_factory(sensor_config)

    def create_pipeline(self,config_load,feature_extractor,camera):
        
    