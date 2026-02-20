import numpy as np

class Landmark:
    def __init__(self):
        self.position=None
        self.image_points=None
        self.descriptor=None
        self.observations=[]
        self.active=True
        self.id=None
        self.confidence=None
        self.frame_id=None
        self.all_descriptors=[]
        

    def update(self,position,image_points,frame_id):
        assert len(position)==3, "Position must be a 3D point."

        position = np.array(position)
        image_points = np.array(image_points)

        self.position=position
        self.image_points=image_points
        self.frame_id=frame_id
        
        
    
    def add_observation(self,id, image_point,descriptor):  
        observation={'frame_id':id, 'image_point':image_point}
        self.observations.append(observation)
        self.all_descriptors.append(descriptor)

