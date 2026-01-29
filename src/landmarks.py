import numpy as np

class landmarks:
    def __init__(self):
        self.position=None
        self.r_image_points=None
        self.image_points=None
        self.descriptor=None
        self.observations=[]
        self.active=True
        self.id=None

    def update(self,position,r_image_points,image_points):
        assert len(position)==3, "Position must be a 3D point."
        assert len(r_image_points)>0, "Descriptor cannot be empty."

        position = np.array(position)
        r_image_points = np.array(r_image_points)
        image_points = np.array(image_points)

        self.position=position
        self.r_image_points=r_image_points
        self.image_points=image_points
        
        
    
    def add_observation(self,id, image_point, r_image_point):
        
        observation={'frame_id':id, 'image_point':image_point, 'r_image_point': r_image_point }
        self.observations.append(observation)

