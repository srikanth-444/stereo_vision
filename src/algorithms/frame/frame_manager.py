from .frame import Frame
import numpy as np
class FrameManager():
    def __init__(self, camera_map):
        self.camera_map=camera_map
        self.frames=[]
        self.keyframe_map=set()
        self.index=0
        self.frame_id_by_camera_id={camera_id:set() for camera_id,_ in self.camera_map.items()}
        self.keyframes_voxel_map={}

    def capture_frames(self):
        current_frames=[]
        for camera_id, camera in self.camera_map.items():
            frame,timestamp=camera.get_frame()
            if frame is not None:
                frame=Frame(self.index, camera_id, frame, timestamp)
                self.frames.append(frame)
                self.frame_id_by_camera_id[camera_id].add(self.index)
                self.index+=1
                current_frames.append(frame)
        return current_frames
        
            
    def set_keyframe(self, id, threshold=0.1):
        self.frames[id].keyframe=True
        self.keyframe_map.add(id)
        t1=self.frames[id].T[:3,3]
        t1=np.floor(t1/threshold).astype(int)
        t1=tuple(t1)
        if t1 not in self.keyframes_voxel_map:
            self.keyframes_voxel_map[t1]=set()
        self.keyframes_voxel_map[t1].add(id)

    def get_frame_id_by_camera_id(self, camera_id):
        return self.frame_id_by_camera_id[camera_id]
    
    def get_current_frame_of_camera(self, camera_id):
        frame_ids=self.get_frame_id_by_camera_id(camera_id)
        if len(frame_ids)>0:
            return self.frames[max(frame_ids)]
        print(f"No frames found for camera_id: {camera_id}")
        return None
    
    def find_closest_keyframe(self, T, threshold=0.1):
        t1=T[:3,3]
        t1=np.floor(t1/threshold).astype(int)
        t1=tuple(t1)
        try:
            return self.keyframes_voxel_map[t1]
        except KeyError:
            return []