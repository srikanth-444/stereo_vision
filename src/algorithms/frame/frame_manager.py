from .frame import Frame
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
class FrameManager():
    def __init__(self, camera_map):
        self.camera_map=camera_map
        self.keyframe_map={}
        self.index=0
        self.frame_id_by_camera_id={camera_id:set() for camera_id,_ in self.camera_map.items()}

    def compute_bow(self,descriptors, vocab):
        # Map each descriptor to its visual word
        word_ids = [vocab.query_word(desc) for desc in descriptors]
        # print(word_ids)
        # Convert word IDs into indices
        # (dictionary word index from sorted leaf list)
        leaf_to_index = {wid: i for i, wid in enumerate(vocab.word_ids)}

        # Build histogram
        hist = np.zeros(len(vocab.word_ids), dtype=np.float32)
        for wid in word_ids:
            hist[leaf_to_index[wid]] += 1

        # Normalize (L2)
        norm = np.linalg.norm(hist) + 1e-6
        hist /= norm
        return hist
    
    # def match_by_keyframe(self, T, new_des):
    #     print(new_des.shape)
    #     if len(self.frames)<=2:
    #         return None
    #     frame_ids= list(self.keyframe_map)
    #     if len(frame_ids)==1:
    #         return self.frames[frame_ids[0]]
    #     descriptors_list=np.array([self.frames[frame_id].all_descriptors for frame_id in frame_ids])
    #     print(descriptors_list.shape)
    #     all_descriptors = np.vstack(descriptors_list)
    #     self.kmeans.fit(all_descriptors)
    #     bow_vectors = []
    #     for des in descriptors_list:
    #         bow_vectors.append(self.compute_bow(des))
    #     bow_vectors = np.vstack(bow_vectors)
    #     new_bow=self.compute_bow(new_des)
    #     scores = bow_vectors @ new_bow 
    #     best_idx = np.argmax(scores)
    #     return self.frames[frame_ids[best_idx]]

    def capture_frames(self):
        current_frames=[]
        for camera_id, camera in self.camera_map.items():
            frame,timestamp=camera.get_frame()
            if frame is not None:
                frame=Frame(self.index,camera_id, frame, timestamp,camera.new_intrinsic,camera.extrinsic)
                self.frame_id_by_camera_id[camera_id].add(self.index)
                self.index+=1
                current_frames.append(frame)
        return current_frames
        
            
    def set_keyframe(self, frame):
        frame.keyframe=True
        self.keyframe_map[frame.id]=frame
    
    def get_last_keyframe(self):
        my_list = sorted(list(self.keyframe_map.keys()))
        return self.keyframe_map[my_list[-1]]
    
    # def get_frame_id_by_camera_id(self, camera_id):
    #     return self.frame_id_by_camera_id[camera_id]
    
    # def get_current_frame_of_camera(self, camera_id):
    #     frame_ids=self.get_frame_id_by_camera_id(camera_id)
    #     if len(frame_ids)>0:
    #         return self.frames[max(frame_ids)]
    #     print(f"No frames found for camera_id: {camera_id}")
    #     return None
    
    def find_closest_keyframe(self):
        my_list = sorted(list(self.keyframe_map.keys()))
        return(my_list[-5:])
    
    def get_len_keyframes(self):
        return len(list(self.keyframe_map))
        
        