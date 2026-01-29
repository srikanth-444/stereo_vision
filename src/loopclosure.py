import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


class Loop_closure:
    def __init__(self, video_path, frame_stride=5, max_features=2000):
        self.frame_ids = []
        self.descriptors_list = []
        self.frame_images = []
        self.N_CLUSTERS = 800
        self.LOOP_THRESHOLD = 0.70

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.N_CLUSTERS,
                batch_size=2000,
                verbose=0
            )

    def compute_bow(self,des):
        words = self.kmeans.predict(des)
        hist, _ = np.histogram(words, bins=np.arange(self.N_CLUSTERS+1))
        hist = hist.astype(np.float32)
        hist /= (np.linalg.norm(hist) + 1e-6)
        return hist
        
    def add_keyframe(self, frame_id, descriptors, frame_image):
        self.frame_ids.append(frame_id)
        self.descriptors_list.append(descriptors)
        self.frame_images.append(frame_image) 

    def loop_check(self):

        all_descriptors = np.vstack(self.descriptors_list)
        self.kmeans.fit(all_descriptors)
        bow_vectors = []
    
        for des in self.descriptors_list:
            bow_vectors.append(self.compute_bow(des))
        bow_vectors = np.vstack(bow_vectors)
        similarity_matrix = cosine_similarity(bow_vectors)

        for i in range(len(self.frame_ids)):
            for j in range(i - 10):
                if similarity_matrix[i, j] > self.LOOP_THRESHOLD:
                    print(f"Frame {self.frame_ids[i]}  <-->  Frame {self.frame_ids[j]}  | score = {similarity_matrix[i,j]:.2f}")
                    # visualize = np.hstack((self.frame_images[i], self.frame_images[j]))
                    # cv2.imshow(f"Revisit: Frame {self.frame_ids[i]} & Frame {self.frame_ids[j]}", visualize)
                    # cv2.waitKey(0)
                    return self.frame_ids[j]
        return None
   
                

