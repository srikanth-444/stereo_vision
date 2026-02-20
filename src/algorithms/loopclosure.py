import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity



class Loop_closure:
    def __init__(self,max_features=2000):

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
                    return self.frame_ids[j]
        return None
   
                

