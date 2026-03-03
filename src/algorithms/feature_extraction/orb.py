import cv2
import numpy as np
from .feature_extractor import FeatureExtractor

class ORB(FeatureExtractor):
    def __init__(self,nfeatures=1000,scaleFactor=1.2,nlevels=8,edgeThreshold=31,firstLevel=0,WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE,patchSize=31,fastThreshold=20,anms=500):
        
        self.nfeatures = nfeatures

        self.orb = cv2.ORB_create(
            nfeatures=2000,   # detect many first
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            firstLevel=firstLevel,
            WTA_K=WTA_K,
            scoreType=scoreType,
            patchSize=patchSize,
            fastThreshold=fastThreshold
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.MIN_TRACKED = 50
        self.FB_MAX_DIST = 0.8
        self.LK_PARAMS = dict(winSize=(31,31), maxLevel=8,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.level_features = self.compute_level_features()

        self.fast_high = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        self.fast_low = cv2.FastFeatureDetector_create(threshold=7, nonmaxSuppression=True)

    def extract_features(self, frame):
        # frame = self.clahe.apply(frame)

        pyramid = [frame]
        scaleFactor = self.orb.getScaleFactor()
        nlevels = self.orb.getNLevels()

        for l in range(1, nlevels):
            resized = cv2.resize(
                pyramid[-1],
                None,
                fx=1/scaleFactor,
                fy=1/scaleFactor,
                interpolation=cv2.INTER_LINEAR
            )
            pyramid.append(resized)

        all_kp = []

        for level, img in enumerate(pyramid):

            
            kps = self.fast_high.detect(img, None)

            # If not enough → lower threshold
            if len(kps) < self.level_features[level]:
                
                kps = self.fast_low.detect(img, None)

            # Octree distribute per level
            kps = self.octree_distribute(
                kps,
                img.shape[1],
                img.shape[0],
                self.level_features[level]
            )

            # Rescale keypoints to original image
            scale = scaleFactor ** level
            for kp in kps:
                kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
                kp.octave = level

            all_kp.extend(kps)
        # Compute descriptors
        all_kp, des = self.orb.compute(frame, all_kp)
        return all_kp, des
    
    def compute_level_features(self):
        nlevels = self.orb.getNLevels()
        scaleFactor = self.orb.getScaleFactor()
        N = self.nfeatures

        factor = 1.0 / scaleFactor
        nDesired = []

        # geometric series normalization
        total = 0
        for l in range(nlevels):
            total += factor ** l

        for l in range(nlevels):
            nl = int(round(N * (factor ** l) / total))
            nDesired.append(nl)

        return nDesired
    def grid_suppress(self, keypoints, width, height, N, grid_rows=8, grid_cols=8):
        if len(keypoints) <= N:
            return keypoints

        cell_w = width / grid_cols
        cell_h = height / grid_rows

        # Create empty grid
        grid = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]

        # Assign keypoints to cells
        for kp in keypoints:
            col = int(kp.pt[0] / cell_w)
            row = int(kp.pt[1] / cell_h)

            # Clamp boundary case
            col = min(col, grid_cols - 1)
            row = min(row, grid_rows - 1)

            grid[row][col].append(kp)

        # Select strongest from each cell
        selected = []
        per_cell = max(1, N // (grid_rows * grid_cols))

        for r in range(grid_rows):
            for c in range(grid_cols):
                if not grid[r][c]:
                    continue

                # sort by response
                cell_kp = sorted(grid[r][c], key=lambda kp: kp.response, reverse=True)
                selected.extend(cell_kp[:per_cell])

        # If still too many, globally sort and trim
        if len(selected) > N:
            selected = sorted(selected, key=lambda kp: kp.response, reverse=True)
            selected = selected[:N]

        return selected
    def octree_distribute(self, keypoints, width, height, N):

        if len(keypoints) <= N:
            return keypoints

        root = ExtractorNode(0, width, 0, height, keypoints)
        nodes = [root]

        while len(nodes) < N:
            new_nodes = []
            expandable = False

            for node in nodes:
                if len(node.keypoints) > 1:
                    children = node.divide()
                    if len(children) > 1:
                        new_nodes.extend(children)
                        expandable = True
                    else:
                        new_nodes.append(node)
                else:
                    new_nodes.append(node)

            nodes = new_nodes

            if not expandable:
                break

        # Select strongest from each node
        final_kp = []
        for node in nodes:
            best = max(node.keypoints, key=lambda kp: kp.response)
            final_kp.append(best)

        # If we still have too many (rare), sort globally
        if len(final_kp) > N:
            final_kp = sorted(final_kp, key=lambda kp: kp.response, reverse=True)
            final_kp = final_kp[:N]

        return final_kp

        
class ExtractorNode:
    def __init__(self, x_min, x_max, y_min, y_max, keypoints):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.keypoints = keypoints

    def divide(self):
        mx = (self.x_min + self.x_max) / 2
        my = (self.y_min + self.y_max) / 2

        q1, q2, q3, q4 = [], [], [], []

        for kp in self.keypoints:
            x, y = kp.pt
            if x < mx:
                if y < my:
                    q1.append(kp)
                else:
                    q3.append(kp)
            else:
                if y < my:
                    q2.append(kp)
                else:
                    q4.append(kp)

        children = []
        if q1: children.append(ExtractorNode(self.x_min, mx, self.y_min, my, q1))
        if q2: children.append(ExtractorNode(mx, self.x_max, self.y_min, my, q2))
        if q3: children.append(ExtractorNode(self.x_min, mx, my, self.y_max, q3))
        if q4: children.append(ExtractorNode(mx, self.x_max, my, self.y_max, q4))

        return children


import numpy as np
import cv2

class ORBVocabulary:
    def __init__(self, path):
        self.path = path
        self.nodes = {}      # node_id -> descriptor (np.uint8 array)
        self.children = {}   # node_id -> list of child_ids
        self.word_ids = []   # final leaf IDs (visual words)
        self._load_txt()

    def _load_txt(self):
        with open(self.path, "r") as f:
            # Skip header lines (usually the first 4-5 lines define k, L, etc.)
            lines = f.readlines()

        for line in lines:
            parts = line.split()
            if len(parts) < 34: # ID + Parent + 32 bytes = 34 minimum
                continue

            node_id = int(parts[0])
            parent_id = int(parts[1])
            
            # Read the 32 descriptor bytes (indices 2 to 33)
            desc_values = [int(x) for x in parts[2:34]]
            self.nodes[node_id] = np.array(desc_values, dtype=np.uint8)

            # In DBoW2, we build the children list by looking at parents
            if parent_id != -1:
                if parent_id not in self.children:
                    self.children[parent_id] = []
                self.children[parent_id].append(node_id)

            # Check if there is a weight at the end (index 34)
            # If it exists, this node is a leaf (a "Word")
            if len(parts) > 34:
                weight = float(parts[34])
                self.word_ids.append(node_id)
                # You might want to store self.weights[node_id] = weight

    def query_word(self, descriptor):
        """
        Traverse the tree to find the leaf node (Word ID).
        In ORBvoc.txt, the top-level nodes all have Parent ID = 0.
        """
        # Start by getting the children of the virtual root (0)
        current_children = self.children.get(0, [])
        
        if not current_children:
            raise ValueError("Vocabulary tree is empty or not loaded correctly.")

        node_id = 0 # Virtual root
        
        while True:
            # Get children of the current node
            nodes_to_compare = self.children.get(node_id, [])
            
            # If no children, we've reached a leaf (a Word)
            if not nodes_to_compare:
                return node_id

            best_child = -1
            best_dist = 257 # Max Hamming distance for 256 bits is 256

            for c_id in nodes_to_compare:
                # Calculate Hamming Distance
                # Using XOR and bit_count is the fastest way in Python 3.10+
                # If using older Python/Numpy, use: np.count_nonzero(descriptor != self.nodes[c_id])
                dist = self._hamming_fast(descriptor, self.nodes[c_id])
                
                if dist < best_dist:
                    best_dist = dist
                    best_child = c_id
            
            # Move down the tree
            node_id = best_child

    def _hamming_fast(self, a, b):
        # a and b are numpy uint8 arrays of length 32
        # XOR the arrays, then count bits
        return np.unpackbits(np.bitwise_xor(a, b)).sum()