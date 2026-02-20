import cv2
import numpy as np
from .pnp import PNPSolver
from .tracker import Tracker

class DescriptorTracker(Tracker):
    def __init__(self,reprojection_error,confidence,iterationsCount,feature_extractor,camera) -> None:
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.pnp_solver=PNPSolver(reprojection_error,confidence,iterationsCount,camera)
        self.feature_extractor=feature_extractor
    
    def track_landmarks(self, kp, des, landmarks):

        prev_desc = np.array([lm.descriptor for lm in landmarks], dtype=np.uint8)

        for lm in landmarks:
            lm.active = False

        new_points, new_desc = kp, des

        if len(prev_desc) == 0 or len(new_desc) == 0:
            return None, None

        matches = self.bf.knnMatch(new_desc, prev_desc, k=2)
        if matches is None or len(matches) == 0:
            return None, None

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            return None, None,None

        image_points = np.float32(
            [new_points[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 2)

        tracked_landmarks = [landmarks[m.trainIdx] for m in good_matches]
        object_points = np.array([lm.position for lm in tracked_landmarks])

        rvec, tvec, inliers = self.pnp_solver.estimate_pose_pnp(
            object_points, image_points
        )

        if inliers is None or len(inliers) < 4:
            return rvec, tvec, None

        inlier_query_indices = set()

        for i in inliers:
            idx = good_matches[i[0]].queryIdx
            lm = tracked_landmarks[i[0]]
            lm.image_points = image_points[i[0]]
            lm.active = True
            inlier_query_indices.add(idx)

        return rvec, tvec, inlier_query_indices


