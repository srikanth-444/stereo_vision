import numpy as np
import cv2
from .pnp import PNPSolver
from .tracker import Tracker

class OpticalTracker(Tracker):
    def __init__(self,reprojection_error,confidence,iterationsCount,camera,FB_MAX_DIST,winSize=(31,31),maxLevel=8) -> None:
        self.LK_PARAMS = dict(winSize=winSize, maxLevel=8,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.pnp_solver=PNPSolver(reprojection_error,confidence,iterationsCount,camera)
        self.FB_MAX_DIST = FB_MAX_DIST


    def track_landmarks(self,previous_frame,frame,land_marks):

            tracked_ids = []
            tracked_curr_pts = []
            print(f"Tracking {len(land_marks)} landmarks")
            if len(land_marks) > 0 :
        
                prev_pts = np.array([lm.image_points for lm in land_marks if lm.active], dtype=np.float32)
                ids = np.array([lm.id for lm in land_marks if lm.active], dtype=np.int32)
                for lm in land_marks:
                    lm.active = False 
                print(f"Tracking {len(prev_pts)} active landmarks")
                if prev_pts.size > 0:
                
                    pts_next, st, err = cv2.calcOpticalFlowPyrLK(previous_frame, frame, prev_pts, None, **self.LK_PARAMS)
                    st = st.reshape(-1)
                    pts_next = pts_next.reshape(-1,2)

                    # Keep only successfully tracked
                    good_prev = prev_pts[st == 1]
                    good_next = pts_next[st == 1]
                    good_ids = ids[st == 1]
                
                    # Forward-backward check: track back good_next -> prev and compare
                    pts_back, st2, err2 = cv2.calcOpticalFlowPyrLK(frame, previous_frame, good_next, None, **self.LK_PARAMS)
                    st2 = st2.reshape(-1)
                    pts_back = pts_back.reshape(-1,2)
                    # print(f"left tracked: {len(good_prev)}, right tracked: {len(good_r_prev)}")
                    fb_err = np.linalg.norm(good_prev - pts_back, axis=1)
                    fb_mask = (fb_err < self.FB_MAX_DIST)

                    # Final accepted tracks
                    final_next = good_next[fb_mask]
                    final_ids = good_ids[fb_mask]
                    tracked_ids = final_ids.tolist()
                    tracked_curr_pts = final_next
                    print(f"After FB check: {len(tracked_ids)} tracked landmarks")
                    if len(tracked_ids) > 0:
                        id_to_lm = {lm.id: lm for lm in land_marks}
                        for id_, pt2d in zip(tracked_ids, tracked_curr_pts):
                            lm = id_to_lm.get(id_)
                            if lm is not None:
                                lm.image_points = pt2d  # update stored 2D location for the landmark
                                lm.active = True  # mark as active
                    object_points=[id_to_lm[id_].position for id_ in tracked_ids]
                    image_points=[id_to_lm[id_].image_points for id_ in tracked_ids]
                    print(f"Tracked {len(tracked_ids)} landmarks after FB check")
                    if len(object_points) < 4 or len(image_points) < 4:
                        return None, None
                    rvec,tvec,_=self.pnp_solver.estimate_pose_pnp(object_points,image_points)
                    return rvec,tvec
                    
           