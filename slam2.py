import cv2
import numpy as np
import matplotlib.pyplot as plt
from camera import stereo_camera
from camera import anms
from landmarks import landmarks, bundel_adjustment
from time import time
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation 
from loopclosure import Loop_closure
from mpl_toolkits.mplot3d import Axes3D


class ORB_SLAM:
    def __init__(self,l_k, r_k, l_dist, r_dist, R, t):
        self.cam = stereo_camera(l_k, r_k, l_dist, r_dist, R, t)
        self.orb = cv2.ORB_create(nfeatures=2000,scaleFactor=1.2,nlevels=8,edgeThreshold=31,firstLevel=0,
                                  WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE,patchSize=31,fastThreshold=20)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.loop_closure = Loop_closure(video_path=None)

        # Parameters
        self.MIN_TRACKED = 50
        self.FB_MAX_DIST = 0.8
        self.LK_PARAMS = dict(winSize=(31,31), maxLevel=8,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.land_marks = []
        self.trajectory = []
        self.T_global = np.eye(4)

        self.prev_gray_left = None
        self.prev_gray_right = None
        self.prev_left_pts = None  
        self.prev_ids = None      
        
        self.next_landmark_id = 0  
        self.previous_time = time()
        self.count_frame = 0

        self.camera_poses = []
        self.pose_graph_nodes = []
        self.pose_graph_edges = []

    def get_frames(self,cap):
        ret, frame = cap.read()
        if self.count_frame>1:
            if not ret:
                fps=(time()-self.previous_time)/self.count_frame
                print(f"Average FPS: {1.0/fps}")
                return None, None, False

            right_img = frame[:, :frame.shape[1] // 2]
            left_img = frame[:, frame.shape[1] // 2:]

            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_left = cv2.rotate(gray_left, cv2.ROTATE_180)
            gray_right = cv2.rotate(gray_right, cv2.ROTATE_180)

            
            
            gray_left = self.clahe.apply(gray_left)
            gray_right = self.clahe.apply(gray_right)
            return gray_left, gray_right, True

        else:
            return None, None, True


    def track_landmarks(self,gray_left, gray_right):

        tracked_ids = []
        tracked_curr_pts = []

        if len(self.land_marks) > 0 and self.prev_gray_left is not None:
    
            prev_pts = np.array([lm.image_points for lm in self.land_marks if lm.active], dtype=np.float32)
            r_prev_pts = np.array([lm.r_image_points for lm in self.land_marks if lm.active], dtype=np.float32)
            ids = np.array([lm.id for lm in self.land_marks if lm.active], dtype=np.int32)
            for lm in self.land_marks:
                lm.active = False 

            if prev_pts.size > 0:
               
                pts_next, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray_left, gray_left, prev_pts, None, **self.LK_PARAMS)
                st = st.reshape(-1)
                pts_next = pts_next.reshape(-1,2)

                r_pts_next, st_r, err_r = cv2.calcOpticalFlowPyrLK(self.prev_gray_right, gray_right, r_prev_pts, None, **self.LK_PARAMS)
                st_r = st_r.reshape(-1)
                r_pts_next = r_pts_next.reshape(-1,2)

                # Keep only successfully tracked
                good_prev = prev_pts[st == 1]
                good_next = pts_next[st == 1]
                good_r_prev = r_prev_pts[st_r == 1]
                good_r_next = r_pts_next[st_r == 1]
                good_ids = ids[st == 1]
                good_r_ids = ids[st_r == 1]

                # Forward-backward check: track back good_next -> prev and compare
                pts_back, st2, err2 = cv2.calcOpticalFlowPyrLK(gray_left, self.prev_gray_left, good_next, None, **self.LK_PARAMS)
                st2 = st2.reshape(-1)
                pts_back = pts_back.reshape(-1,2)

                r_pts_back, st2_r, err2_r = cv2.calcOpticalFlowPyrLK(gray_right, self.prev_gray_right, good_r_next, None, **self.LK_PARAMS)
                st2_r = st2_r.reshape(-1)
                r_pts_back = r_pts_back.reshape(-1,2)

                # print(f"left tracked: {len(good_prev)}, right tracked: {len(good_r_prev)}")

                # Compute FB error
                fb_err = np.linalg.norm(good_prev - pts_back, axis=1)
                r_fb_err = np.linalg.norm(good_r_prev - r_pts_back, axis=1)
                fb_mask = (fb_err < self.FB_MAX_DIST)
                fb_r_mask = (r_fb_err < self.FB_MAX_DIST)

                # Final accepted tracks
                final_prev = good_prev[fb_mask]
                final_next = good_next[fb_mask]
                final_ids = good_ids[fb_mask]
                final_r_prev = good_r_prev[fb_r_mask]
                final_r_next = good_r_next[fb_r_mask]
                final_r_ids = good_r_ids[fb_r_mask]

                tracked_ids = final_ids.tolist()
                tracked_r_ids = final_r_ids.tolist()
                tracked_prev_pts = final_prev
                tracked_r_prev_pts = final_r_prev
                tracked_curr_pts = final_next
                tracked_r_curr_pts = final_r_next
                # valid_landmarks = []
                
                for lm in self.land_marks:
                    if lm.id in tracked_ids and lm.id in tracked_r_ids:
                
                        x, y = lm.image_points
                        if 0 <= x < gray_left.shape[1] and 0 <= y < gray_left.shape[0]:
                            lm.active = True
                        else:
                            lm.active = False
                if len(tracked_ids) > 0:
                    id_to_lm = {lm.id: lm for lm in self.land_marks}
                    for id_, pt2d in zip(tracked_ids, tracked_curr_pts):
                        lm = id_to_lm.get(id_)
                        if lm is not None:
                            lm.image_points = pt2d  # update stored 2D location for the landmark
                    for id_, r_pt2d in zip(tracked_r_ids, tracked_r_curr_pts):
                        lm = id_to_lm.get(id_)
                        if lm is not None:
                            lm.r_image_points = r_pt2d  # update stored 2D location for the landmark

            # ---- Visualize tracked points on left image ----
            left_vis  = cv2.cvtColor(gray_left,  cv2.COLOR_GRAY2BGR)
            right_vis = cv2.cvtColor(gray_right, cv2.COLOR_GRAY2BGR)
            # print(f"Frame {self.count_frame}: Tracked {len(tracked_ids)} landmarks.")
            vis = np.hstack((left_vis, right_vis))
            for lm in self.land_marks:
                if not lm.active:
                    continue
                x, y = int(lm.image_points[0]), int(lm.image_points[1])
                x_r, y_r = int(lm.r_image_points[0])+gray_left.shape[1], int(lm.r_image_points[1])
                cv2.circle(vis, (x,y), 3, (0,255,0), -1)
                cv2.circle(vis, (x_r,y_r), 3, (255,0,0), -1)

                cv2.putText(vis, f"{lm.id}", (x+3,y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1)
                cv2.putText(vis, f"{lm.id}", (x_r+3,y_r-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1)
            cv2.circle(vis, (int(self.cam.l_K[0,2]), int(self.cam.l_K[1,2])), 5, (0,0,255), -1)  # just a marker for visualization
            cv2.putText(vis, f"Frame: {self.count_frame}, Tracked: {len([lm for lm in self.land_marks if lm.active])}, Landmarks: {len(self.land_marks)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            self.frame_vis=vis.copy()
            cv2.imshow("Tracked", vis)
            cv2.waitKey(1)
        
    def estimate_pose_pnp(self,):
        object_points = []
        image_points = []
        image_r_points = []
        land_id=[]
        if len([lm for lm in self.land_marks if lm.active]) >= 6:
            for lm in self.land_marks:
                if lm.active :
                    object_points.append(lm.position)
                    image_points.append(lm.image_points)
                    image_r_points.append(lm.r_image_points)
                    land_id.append(lm.id)
            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)
            image_r_points = np.array(image_r_points, dtype=np.float32)
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, self.cam.l_K, self.cam.l_dist, flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=1.5, confidence=0.99, iterationsCount=500)

            if retval:
                reproj = cv2.projectPoints(object_points, rvec, tvec, self.cam.l_K, self.cam.l_dist)[0].squeeze()
                reproj_err = np.linalg.norm(reproj - image_points, axis=1)
                land_id = [land_id[i] for i in inliers.flatten()]
                # print("PnP reprojection error:", np.mean(reproj_err))
                for lm in self.land_marks:
                    lm.add_observation( len(self.camera_poses), lm.image_points,lm.r_image_points)
               
                R_pnp, _ = cv2.Rodrigues(rvec)
                t_pnp = tvec.reshape(3,1)
                T_increment = np.eye(4)
                T_increment[:3,:3] = R_pnp.T
                T_increment[:3,3] = -R_pnp.T @ t_pnp.flatten()
                # T_increment[:3,3] = 0.9*self.T_global[:3,3]+0.1* T_increment[:3,3]
                self.T_global = T_increment
                self.camera_poses.append({"frame_id": len(self.camera_poses), "rvec": rvec, "tvec": tvec,"T_global": self.T_global.copy()})
                self.trajectory.append(self.T_global[:3,3].copy())
                if self.count_frame % 20 == 0:
                    self.pose_graph_nodes.append(len(self.camera_poses)-1)
                    # print(len(self.pose_graph_nodes))
                    if len(self.pose_graph_nodes) > 1:
                        i = self.pose_graph_nodes[-2]
                        j = self.pose_graph_nodes[-1]
                        T_i = self.camera_poses[i]['T_global']
                        T_j = self.camera_poses[j]['T_global']
                        Z_ij = np.linalg.inv(T_i) @ T_j
                        self.pose_graph_edges.append((i, j, Z_ij))
                # print(len(self.pose_graph_edges))
            else:
                print("PnP failed. at frame", self.count_frame)
                plt.plot(self.trajectory[-1][0],self.trajectory[-1][2],'rx')  # mark failure
    
    def pose_graph_residuals(self, x):
        # x = flattened 6D poses (rvec + tvec) for all keyframes
        n = len(self.pose_graph_nodes)
        res = []
        # print(x.shape)
        for k,(i, j, Z_ij) in enumerate(self.pose_graph_edges):
            #print(i, j)
            r_i = x[k*6:k*6+3]
            t_i = x[k*6+3:k*6+6]
            r_j = x[(k+1)*6:(k+1)*6+3]
            t_j = x[(k+1)*6+3:(k+1)*6+6]

            print(r_i, t_i, r_j, t_j)

            T_i = np.eye(4)
            T_i[:3,:3] = cv2.Rodrigues(r_i)[0]
            T_i[:3,3] = t_i

            T_j = np.eye(4)
            T_j[:3,:3] = cv2.Rodrigues(r_j)[0]
            T_j[:3,3] = t_j

            E = np.linalg.inv(Z_ij) @ np.linalg.inv(T_i) @ T_j
            r_err = Rotation.from_matrix(E[:3,:3]).as_rotvec()
            t_err = E[:3,3]

            res.extend(r_err)
            res.extend(t_err)
        return np.array(res)
    
    def optimize_pose_graph(self):
        n = len(self.pose_graph_nodes)
        x0 = []
        for node_idx in self.pose_graph_nodes:
            pose = self.camera_poses[node_idx]
            x0.extend(pose['rvec'].flatten())
            x0.extend(pose['tvec'].flatten())
        x0 = np.array(x0)

        # Fix first node by zeroing its residuals (or remove its variables)
        def wrapper(x):
            x[0:6] = x0[0:6]  # keep first pose fixed
            return self.pose_graph_residuals(x)

        result = least_squares(wrapper, x0, verbose=2, ftol=1e-2, xtol=1e-4, gtol=1e-4, max_nfev=100)

        # Update camera poses
        for idx, node_idx in enumerate(self.pose_graph_nodes):
            self.camera_poses[node_idx]['rvec'] = result.x[idx*6:idx*6+3].reshape(3,1)
            self.camera_poses[node_idx]['tvec'] = result.x[idx*6+3:idx*6+6].reshape(3,1)

            # Update T_global
            R_opt = cv2.Rodrigues(self.camera_poses[node_idx]['rvec'])[0]
            t_opt = self.camera_poses[node_idx]['tvec'].reshape(3,1)
            T_opt = np.eye(4)
            T_opt[:3,:3] = R_opt
            T_opt[:3,3] = t_opt.flatten()
            self.camera_poses[node_idx]['T_global'] = T_opt
    

    def create_new_landmarks(self, gray_left, gray_right, nonlinear_triangulation=True):
        
            kp1, des1 = self.orb.detectAndCompute(gray_left, None)
            kp2, des2 = self.orb.detectAndCompute(gray_right, None)

            self.loop_closure.add_keyframe(self.count_frame, des1, gray_left)
            closed=self.loop_closure.loop_check()
            if closed is not None:
                print(f"Loop closure detected with frame {closed} at frame {self.count_frame}")
                #self.T_global=self.camera_poses[closed]['T_global']
            mask = np.array([kp.response > 1.e-5 for kp in kp1])
            kp1 = [kp for kp, m in zip(kp1, mask) if m]
            des1 = des1[mask]
        
            kp2_mask = np.array([kp.response > 1.e-5 for kp in kp2])
            kp2 = [kp for kp, m in zip(kp2, kp2_mask) if m]
            des2 = des2[kp2_mask]
 
            keep_idx1 = anms(kp1, num_keep=500)
            keep_idx2 = anms(kp2, num_keep=500)
        
            kp1 = [kp1[i] for i in keep_idx1]
            des1 = des1[keep_idx1]
            kp2 = [kp2[i] for i in keep_idx2]
            des2 = des2[keep_idx2]

            if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
                matches = self.bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                ptsL = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,2)
                desL = np.array([des1[m.queryIdx] for m in matches])
                ptsR = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,2)
                desR = np.array([des2[m.trainIdx] for m in matches])
                points_3d, reprojection_errors = self.cam.triangulate_points(ptsL, ptsR)
                if nonlinear_triangulation:
                    points_3d= [p for i, p in enumerate(points_3d) if reprojection_errors[i] < 3.0]
                    ptsL= [p for i, p in enumerate(ptsL) if reprojection_errors[i] < 3.0]
                    ptsR= [p for i, p in enumerate(ptsR) if reprojection_errors[i] < 3.0]
                    desL= [d for i, d in enumerate(desL) if reprojection_errors[i] < 3.0]  
                    desR= [d for i, d in enumerate(desR) if reprojection_errors[i] < 3.0] 
                    points_3d=np.array(points_3d)
                    points_3d,reprojection_errors = self.cam.nonlinear_triangulation( [ptsL, ptsR], points_3d) 
                points_3d = np.array(points_3d)
                points_3d= self.T_global[:3,:3] @ points_3d.T + self.T_global[:3,3:4]
                points_3d= points_3d.T.tolist()  

                new_marks = []
                for i, err in enumerate(reprojection_errors):
                    if err > 1.0:
                        continue
                    m = landmarks()  
                    m.position = points_3d[i]         
                    m.r_image_points = ptsR[i].copy()   
                    m.image_points = ptsL[i].copy()
                    m.descriptor = desL[i].copy()
                    m.id = self.next_landmark_id
                    self.next_landmark_id += 1
                    new_marks.append(m)
                final_add = []

                for nm in new_marks:
                    add = True

                    for lm in self.land_marks:
                        desc_dist = cv2.norm(nm.descriptor, lm.descriptor, cv2.NORM_HAMMING)
                        if desc_dist < 50:
                            add = False
                            lm.active=True  # reactivate existing landmark
                            break
                        # if nm.position is not None and lm.position is not None:
                        #     pos_dist = np.linalg.norm(np.array(nm.position) - np.array(lm.position))
                        #     if pos_dist < 10:  # 10 cm threshold
                        #         add = False
                        #         lm.active=True
                        #         break
                        # if lm.active == True:
                        #     distance = np.linalg.norm(np.array(nm.image_points) - np.array(lm.image_points))
                        #     if distance < 5.0:  # 5 pixels threshold
                        #         add = False
                        #         lm.active=True
                        #         break
                    if add:
                        final_add.append(nm)

                self.land_marks.extend(final_add)
                
        

if __name__ == "__main__":
    video_path = 'output_video.mp4'
    l_k='camera_mat/leftintrinsic.npy'
    r_k='camera_mat/rightintrinsic.npy'
    l_dist='camera_mat/leftdistortion.npy'
    r_dist='camera_mat/rightdistortion.npy'
    R='camera_mat/r_matrix.npy'
    t='camera_mat/t_matrix.npy'

    slam = ORB_SLAM(l_k, r_k, l_dist, r_dist, R, t)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        gray_left, gray_right, ret = slam.get_frames(cap)
        if not ret:
            break
        if gray_left is not None and gray_right is not None:
            slam.track_landmarks(gray_left, gray_right)
            slam.estimate_pose_pnp()
            if slam.count_frame % 20==0:
                slam.create_new_landmarks(gray_left, gray_right,nonlinear_triangulation=False)
            slam.prev_gray_left = gray_left.copy()
            slam.prev_gray_right = gray_right.copy()
        # if slam.count_frame>14:
        #     break
        slam.count_frame += 1

    np.save('camera_poses.npy', slam.camera_poses)
    
    # bundel_adjustment( slam.land_marks, slam.camera_poses, slam.cam)
    

    
    lx= [lm.position[0] for lm in slam.land_marks]
    lz = [lm.position[2] for lm in slam.land_marks]
    trajectory=[]
    for node_idx in slam.camera_poses:
        rvec = node_idx['rvec']
        tvec = node_idx['tvec']
        R_pnp, _ = cv2.Rodrigues(rvec)
        t_pnp = tvec.reshape(3,1)
        T_increment = np.eye(4)
        T_increment[:3,:3] = R_pnp.T
        T_increment[:3,3] = -R_pnp.T @ t_pnp.flatten()
        trajectory.append(T_increment[:3,3].copy())
    
    tx = [pos[0] for pos in slam.trajectory]
    tz = [pos[2] for pos in slam.trajectory]
    plt.plot(tx, tz, 'b-') # Trajectory
    plt.scatter(lx, lz, c='g', s=5)
    plt.xlabel('X'); plt.ylabel('Z')
    # plt.xlim(-300,600)
    # plt.ylim(-600,600)



    plt.show()
    # cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    
    