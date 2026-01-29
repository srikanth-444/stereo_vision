import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2
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

def bundel_adjustment( land_marks, camera_poses, cam):
        X_3d=[]
        for pose in camera_poses:
            X_3d.extend(pose['rvec'].flatten().tolist())
            X_3d.extend(pose['tvec'].flatten().tolist())
        for lm in land_marks:
            X_3d.extend(lm.position)
        lm_indices = {lm.id: idx for idx, lm in enumerate(land_marks)}
        frameid_to_camidx = {pose["frame_id"]: i for i, pose in enumerate(camera_poses)}
        X_3d = np.array(X_3d, dtype=np.float64)
        n_cams = len(camera_poses)
        n_lms = len(land_marks)
        # print(X_3d.shape, len(self.camera_poses), len(self.land_marks))
        total_observations = sum([len(lm.observations) for lm in land_marks])
        A = lil_matrix((total_observations*2, X_3d.shape[0]), dtype=np.float64)
        i=0
        for lm in land_marks:
            for obs in lm.observations:
                frame_id = obs['frame_id'] 
                for j in range(6):
                    A[i, (frameid_to_camidx[frame_id])*6 + j] = 1
                    A[i+1, (frameid_to_camidx[frame_id])*6 + j] = 1
                for k in range(3):
                    # print('frame_id:', frame_id, 'lm.id:', lm.id)
                    A[i, (len(camera_poses))*6 + lm_indices[lm.id]*3 + k] = 1
                    A[i+1, (len(camera_poses))*6 + lm_indices[lm.id]*3 + k] = 1
                i+=2
        def residuals(X):
            cam_params = X[:n_cams*6].reshape(n_cams, 6)
            lm_params  = X[n_cams*6:].reshape(n_lms, 3)

            res = []
            # print(cam_params.shape, lm_params.shape)
            for lm in land_marks:
                Pw = lm_params[lm_indices[lm.id]].reshape(1, 3)

                for obs in lm.observations:
                    cam_idx = obs['frame_id']
                    # print(frameid_to_camidx[cam_idx], obs['frame_id'], len(self.camera_poses))
                    rvec = cam_params[frameid_to_camidx[cam_idx], :3]
                    tvec = cam_params[frameid_to_camidx[cam_idx], 3:].reshape(3,1)

                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec

                    Rr = cam.R @ R
                    tr = cam.R @ t + cam.t
                    rvec_r, _ = cv2.Rodrigues(Rr)

                    proj, _ = cv2.projectPoints(
                        Pw, rvec, tvec,
                        cam.l_K, cam.l_dist
                    )
                    proj = proj.ravel()
                    proj_r, _ = cv2.projectPoints(
                        Pw, rvec_r, tr,
                        cam.r_K, cam.r_dist
                    )
                    proj_r = proj_r.ravel()
                    u, v = obs['image_point']
                    u_r, v_r = obs['r_image_point']
                    a=(proj[0]-u)+(proj_r[0]-u_r)
                    b=(proj[1]-v)+(proj_r[1]-v_r)

                    res.extend([a, b])

            return np.array(res)

        result = least_squares(residuals, X_3d, jac_sparsity=A, verbose=0, x_scale='jac', ftol=0.1, method='trf')
        opt = result.x

        # --- Camera poses ---
        opt_cam_params = opt[:n_cams * 6].reshape(n_cams, 6)

        # --- Landmarks ---
        opt_lm_params = opt[n_cams * 6:].reshape(n_lms, 3)

        for cam_idx, pose in enumerate(camera_poses):
            pose["rvec"] = opt_cam_params[cam_idx, :3].copy()
            pose["tvec"] = opt_cam_params[cam_idx, 3:].reshape(3, 1).copy()

        for lm in land_marks:
            lm_idx = lm_indices[lm.id]
            lm.position = opt_lm_params[lm_idx].copy()   