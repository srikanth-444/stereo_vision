import csv
import numpy as np
from src.core.slam_service import start_service

class EuRoCGroundTruth:
    def __init__(self, gt_csv_path: str):
        self.poses = []
        self._ts_index = {}
        try:
            with open(gt_csv_path, newline='') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    ts = int(row["#timestamp"])
                    position = np.array([
                        float(row[" p_RS_R_x [m]"]),
                        float(row[" p_RS_R_y [m]"]),
                        float(row[" p_RS_R_z [m]"])
                    ], dtype=np.float64)
                    quaternion = np.array([
                        float(row[" q_RS_w []"]),
                        float(row[" q_RS_x []"]),
                        float(row[" q_RS_y []"]),
                        float(row[" q_RS_z []"])
                    ], dtype=np.float64)
                    self.poses.append((ts, position, quaternion))
                    self._ts_index[ts] = i
        except Exception as e:
            raise ValueError(f"Failed to read ground truth at {gt_csv_path}: {e}")
        print(f"Loaded {len(self.poses)} ground truth poses.")

    def get_all_positions(self):
        return np.array([p[1] for p in self.poses])

    def get_all_timestamps(self):
        return np.array([p[0] for p in self.poses])

    def _align_umeyama(self, est, gt):
        mu_e, mu_g = est.mean(0), gt.mean(0)
        e0, g0 = est - mu_e, gt - mu_g
        cov = g0.T @ e0 / len(est)
        U, s, Vt = np.linalg.svd(cov)
        D = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            D[2, 2] = -1
        R = U @ D @ Vt
        t = mu_g - R @ mu_e
        return (R @ est.T).T + t

    def compute_ATE(self, traj_file: str):
        est = np.loadtxt(traj_file)
        if len(est) == 0:
            print("Trajectory file is empty!")
            return None

        est_t   = est[:, 0] / 1e9   # ns -> seconds
        est_pos = est[:, 1:4]

        gt_t   = self.get_all_timestamps() / 1e9
        gt_pos = self.get_all_positions()

        gt_interp = np.column_stack([
            np.interp(est_t, gt_t, gt_pos[:, i]) for i in range(3)
        ])

        est_aligned = self._align_umeyama(est_pos, gt_interp)
        errors = np.linalg.norm(est_aligned - gt_interp, axis=1)

        print(f"\n=== ATE Results ({len(errors)} keyframes) ===")
        print(f"RMSE : {np.sqrt(np.mean(errors**2)):.4f} m")
        print(f"Mean : {errors.mean():.4f} m")
        print(f"Max  : {errors.max():.4f} m")
        print(f"Min  : {errors.min():.4f} m")
        return errors

if __name__ == "__main__":
    
    GT_CSV = "/home/srikanth/machine_hall/machine_hall/MH_01_easy/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv"
    gt = EuRoCGroundTruth(GT_CSV)
    start_service()
    gt.compute_ATE("trajectory.txt")