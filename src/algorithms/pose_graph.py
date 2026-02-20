import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# SE(3) utilities
# ------------------------------------------------------------

def vec_to_T(x):
    """6D vector -> SE(3)"""
    t = x[:3]
    w = x[3:]
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(w).as_matrix()
    T[:3, 3] = t
    return T

def T_to_vec(T):
    """SE(3) -> 6D vector"""
    t = T[:3, 3]
    w = R.from_matrix(T[:3, :3]).as_rotvec()
    return np.hstack([t, w])

def log_se3(T):
    """SE(3) -> se(3)"""
    return T_to_vec(T)

# ------------------------------------------------------------
# Pose graph residual
# ------------------------------------------------------------

def pose_graph_residual(
    x,
    T_hat,
    Z,
    w_prior=1.0,
    w_smooth=1.0,
    w_loop=100.0,
    w_gauge=1000.0
):
    N = len(T_hat)
    X = x.reshape(N, 6)
    T = [vec_to_T(X[i]) for i in range(N)]

    res = []

    # # ---- Unary priors (use ALL poses) ----
    # for i in range(N):
    #     e = log_se3(np.linalg.inv(T_hat[i]) @ T[i])
    #     res.append(w_prior * e)

    # ---- Smoothness (pseudo-odometry) ----
    for i in range(N - 1):
        Tij = np.linalg.inv(T[i]) @ T[i + 1]
        e = log_se3(np.linalg.inv(Z[i]) @ Tij)
        res.append(w_smooth * e)

    # ---- Loop closure (start = end) ----
    for i in range(N - 480):
        e_loop = log_se3(np.linalg.inv(T[0]) @ T[i + 480])
        res.append(w_loop * e_loop)

    # ---- Gauge fix (fix first pose) ----
    e_gauge = log_se3(np.linalg.inv(T_hat[0]) @ T[0])
    res.append(w_gauge * e_gauge)

    return np.concatenate(res)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    # --------------------------------------------------------
    # Load your per-frame poses
    # camera_poses[i]['T_global'] must be 4x4 SE(3)
    # --------------------------------------------------------
    print("🔄 Loading camera poses from camera_poses.npy...")
    camera_poses = np.load("camera_poses.npy", allow_pickle=True)

    T_hat = [pose["T_global"] for pose in camera_poses]
    N = len(T_hat)

    # --------------------------------------------------------
    # Build pseudo-odometry (relative poses)
    # Z[i] = T_i^{-1} T_{i+1}
    # --------------------------------------------------------
    print("🔄 Building pseudo-odometry (relative poses)...")
    Z = []
    for i in range(N - 1):
        Ti = T_hat[i]
        Tj = T_hat[i + 1]
        Z.append(np.linalg.inv(Ti) @ Tj)

    # --------------------------------------------------------
    # Initial state vector
    # --------------------------------------------------------
    x0 = np.array([T_to_vec(T) for T in T_hat]).reshape(-1)

    # --------------------------------------------------------
    # Optimize
    # --------------------------------------------------------
    result = least_squares(
        pose_graph_residual,
        x0,
        args=(T_hat, Z),
        loss="huber",
        verbose=2
    )

    # --------------------------------------------------------
    # Extract optimized poses
    # --------------------------------------------------------
    X_opt = result.x.reshape(N, 6)
    T_opt = [vec_to_T(X_opt[i]) for i in range(N)]

    # --------------------------------------------------------
    # Save optimized trajectory
    # --------------------------------------------------------
    np.save("optimized_poses.npy", T_opt)
    print("✅ Optimized trajectory saved to optimized_poses.npy")

    orig = np.array([T[:3, 3] for T in T_hat])
    opt  = np.array([T[:3, 3] for T in T_opt])

    plt.figure()
    plt.plot(orig[:, 0], orig[:, 2], 'r--', label='Original')
    plt.plot(opt[:, 0],  opt[:, 2],  'g-',  label='Optimized')
    plt.axis('equal')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Trajectory comparison')
    plt.show()

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
