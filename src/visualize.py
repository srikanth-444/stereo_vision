import cv2
import numpy as np
import open3d as o3d
import time
from scipy.spatial.transform import Rotation as R


class Visualize:
    def __init__(self, atlas) -> None:
        self.atlas = atlas
        self.prev_time = time.perf_counter()
        self.fps = 0.0
        self.fps_alpha = 0.1   # smoothing factor

        # Open3D window
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="Map + Trajectory",
            width=752, height=480,
            left=50, top=50
        )

        # Pre-allocated point storage
        self.max_points = 10000000
        self.points = np.zeros((self.max_points, 3))
        self.colors = np.zeros((self.max_points, 3))
        self.ptr = 0

        # Geometries
        self.pcd_landmarks = o3d.geometry.PointCloud()
        self.active_landmarks = o3d.geometry.PointCloud()
        self.traj_line_set = o3d.geometry.LineSet()
        self.frustum = o3d.geometry.LineSet()

        self.vis.add_geometry(self.pcd_landmarks)
        self.vis.add_geometry(self.active_landmarks)
        self.vis.add_geometry(self.traj_line_set)
        self.vis.add_geometry(self.frustum)

        # Trajectory data
        self.traj_points = []
        self.traj_lines = []

        # Coordinate system transform for Open3D visualization
        self.R_transform = np.array([[0, 1, 0],
                                     [1, 0, 0],
                                     [0, 0, -1]])
        self.first_view = True

        # Render options
        render_option = self.vis.get_render_option()
        render_option.point_size = 2.0


    # ---------------- FPS & Frame Overlay ----------------
    def visualize_pipeline(self, frame) -> None:
        now = time.perf_counter()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            self.fps = (1 - self.fps_alpha) * self.fps + self.fps_alpha * inst_fps

        display_img = frame.image.copy()
        if len(display_img.shape) == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

        # Draw tracked points
        for pt in frame.getTrackedPoints():
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(display_img, (x, y), 2, (0, 255, 0), -1)

        # Draw FPS
        cv2.putText(
            display_img,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow('Frame', display_img)
        cv2.waitKey(0)

    # ---------------- Add new points to point cloud ----------------
    def add_points(self, new_pts, new_colors):
        n = new_pts.shape[0]
        if self.ptr + n >= self.max_points:
            raise ValueError("Max points exceeded!")
        new_pts = new_pts @ self.R_transform.T
        self.points[self.ptr:self.ptr+n] = new_pts
        self.colors[self.ptr:self.ptr+n] = new_colors
        self.ptr += n

    # ---------------- Update point clouds and trajectory ----------------
    def visualize_as_point_cloud(self, T=None):
        # --- Landmarks from aged frames ---
        agedFrame = self.atlas.getActiveMap().getAgedFrame(2)
        if agedFrame is not None:
            aged_points = np.array([lm.point3D for lm in agedFrame.getLandmarks()])
            aged_colors = np.zeros((aged_points.shape[0], 3))  # black
            self.add_points(aged_points, aged_colors)

        # Update global landmarks
        if self.ptr > 0:
            self.pcd_landmarks.points = o3d.utility.Vector3dVector(self.points[:self.ptr])
            self.pcd_landmarks.colors = o3d.utility.Vector3dVector(self.colors[:self.ptr])
        

        # Update active landmarks (red)
        active_points = np.array([lm.point3D for lm in self.atlas.getActiveMap()
                                  .getLastKeyFrame().getLandmarks()])
        active_colors = np.tile([1.0, 0.0, 0.0], (active_points.shape[0], 1))

        active_points = active_points @ self.R_transform.T
        self.active_landmarks.points = o3d.utility.Vector3dVector(active_points)
        self.active_landmarks.colors = o3d.utility.Vector3dVector(active_colors)

        # Trajectory
        if T is not None:
            cam_pos = np.asarray(T[1]).reshape(3,)@self.R_transform.T
            self.traj_points.append(cam_pos)
            if len(self.traj_points) > 1:
                self.traj_lines.append([len(self.traj_points)-2, len(self.traj_points)-1])
            self.traj_line_set.points = o3d.utility.Vector3dVector(np.array(self.traj_points))
            self.traj_line_set.lines = o3d.utility.Vector2iVector(self.traj_lines)
            self.traj_line_set.colors = o3d.utility.Vector3dVector([[0,0,1] for _ in self.traj_lines])

        # Frustum
        if T is not None:
            self.update_camera_frustum(T)

        # First-time view reset
        if self.first_view:
            self.vis.reset_view_point(True)
            self.first_view = False

        # Poll events & render
        self.vis.update_geometry(self.pcd_landmarks)
        self.vis.update_geometry(self.active_landmarks)
        self.vis.update_geometry(self.traj_line_set)
        self.vis.update_geometry(self.frustum)
        self.vis.poll_events()
        self.vis.update_renderer()

        # Follow camera
        if T is not None:
            self.follow_camera(T)

    # ---------------- Camera frustum ----------------
    def update_camera_frustum(self, T, scale=0.2):
        R_mat = self.quat_to_rotmat(T[0])@self.R_transform.T
        t=np.asarray(T[1]).reshape(3,)@self.R_transform.T
        # Define frustum in camera frame
        points = np.array([
            [0,0,0],
            [0.5,0.5,0.5],
            [0.5,-0.5,0.5],
            [-0.5,-0.5,0.5],
            [-0.5,0.5,0.5]
        ], dtype=np.float64) * scale

        points = (points @ R_mat.T) + t

        lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
        self.frustum.points = o3d.utility.Vector3dVector(points)
        self.frustum.lines = o3d.utility.Vector2iVector(lines)
        self.frustum.colors = o3d.utility.Vector3dVector([[0,0,1] for _ in lines])

    # ---------------- Follow camera view ----------------
    def follow_camera(self, T):
        view_ctl = self.vis.get_view_control()
        R_mat = self.quat_to_rotmat(T[0])@self.R_transform.T
        t=np.asarray(T[1]).reshape(3,)@self.R_transform.T

        cam_pos = np.asarray(t).reshape(3,)

        # Camera axes
        forward = -R_mat[:, 2]   # camera looks along -Z
        up      = -R_mat[:, 0]   # camera Y axis
        lookat = cam_pos + forward*1.0
        view_ctl.set_lookat(lookat)
        view_ctl.set_front(forward)
        view_ctl.set_up(up)
        view_ctl.set_zoom(0.1)
    def quat_to_rotmat(self,q):
        """
        q: [w, x, y, z]
        returns 3x3 rotation matrix
        """
        r = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w]
        return r.as_matrix()