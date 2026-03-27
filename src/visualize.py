import cv2
import numpy as np
import open3d as o3d
import time

from .server.app import WebVisualizer  # noqa: F401 – re-exported for convenience


class Visualize:
    def __init__(self, atlas) -> None:
        self.atlas=atlas
        self.prev_time = time.perf_counter()
        self.fps = 0.0
        self.fps_alpha = 0.1   # smoothing factor
        self.first_view = True
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Map + Trajectory",
                                width=640,
                                height=480,
                                left=100,
                                top=100)
        self.max_points = 1000000
        self.points = np.zeros((self.max_points, 3))
        self.colors = np.zeros((self.max_points, 3))
        self.ptr = 0
        self.pcd_landmarks = o3d.geometry.PointCloud()
        self.active_landmarks = o3d.geometry.PointCloud()
        self.traj_line_set = o3d.geometry.LineSet()
        self.axes = o3d.geometry.LineSet()

        self.vis.add_geometry(self.pcd_landmarks)
        self.vis.add_geometry(self.traj_line_set)
        self.vis.add_geometry(self.axes)
        self.vis.add_geometry(self.active_landmarks)
        self.landmark_points = []
        self.traj_points=[]
        self.traj_lines=[]
        self.vis.poll_events()
        self.vis.update_renderer()
        self.i=0
        self.R_transform = np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]])
        render_option = self.vis.get_render_option()
        render_option.point_size = 2.0
    
    def add_points(self, new_pts, new_colors):
        n = new_pts.shape[0]
        self.points[self.ptr:self.ptr+n] = new_pts
        self.colors[self.ptr:self.ptr+n] = new_colors
        self.ptr += n
    
    def visualize_pipeline(self, frame)->None:

            # --- FPS computation ---
        now = time.perf_counter()
        dt = now - self.prev_time
        self.prev_time = now

        if dt > 0:
            inst_fps = 1.0 / dt
            self.fps = (1 - self.fps_alpha) * self.fps + self.fps_alpha * inst_fps

        if len(frame.image.shape) == 2:
            display_img = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
        else:
            display_img = frame.image.copy() # Keep original if already color
        image_points=frame.getTrackedPoints()
        # untracked_points=frame.getNotAssociatedPoints()
        # projected_points=frame.projectedPoints

        for pt in image_points:
            x=int(pt[0])
            y=int(pt[1])
            cv2.circle(display_img, (x,y), 2, (0,255,0), -1)
        # for pt in untracked_points:
        #     x=int(pt[0])
        #     y=int(pt[1])
        #     cv2.circle(display_img, (x,y), 2, (0,0,255), -1)
        # for pt in projected_points:
        #     x=int(pt[0])
        #     y=int(pt[1])
        #     cv2.circle(display_img, (x,y), 2, (255,0,0), -1)
            # --- Draw FPS ---
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
        cv2.waitKey(1)

    def visualize_as_point_cloud(self,T=None):
        agedFrame=self.atlas.getAgedFrame(2)
        if agedFrame is not None:
            aged_points = np.array([lm.point3D for lm in agedFrame.getLandmarks()])
            aged_colors = np.zeros((aged_points.shape[0], 3))  # black RGB
            self.add_points(aged_points,aged_colors)
        self.pcd_landmarks.points = o3d.utility.Vector3dVector(self.points[:self.ptr])
        self.pcd_landmarks.colors = o3d.utility.Vector3dVector(self.colors[:self.ptr])
        active_points = np.array([lm.point3D for lm in self.atlas.getLastKeyFrame().getLandmarks() ])
        active_colors = np.tile([1.0, 0.0, 0.0], (active_points.shape[0], 1))  # red RGB
        self.active_landmarks.points = o3d.utility.Vector3dVector(active_points)
        self.active_landmarks.colors=o3d.utility.Vector3dVector(active_colors)
        self.pcd_landmarks.transform(self.R_transform)
        self.active_landmarks.transform(self.R_transform)
        self.vis.update_geometry(self.pcd_landmarks)
        self.vis.update_geometry(self.active_landmarks)

        if T is not None:
            pose = T[:3, 3]
            self.traj_points.append(pose)

            if len(self.traj_points) > 1:
                self.traj_lines.append([len(self.traj_points) - 2, len(self.traj_points) - 1])
            self.traj_line_set.points = o3d.utility.Vector3dVector(np.array(self.traj_points))
            self.traj_line_set.lines = o3d.utility.Vector2iVector(self.traj_lines)
            self.traj_line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in self.traj_lines])
            self.traj_line_set.transform(self.R_transform)

            self.vis.update_geometry(self.traj_line_set)
        
        if self.first_view:
                self.vis.reset_view_point(True)
                self.first_view = False
        # self.i=len(self.landmark_manager.landmarks)-1
        self.vis.poll_events()
        self.vis.update_renderer()
        # self.follow_camera(T)

    def follow_camera(self, T):
        view_ctl = self.vis.get_view_control()
        
        # Camera position in world space
        cam_pos = T[:3, 3]
        
        # Forward vector (Z-axis of the camera matrix)
        forward = T[:3, 2] 
        
        # Up vector (Y-axis of the camera matrix)
        up = -T[:3, 1] 
        
        view_ctl.set_lookat(cam_pos + forward) # Look slightly ahead of camera
        view_ctl.set_front(-forward)           # Direction the camera is facing
        view_ctl.set_up(up)                    # Keep the 'sky' up
        view_ctl.set_zoom(0.05)

    
