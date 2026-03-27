"""
Flask + SocketIO server for the PerceptionLab remote GUI.

The server exposes:
  - GET /          : dashboard HTML
  - SocketIO events: frame_update, pointcloud_update, stats_update, log_message
  - SocketIO commands from client: start_tracking, stop_tracking, reset_map
"""

import base64
import logging
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

logger = logging.getLogger("Server")

# Module-level SocketIO instance so it can be imported elsewhere.
socketio = SocketIO()

# Simple in-memory log buffer for the dashboard log panel.
_log_buffer: list[dict] = []
_log_lock = threading.Lock()
_MAX_LOGS = 200


class _SocketIOLogHandler(logging.Handler):
    """Forwards Python log records to the SocketIO log channel."""

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "level": record.levelname,
            "name": record.name,
            "message": self.format(record),
            "timestamp": time.strftime("%H:%M:%S", time.localtime(record.created)),
        }
        with _log_lock:
            _log_buffer.append(entry)
            if len(_log_buffer) > _MAX_LOGS:
                _log_buffer.pop(0)
        # emit to all connected clients
        socketio.emit("log_message", entry)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates")
    app.config["SECRET_KEY"] = "perceptionlab-secret"

    socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

    # Attach SocketIO log handler to the root logger so all pipeline logs
    # are forwarded to the dashboard.
    handler = _SocketIOLogHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(message)s",
                          datefmt="%H:%M:%S")
    )
    logging.getLogger().addHandler(handler)

    @app.route("/")
    def index():
        return render_template("index.html")

    @socketio.on("connect")
    def on_connect():
        logger.info("Dashboard client connected")
        # Send recent log history to newly connected client.
        with _log_lock:
            for entry in list(_log_buffer):
                socketio.emit("log_message", entry)

    @socketio.on("disconnect")
    def on_disconnect():
        logger.info("Dashboard client disconnected")

    @socketio.on("command")
    def on_command(data: dict):
        cmd = data.get("action", "")
        logger.info(f"Remote command received: {cmd}")
        # Publish command as a SocketIO event so pipeline listeners can react.
        socketio.emit("pipeline_command", {"action": cmd})

    return app


# ---------------------------------------------------------------------------
# WebVisualizer
# ---------------------------------------------------------------------------

class WebVisualizer:
    """
    Drop-in replacement for :class:`src.visualize.Visualize` that streams
    frame images and 3-D point-cloud data to connected web-browser clients
    via SocketIO instead of displaying local OpenCV / Open3D windows.

    Frame data consumed from the ``frame`` object:

    - ``frame.image``          – raw image as a NumPy array (H×W or H×W×C)
    - ``frame.getTrackedPoints()`` – list of (x, y) 2-D tracked feature points
    - ``frame.getLandmarks()`` – list of Landmark objects; each has ``point3D``
                                 (Eigen/NumPy 3-vector)
    - ``frame.id``             – integer frame identifier
    - ``frame.timeStamp``      – int64 timestamp
    - ``frame.nVisible``       – number of visible landmarks
    """

    def __init__(
        self,
        atlas,
        host: str = "0.0.0.0",
        port: int = 5000,
        jpeg_quality: int = 70,
        max_cloud_points: int = 50_000,
    ) -> None:
        self.atlas = atlas
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.max_cloud_points = max_cloud_points

        self._fps = 0.0
        self._fps_alpha = 0.1
        self._prev_time = time.perf_counter()

        # Accumulated point cloud uses a fixed-size deque to avoid unbounded growth.
        self._cloud_pts: deque[list[float]] = deque(maxlen=max_cloud_points)
        self._traj_pts: list[list[float]] = []

        self._app = create_app()
        self._server_thread = threading.Thread(
            target=self._run_server, daemon=True, name="WebVisualizerServer"
        )
        self._server_thread.start()
        logger.info(f"WebVisualizer server started at http://{host}:{port}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_server(self) -> None:
        socketio.run(self._app, host=self.host, port=self.port, log_output=False)

    def _encode_image(self, img: np.ndarray) -> str:
        """Return a base-64-encoded JPEG data-URI string."""
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            bgr = img
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    def _update_fps(self) -> float:
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            self._fps = (1 - self._fps_alpha) * self._fps + self._fps_alpha / dt
        return self._fps

    # ------------------------------------------------------------------
    # Public API – mirrors Visualize interface
    # ------------------------------------------------------------------

    def visualize_pipeline(self, frame) -> None:
        """
        Encode the frame image and send tracking overlays to the dashboard.

        Reads from *frame*:
        - ``image``             raw camera image
        - ``getTrackedPoints()``2-D feature positions for overlay dots
        - ``id``, ``timeStamp``, ``nVisible`` for the stats panel
        """
        fps = self._update_fps()

        # --- Build annotated image ---
        img = np.array(frame.image)
        if img.ndim == 2:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            display = img.copy()

        tracked = frame.getTrackedPoints()
        for pt in tracked:
            cv2.circle(display, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        cv2.putText(
            display, f"FPS: {fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
        )

        img_data = self._encode_image(display)

        # --- Gather stats ---
        n_tracked = len(tracked)
        try:
            n_kf = self.atlas.getLengthKeyFrame()
        except Exception:
            n_kf = 0

        n_visible = getattr(frame, "nVisible", 0)

        socketio.emit("frame_update", {
            "image": img_data,
            "frame_id": int(frame.id),
            "timestamp": int(frame.timeStamp),
            "fps": round(fps, 1),
            "n_tracked": n_tracked,
            "n_visible": n_visible,
            "n_keyframes": n_kf,
        })

    def visualize_as_point_cloud(self, T: Optional[np.ndarray] = None) -> None:
        """
        Collect landmark 3-D positions from the last keyframe and push them
        to the dashboard's Three.js point-cloud viewer.

        Reads from *frame* (via atlas):
        - ``getLandmarks()`` → ``landmark.point3D`` (x, y, z float)
        """
        # Accumulate aged landmarks (permanent black points).
        try:
            aged = self.atlas.getAgedFrame(2)
            if aged is not None:
                for lm in aged.getLandmarks():
                    p = lm.point3D
                    if len(self._cloud_pts) < self.max_cloud_points:
                        self._cloud_pts.append([float(p[0]), float(p[1]), float(p[2])])
        except Exception:
            pass

        # Active landmarks from the most recent keyframe (red points).
        active_pts: list[list[float]] = []
        try:
            kf = self.atlas.getLastKeyFrame()
            for lm in kf.getLandmarks():
                p = lm.point3D
                active_pts.append([float(p[0]), float(p[1]), float(p[2])])
        except Exception:
            pass

        # Camera trajectory.
        if T is not None:
            pose = T[:3, 3].tolist()
            self._traj_pts.append(pose)

        socketio.emit("pointcloud_update", {
            "landmarks": list(self._cloud_pts),
            "active": active_pts,
            "trajectory": self._traj_pts,
        })
