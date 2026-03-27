"""
Tests for the Flask + SocketIO server module (src/server).

These tests exercise the server without requiring compiled C++ extensions
(ORBExtractor / Atlas), keeping them fast and self-contained.
"""

import sys
import os
import base64
import importlib
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: add src/server directly to sys.path so we can import it without
# triggering src/__init__.py (which requires compiled C++ extensions).
# ---------------------------------------------------------------------------
_repo_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_server_root = os.path.join(_repo_root, "src")

# Prevent src/__init__.py from executing by pre-registering a stub package.
if "src" not in sys.modules:
    import types
    stub = types.ModuleType("src")
    stub.__path__ = [_server_root]
    stub.__package__ = "src"
    sys.modules["src"] = stub

# Now importing src.server will work without the C++ extension chain.
from src.server.app import create_app, socketio, WebVisualizer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _Landmark:
    """Minimal stub matching the Landmark interface used by WebVisualizer."""
    def __init__(self, x, y, z):
        self.point3D = np.array([x, y, z], dtype=np.float32)


class _Frame:
    """Minimal stub matching the Frame interface used by WebVisualizer."""
    def __init__(self, frame_id=0, h=120, w=160):
        self.id        = frame_id
        self.timeStamp = 1_000_000_000
        self.nVisible  = 5
        rng = np.random.default_rng(42)
        self.image = rng.integers(0, 255, (h, w), dtype=np.uint8)
        self._tracked = [(10.0, 20.0), (30.0, 40.0)]
        self._landmarks = [_Landmark(0.1, 0.2, 0.3), _Landmark(-0.1, 0.5, 1.0)]

    def getTrackedPoints(self):
        return self._tracked

    def getLandmarks(self):
        return self._landmarks


class _Atlas:
    """Minimal stub matching the Atlas/Map interface used by WebVisualizer."""
    def __init__(self):
        self._kf = _Frame()

    def getLengthKeyFrame(self):
        return 1

    def getLastKeyFrame(self):
        return self._kf

    def getAgedFrame(self, age):
        return None


# ---------------------------------------------------------------------------
# Flask app tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_returns_200(test_client):
    """The dashboard root endpoint must return HTTP 200."""
    resp = test_client.get("/")
    assert resp.status_code == 200


def test_index_contains_dashboard_title(test_client):
    """The dashboard page must contain the application title."""
    resp = test_client.get("/")
    assert b"PerceptionLab Dashboard" in resp.data


def test_index_contains_socketio_script(test_client):
    """The dashboard must include a Socket.IO client script tag."""
    resp = test_client.get("/")
    html = resp.data.decode("utf-8")
    # Verify the Socket.IO CDN script tag is present in the page.
    assert html.find("socket.io.min.js") != -1  # noqa: S603


# ---------------------------------------------------------------------------
# WebVisualizer unit tests (no network, no real server thread needed)
# ---------------------------------------------------------------------------

def test_encode_image_grayscale():
    """_encode_image must produce a valid base64 JPEG data-URI from a grey image."""
    atlas = _Atlas()
    vis = WebVisualizer.__new__(WebVisualizer)
    vis.jpeg_quality = 70

    img = np.zeros((120, 160), dtype=np.uint8)
    data_uri = vis._encode_image(img)
    assert data_uri.startswith("data:image/jpeg;base64,")
    b64_part = data_uri[len("data:image/jpeg;base64,"):]
    decoded = base64.b64decode(b64_part)
    assert len(decoded) > 0


def test_encode_image_color():
    """_encode_image must handle 3-channel BGR images."""
    vis = WebVisualizer.__new__(WebVisualizer)
    vis.jpeg_quality = 70

    img = np.zeros((120, 160, 3), dtype=np.uint8)
    data_uri = vis._encode_image(img)
    assert data_uri.startswith("data:image/jpeg;base64,")


def test_fps_update_increases():
    """_update_fps should return a positive value after the first call."""
    import time
    vis = WebVisualizer.__new__(WebVisualizer)
    vis._fps       = 0.0
    vis._fps_alpha = 0.1
    vis._prev_time = time.perf_counter() - 0.1  # pretend 100 ms have passed
    fps = vis._update_fps()
    assert fps > 0


def test_visualize_pipeline_emits_frame_update(test_client):
    """
    visualize_pipeline should call socketio.emit('frame_update', …).
    We patch socketio.emit to capture the call.
    """
    import unittest.mock as mock

    atlas = _Atlas()
    frame = _Frame()

    vis = WebVisualizer.__new__(WebVisualizer)
    vis.atlas        = atlas
    vis.jpeg_quality = 70
    vis._fps         = 0.0
    vis._fps_alpha   = 0.1
    import time
    vis._prev_time   = time.perf_counter() - 0.05

    emitted = {}

    def fake_emit(event, data):
        emitted[event] = data

    with mock.patch.object(socketio, "emit", side_effect=fake_emit):
        vis.visualize_pipeline(frame)

    assert "frame_update" in emitted
    payload = emitted["frame_update"]
    assert payload["frame_id"] == frame.id
    assert payload["n_tracked"] == len(frame.getTrackedPoints())
    assert payload["n_keyframes"] == atlas.getLengthKeyFrame()
    assert payload["image"].startswith("data:image/jpeg;base64,")


def test_visualize_as_point_cloud_emits(test_client):
    """
    visualize_as_point_cloud should call socketio.emit('pointcloud_update', …)
    containing active landmark coordinates.
    """
    import unittest.mock as mock

    atlas = _Atlas()

    vis = WebVisualizer.__new__(WebVisualizer)
    vis.atlas             = atlas
    vis.max_cloud_points  = 50_000
    vis._cloud_pts        = []
    vis._traj_pts         = []

    emitted = {}

    def fake_emit(event, data):
        emitted[event] = data

    with mock.patch.object(socketio, "emit", side_effect=fake_emit):
        vis.visualize_as_point_cloud(T=None)

    assert "pointcloud_update" in emitted
    payload = emitted["pointcloud_update"]
    active = payload["active"]
    lms = atlas.getLastKeyFrame().getLandmarks()
    assert len(active) == len(lms)
    # Check first landmark coords
    assert abs(active[0][0] - lms[0].point3D[0]) < 1e-5
