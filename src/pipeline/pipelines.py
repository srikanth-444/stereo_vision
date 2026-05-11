import numpy as np
import cv2
import time
import queue
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from ..atlas import Frame


class Pipeline:

    def __init__(self, atlas, tracker, depth_estimator, visualizer,
                 camera_map, optimizer, no_workers=4):
        self.atlas           = atlas
        self.tracker         = tracker
        self.depth_estimator = depth_estimator
        self.optimizer       = optimizer
        self.visualizer      = visualizer
        self.camera_map      = camera_map

        self.currentMap      = self.atlas.initiateNewMap()
        self.current_frame   = None
        self.index           = 0
        self.create_keyframe = True
        self.trajectory      = []

        # loggers
        self.performance_logger = logging.getLogger("Performance")
        self.pipeline_logger    = logging.getLogger("Pipeline   ")
        self.mapper_logger      = logging.getLogger("Mapper     ")

        # shared thread pool — used for prefetch, feature extraction, mapping, viz
        self._executor = ThreadPoolExecutor(max_workers=max(no_workers, 4))

        # mapping worker (off main loop)
        self._mapping_queue = queue.Queue(maxsize=2)
        self._mapping_thread = threading.Thread(
            target=self._mapping_worker, daemon=True)
        self._mapping_thread.start()

        # viz worker (off main loop)
        self._viz_queue = queue.Queue(maxsize=1)
        self._viz_thread = threading.Thread(
            target=self._viz_worker, daemon=True)
        self._viz_thread.start()

        # fps tracking
        self._fps       = 0.0
        self._fps_alpha = 0.1
        self._prev_time = time.time()

    # ── internal workers ────────────────────────────────────────────────────

    def _mapping_worker(self):
        while True:
            try:
                item = self._mapping_queue.get()
                if item is None:
                    break
                left_frame, landmarks = item
                self.mapper_logger.debug(
                    f"Processing mapping for frame: {left_frame.id}")
                self.currentMap.mapping(left_frame, landmarks)
            except Exception as e:
                self.mapper_logger.error(f"Error in mapping worker: {e}")
            finally:
                self._mapping_queue.task_done()

    def _viz_worker(self):
        while True:
            item = self._viz_queue.get()
            if item is None:
                break
            left_frame, right_frame = item
            self.visualizer.visualize_pipeline(left_frame, right_frame)

    # ── frame acquisition ───────────────────────────────────────────────────

    def _get_raw_frames(self) -> dict:
        """Read frames from all cameras. Returns dict keyed by camera_id."""
        current_frames = {}
        for camera_id, camera in self.camera_map.items():
            frame = camera.get_frame(self.index)
            if frame is not None:
                current_frames[camera_id] = frame
            self.index += 1
        return current_frames

    def _extract_features(self, frame) -> int:
        """Extract features for a single frame. Returns ms taken."""
        t0 = time.time()
        frame.extractFeatures()
        return int((time.time() - t0) * 1000)

    def _fetch_and_process(self) -> tuple[dict, int]:
        """
        Fetch frames from all cameras then extract features in parallel.
        Returns (frames_dict, feature_extraction_ms).
        Designed to run entirely in a background thread so the main loop
        never waits on I/O or feature extraction.
        """
        frames = self._get_raw_frames()
        if not frames:
            return frames, 0

        t0 = time.time()
        feat_futures = {
            cid: self._executor.submit(self._extract_features, f)
            for cid, f in frames.items()
        }
        for cid, fut in feat_futures.items():
            ms = fut.result()
            self.performance_logger.debug(f"feature_extraction {ms}ms")

        total_ms = int((time.time() - t0) * 1000)
        return frames, total_ms

    # ── helpers ─────────────────────────────────────────────────────────────

    def _update_fps(self):
        now = time.time()
        dt  = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            instant = 1.0 / dt
            self._fps = (1.0 - self._fps_alpha) * self._fps \
                      + self._fps_alpha * instant

    def save_trajectory(self, filepath: str):
        with open(filepath, "w") as f:
            for p in self.trajectory:
                f.write(
                    f"{p['timestamp']:.9f} "
                    f"{p['p_x']:.9f} "
                    f"{p['p_y']:.9f} "
                    f"{p['p_z']:.9f}\n"
                )

    def _shutdown(self):
        """Clean shutdown of all background workers."""
        self._mapping_queue.put(None)
        self._viz_queue.put(None)
        self._executor.shutdown(wait=False)

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self):
        # kick off first fetch+process BEFORE the loop starts
        # so frame 0 is already ready when we enter iteration 1
        future: Future = self._executor.submit(self._fetch_and_process)

        depth_time     = 0
        keyframe_time  = 0
        landmarks_time = 0

        for i in range(1, 100_001):
            self.pipeline_logger.debug(f"current iteration {i}")

            # ── capture (~0ms — frames fetched+processed in background) ────
            t_capture = time.time()
            frames, process_time = future.result()
            capture_time = int((time.time() - t_capture) * 1000)

            if not frames:
                self.save_trajectory("trajectory.txt")
                self.pipeline_logger.critical(
                    "No frames received from source. Shutting down pipeline.")
                self.visualizer.show_full_map()
                self._shutdown()
                break

            left_frame  = frames[0]
            right_frame = frames[1]

            # ── prefetch next iteration immediately ─────────────────────────
            # runs in background while we do tracking / depth / mapping below
            future = self._executor.submit(self._fetch_and_process)

            # ── rectify ─────────────────────────────────────────────────────
            # self.depth_estimator.rectifyPoints(left_frame, right_frame)

            # ── tracking ────────────────────────────────────────────────────
            t_track = time.time()
            self.create_keyframe = self.tracker.track(left_frame,right_frame)
            tracking_time = int((time.time() - t_track) * 1000)

            # ── keyframe path ────────────────────────────────────────────────
            if self.create_keyframe:
                self.create_keyframe = False

                if self.tracker.tracking_state == "lost":
                    self.currentMap = self.atlas.initiateNewMap()
                    self.tracker.map_initialized = False
                    self.tracker.tracking_state  = "good"

                # depth
                t_depth = time.time()
                pts_3d, keypoint_idx = self.depth_estimator.getDepth(
                    left_frame, right_frame, 3)
                depth_time = int((time.time() - t_depth) * 1000)

                # keyframe bookkeeping
                t_kf = time.time()
                self.currentMap.setKeyframe(left_frame)
                self.trajectory.append({
                    "timestamp":   left_frame.timeStamp,
                    "p_x":         left_frame.worldPose[1][0],
                    "p_y":         left_frame.worldPose[1][1],
                    "p_z":         left_frame.worldPose[1][2],
                    "orientation": left_frame.worldPose[0],
                })
                self.pipeline_logger.debug(f"new keyframe id {left_frame.id}")
                keyframe_time = int((time.time() - t_kf) * 1000)

                # landmarks
                t_lm = time.time()
                landmarks = self.currentMap.createLandmarks(
                    pts_3d, left_frame, keypoint_idx)
                landmarks_time = int((time.time() - t_lm) * 1000)

                if self.currentMap.getLengthKeyFrame() > 1:
                    try:
                        self._mapping_queue.put_nowait((left_frame, landmarks))
                    except queue.Full:
                        self.pipeline_logger.debug(
                            f"Mapping queue full — dropping frame {left_frame.id}")

            self.pipeline_logger.debug(
                f"length of keyframes {self.currentMap.getLengthKeyFrame()}")

            # ── gui (non-blocking drop) ──────────────────────────────────────
            t_gui = time.time()
            try:
                self._viz_queue.put_nowait((left_frame, right_frame))
            except queue.Full:
                pass
            gui_time = int((time.time() - t_gui) * 1000)

            # ── fps + perf log ───────────────────────────────────────────────
            self._update_fps()
            self.performance_logger.info(
                f"FPS: {self._fps:.1f} | "
                f"capture {capture_time}ms | "
                f"Process {process_time}ms | "
                f"tracking {tracking_time}ms | "
                f"depth {depth_time}ms | "
                f"keyframe {keyframe_time}ms | "
                f"creat lm {landmarks_time}ms | "
                f"gui {gui_time}ms"
            )


def pipeline_factory(atlas, tracker, depth_estimator,
                     visualizer, camera_map, optimizer):
    return Pipeline(atlas, tracker, depth_estimator,
                    visualizer, camera_map, optimizer)