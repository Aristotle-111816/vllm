# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
from abc import abstractmethod
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any
import threading
import queue
import time
import contextlib

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm import envs

from .base import MediaIO
from .image import ImageMediaIO


def resize_video(frames: npt.NDArray, size: tuple[int, int]) -> npt.NDArray:
    num_frames, _, _, channels = frames.shape
    new_height, new_width = size
    resized_frames = np.empty((num_frames, new_height, new_width, channels),
                              dtype=frames.dtype)
    # lazy import cv2 to avoid bothering users who only use text models
    import cv2

    for i, frame in enumerate(frames):
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_frames[i] = resized_frame
    return resized_frames


def rescale_video_size(frames: npt.NDArray, size_factor: float) -> npt.NDArray:
    _, height, width, _ = frames.shape
    new_height = int(height * size_factor)
    new_width = int(width * size_factor)

    return resize_video(frames, (new_height, new_width))


def sample_frames_from_video(frames: npt.NDArray,
                             num_frames: int) -> npt.NDArray:
    total_frames = frames.shape[0]
    if num_frames == -1:
        return frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames


class VideoLoader:

    @classmethod
    @abstractmethod
    def load_bytes(cls,
                   data: bytes,
                   num_frames: int = -1,
                   **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        raise NotImplementedError


class VideoLoaderRegistry:

    def __init__(self) -> None:
        self.name2class: dict[str, type] = {}

    def register(self, name: str):

        def wrap(cls_to_register):
            self.name2class[name] = cls_to_register
            return cls_to_register

        return wrap

    @staticmethod
    def load(cls_name: str) -> VideoLoader:
        cls = VIDEO_LOADER_REGISTRY.name2class.get(cls_name)
        assert cls is not None, f"VideoLoader class {cls_name} not found"
        return cls()


VIDEO_LOADER_REGISTRY = VideoLoaderRegistry()


@VIDEO_LOADER_REGISTRY.register("opencv")
class OpenCVVideoBackend(VideoLoader):

    def get_cv2_video_api(self):
        import cv2.videoio_registry as vr

        api_pref = None
        for backend in vr.getStreamBufferedBackends():
            if not vr.hasBackend(backend):
                continue
            if not vr.isBackendBuiltIn(backend):
                _, abi, api = vr.getStreamBufferedBackendPluginVersion(backend)
                if abi < 1 or (abi == 1 and api < 2):
                    continue
            api_pref = backend
            break
        return api_pref

    @classmethod
    def load_bytes(cls,
                   data: bytes,
                   num_frames: int = -1,
                   **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        import cv2

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames_num / original_fps if original_fps > 0 else 0

        full_read = num_frames == -1 or total_frames_num < num_frames
        if full_read:
            num_frames = total_frames_num
            frame_idx = list(range(0, num_frames))
        else:
            uniform_sampled_frames = np.linspace(0,
                                                 total_frames_num - 1,
                                                 num_frames,
                                                 dtype=int)
            frame_idx = uniform_sampled_frames.tolist()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((len(frame_idx), height, width, 3), dtype=np.uint8)

        i = 0
        for idx in range(total_frames_num):
            ok = cap.grab()
            if not ok:
                break
            if idx in frame_idx:
                ret, frame = cap.retrieve()
                if ret:
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    i += 1

        assert i == num_frames, (f"Expected reading {num_frames} frames, "
                                 f"but only loaded {i} frames from video.")

        # Use transformers transformers.video_utils.VideoMetadata format
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": duration,
            "video_backend": "opencv"
        }

        return frames, metadata


class VideoMediaIO(MediaIO[npt.NDArray]):

    def __init__(
        self,
        image_io: ImageMediaIO,
        num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality.
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs
        video_loader_backend = envs.VLLM_VIDEO_LOADER_BACKEND
        self.video_loader = VIDEO_LOADER_REGISTRY.load(video_loader_backend)

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, dict[str, Any]]:
        return self.video_loader.load_bytes(data,
                                            num_frames=self.num_frames,
                                            **self.kwargs)

    def load_base64(self, media_type: str,
                    data: str) -> tuple[npt.NDArray, dict[str, Any]]:
        if media_type.lower() == "video/jpeg":
            load_frame = partial(
                self.image_io.load_base64,
                "image/jpeg",
            )

            return np.stack([
                np.asarray(load_frame(frame_data))
                for frame_data in data.split(",")
            ]), {}

        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]:
        with filepath.open("rb") as f:
            data = f.read()

        return self.load_bytes(data)

    def encode_base64(
        self,
        media: npt.NDArray,
        *,
        video_format: str = "JPEG",
    ) -> str:
        video = media

        if video_format == "JPEG":
            encode_frame = partial(
                self.image_io.encode_base64,
                image_format=video_format,
            )

            return ",".join(
                encode_frame(Image.fromarray(frame)) for frame in video)

        msg = "Only JPEG format is supported for now."
        raise NotImplementedError(msg)


class URLFrameStream(threading.Thread):

    def __init__(self,
                 out_q: "queue.Queue[tuple[int, npt.NDArray]]",
                 fps: int,
                 video_url: str,
                 frame_size: tuple[int, int] = (320, 320)) -> None:
        super().__init__(daemon=True)
        self.q = out_q
        self.fps = max(1, int(fps))
        self.video_url = video_url
        self.frame_size = frame_size
        self._stop = threading.Event()
        self._idx = 0

    def run(self) -> None:
        # Lazy import cv2 for users who don't need video
        import cv2

        cap = cv2.VideoCapture(self.video_url)
        if not cap.isOpened():
            # Best effort: print error and exit thread
            print("[ERROR] Cannot open video stream:", self.video_url)
            return

        with contextlib.suppress(Exception):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        interval = 1.0 / float(self.fps)
        next_t = time.perf_counter()

        while not self._stop.is_set():
            # try to keep freshest frame
            for _ in range(5):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            w, h = self.frame_size
            if w > 0 and h > 0:
                frame = cv2.resize(frame, (w, h))

            if self.q.full():
                with contextlib.suppress(queue.Empty):
                    self.q.get_nowait()
            self.q.put((self._idx, frame))
            self._idx += 1

            next_t += interval
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)

        cap.release()

    def stop(self) -> None:
        self._stop.set()


def start_url_stream(video_url: str,
                     fps: int = 1,
                     width: int = 320,
                     height: int = 320,
                     queue_size: int = 1) -> tuple[URLFrameStream, "queue.Queue[tuple[int, npt.NDArray]]"]:
    """
    Create and start a URL frame streaming thread.

    Returns (thread, queue). Queue items are (frame_index, BGR ndarray).
    """
    q: "queue.Queue[tuple[int, npt.NDArray]]" = queue.Queue(maxsize=queue_size)
    t = URLFrameStream(q, fps, video_url, (width, height))
    t.start()
    return t, q
