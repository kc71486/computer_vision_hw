import time
import threading

import numpy as np

import ffmpeg

from typing import Callable, Sequence, Optional, Any

import cv2


class CircularSequenceIter:
    seq: Sequence[Any]
    __idx: int

    def __init__(self, seq: Sequence) -> None:
        if len(seq) == 0:
            raise IndexError("seq should > 0")
        self.seq = seq
        self.__idx = 0

    def next(self) -> Any:
        cur = self.seq[self.__idx]
        self.__idx = (self.__idx + 1) % len(self.seq)
        return cur


class ImageWrapper:
    path: str
    valid: bool
    img: Optional[np.ndarray]

    def __init__(self, path: str = "") -> None:
        self.path = path
        self.valid = False
        self.img = None

    def set_path(self, path: str) -> None:
        self.path = path
        self.valid = False

    def __read(self) -> None:
        if not self.valid:
            self.img = cv2.imread(self.path)
            self.valid = True

    def read(self) -> np.ndarray:
        self.__read()
        return self.img


class VideoWrapper:
    path: str
    valid: bool
    frames: Optional[np.ndarray]
    fps: float

    def __init__(self, path: str = "") -> None:
        self.path = path
        self.valid = False
        self.img = None
        self.frames = None
        self.fps = 0

    def set_path(self, path: str) -> None:
        self.path = path
        self.valid = False

    def __read(self) -> None:
        if not self.valid:
            probe = ffmpeg.probe(self.path)
            video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
            self.fps = int(video_info["r_frame_rate"].split("/")[0]) / int(video_info["r_frame_rate"].split("/")[1])
            width = int(video_info["width"])
            height = int(video_info["height"])
            out, err = (ffmpeg.input(self.path)
                        .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="quiet")
                        .run(capture_stdout=True))
            self.frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            self.valid = True

    def get_fps(self) -> float:
        if not self.valid:
            self.__read()
        return self.fps

    def read(self) -> np.ndarray:
        self.__read()
        return self.frames


class Interval:
    interval: float
    action: Callable
    stopEvent: threading.Event

    def __init__(self, interval: float | int, action: Callable) -> None:
        self.interval = interval
        self.action = action
        self.stopEvent = threading.Event()

    def __interval(self) -> None:
        next_time = time.time() + self.interval
        while not self.stopEvent.wait(next_time - time.time()):
            next_time += self.interval
            self.action()

    def __interval_imm(self) -> None:
        next_time = time.time()
        while not self.stopEvent.wait(next_time - time.time()):
            next_time += self.interval
            self.action()

    def start(self) -> None:
        thread = threading.Thread(target=self.__interval)
        thread.start()

    def start_instant(self) -> None:
        thread = threading.Thread(target=self.__interval_imm)
        thread.start()

    def cancel(self) -> None:
        self.stopEvent.set()
