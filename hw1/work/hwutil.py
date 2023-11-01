import os

import time
import threading

import numpy as np

from typing import Callable
from typing_extensions import Self

import cv2


class ImageLoader:
    files: list[str]
    last_updated: float
    __idx: int

    def __init__(self):
        self.files = []
        self.last_updated = -1
        self.__idx = -1

    def __len__(self):
        if self.files is None:
            return 0
        return len(self.files)

    def __getitem__(self, index: int) -> np.ndarray:
        maxsize = len(self.files)
        if index < 0 or index >= maxsize:
            raise IndexError
        return cv2.imread(self.files[index])

    def __iter__(self) -> Self:
        self.__idx = -1
        return self

    def __next__(self) -> np.ndarray:
        self.__idx += 1
        if self.__idx < len(self.files):
            return cv2.imread(self.files[self.__idx])
        else:
            raise StopIteration

    def set_path(self, folder: str) -> None:
        files = []
        for file in os.listdir(folder):
            fullpath = os.path.join(folder, file)
            if os.path.isfile(fullpath):
                files.append(fullpath)
        self.files = files
        self.last_updated = time.time()


class ImageWrapper:
    file: str
    last_updated: float

    def __init__(self) -> None:
        self.file = ""
        self.last_updated = -1

    def set_path(self, file: str) -> None:
        self.file = file
        self.last_updated = time.time()

    def read(self) -> np.ndarray:
        return cv2.imread(self.file)


class SetInterval:
    interval: float
    action: Callable
    stopEvent: threading.Event

    def __init__(self, interval: float | int, action: Callable) -> None:
        self.interval = interval
        self.action = action
        self.stopEvent = threading.Event()
        thread = threading.Thread(target=self.__set_interval)
        thread.start()

    def __set_interval(self) -> None:
        next_time = time.time() + self.interval
        while not self.stopEvent.wait(next_time - time.time()):
            next_time += self.interval
            self.action()

    def cancel(self) -> None:
        self.stopEvent.set()
