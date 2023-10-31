import os

import time
import threading

from multimethod import multimethod
from typing import Any, Union, Callable, Iterable

import cv2

class ImageLoader():
    files = None
    lastupdated = -1
    __idx = -1

    @multimethod
    def __init__(self):
        pass

    @multimethod
    def __len__(self):
        if self.files is None:
            return 0
        return len(self.files)

    @multimethod
    def __getitem__(self, index:int):
        maxsize = len(self.files)
        if index < 0 or index >= maxsize:
            raise IndexError
        return cv2.imread(self.files[index])

    @multimethod
    def __iter__(self):
        self.__idx = -1
        return self

    @multimethod
    def __next__(self):
        self.__idx += 1
        if self.__idx < len(self.files):
            return cv2.imread(self.files[self.__idx])
        else:
            raise StopIteration

    @multimethod
    def setPath(self, folder:str):
        files = []
        for file in os.listdir(folder):
            fullpath = os.path.join(folder, file)
            if os.path.isfile(fullpath):
                files.append(fullpath)
        self.files = files
        lastupdated = time.time()
       
class ImageWrapper():
    file = None
    lastupdated = -1

    @multimethod
    def __init__(self):
        pass
    
    @multimethod
    def setPath(self, file: str):
        self.file = file
        lastupdated = time.time()
   
    @multimethod
    def read(self):
        return cv2.imread(self.file)

class SetInterval():
    interval = None
    action = None
    stopEvent = None

    @multimethod
    def __init__(self, interval:Union[float, int], action:Callable):
        self.interval = interval
        self.action = action
        self.stopEvent = threading.Event()
        thread = threading.Thread(target=self.__setInterval)
        thread.start()
    
    @multimethod
    def __setInterval(self):
        nextTime = time.time() + self.interval
        while not self.stopEvent.wait(nextTime - time.time()):
            nextTime += self.interval
            self.action()
    
    @multimethod
    def cancel(self):
        self.stopEvent.set()
