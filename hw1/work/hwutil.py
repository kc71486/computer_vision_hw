import os

import time
import threading

import cv2

class ImageLoader():
    files = None
    __idx = -1
    def __init__(self):
        pass

    def __len__(self):
        if self.files is None:
            return 0
        return len(self.files)

    def __getitem__(self, index):
        maxsize = len(self.files)
        if index < 0 or index >= maxsize:
            raise IndexError
        return cv2.imread(self.files[index])

    def __iter__(self):
        self.__idx = -1
        return self

    def __next__(self):
        self.__idx += 1
        if self.__idx < len(self.files):
            return cv2.imread(self.files[self.__idx])
        else:
            raise StopIteration

    def setpath(self, folder):
        files = []
        for file in os.listdir(folder):
            fullpath = os.path.join(folder, file)
            if os.path.isfile(fullpath):
                files.append(fullpath)
        self.files = files
        
class SetInterval():
    interval = None
    action = None
    stopEvent = None
    def __init__(self, interval, action):
        self.interval = interval
        self.action = action
        self.stopEvent = threading.Event()
        thread = threading.Thread(target=self.__setInterval)
        thread.start()

    def __setInterval(self):
        nextTime = time.time() + self.interval
        while not self.stopEvent.wait(nextTime - time.time()):
            nextTime += self.interval
            self.action()

    def cancel(self):
        self.stopEvent.set()
