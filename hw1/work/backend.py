import os

import numpy as np

import time
import threading

import cv2


os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

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

class Assign1():
    board_corner = (11, 8)
    imageloader = None
    __ctr = 0
    def __init__(self, imageloader):
        self.imageloader = imageloader
    
    def loopthrough(self, selffunc):
        img = selffunc(self.__ctr)
        self.__ctr = (self.__ctr + 1) % len(self.imageloader)
        return img

    def findCorner(self, idx):
        img = self.imageloader[idx]
        findret, corners = cv2.findChessboardCorners(
                img, self.board_corner, None, None)
        cv2.drawChessboardCorners(
                img, self.board_corner, corners, findret)
        
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def findIntrinsic(self):
        imgshape = self.imageloader[0].shape
        cx = self.board_corner[0]
        cy = self.board_corner[1]

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        objpoint = np.zeros((cx * cy, 3), np.float32)
        objpoint[:,:2] = np.mgrid[0:cx,0:cy].T.reshape(-1,2)
        imgpoint = None

        objpoints = []
        imgpoints = []
        
        for img in self.imageloader:
            findret, corners = cv2.findChessboardCorners(
                img, self.board_corner, None, None)
            if findret:
                imgpoints.append(corners)
                objpoints.append(objpoint)
        
        cammatr = np.identity(3)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, imgshape[0:2], cammatr, None)

        return mtx

    def findExtrinsic(self, idx):
        imgshape = self.imageloader[0].shape
        cx = self.board_corner[0]
        cy = self.board_corner[1]

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        objpoint = np.zeros((cx * cy, 3), np.float32)
        objpoint[:,:2] = np.mgrid[0:cx,0:cy].T.reshape(-1,2)
        imgpoint = None

        objpoints = []
        imgpoints = []
        
        for img in self.imageloader:
            findret, corners = cv2.findChessboardCorners(
                img, self.board_corner, None, None)
            if findret:
                imgpoints.append(corners)
                objpoints.append(objpoint)
        
        cammatr = np.identity(3)
        rvecs = np.zeros(3)
        tvecs = np.zeros(3)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, imgshape[0:2], cammatr, None)
        
        rvec = cv2.Rodrigues(rvecs[idx])
        tvec = tvecs[idx]

        return np.concatenate((rvec[0], tvec), axis=1)

    def findDistortion(self):
        imgshape = self.imageloader[0].shape
        cx = self.board_corner[0]
        cy = self.board_corner[1]

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        objpoint = np.zeros((cx * cy, 3), np.float32)
        objpoint[:,:2] = np.mgrid[0:cx,0:cy].T.reshape(-1,2)
        imgpoint = None

        objpoints = []
        imgpoints = []
        
        for img in self.imageloader:
            findret, corners = cv2.findChessboardCorners(
                img, self.board_corner, None, None)
            if findret:
                imgpoints.append(corners)
                objpoints.append(objpoint)
        
        cammatr = np.identity(3)
        rvecs = np.zeros(3)
        tvecs = np.zeros(3)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, imgshape[0:2], cammatr, None)
        
        return dist

    def showUndistorted(self, idx):
        imgshape = self.imageloader[0].shape
        cx = self.board_corner[0]
        cy = self.board_corner[1]

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        objpoint = np.zeros((cx * cy, 3), np.float32)
        objpoint[:,:2] = np.mgrid[0:cx,0:cy].T.reshape(-1,2)
        imgpoint = None

        objpoints = []
        imgpoints = []
        
        for img in self.imageloader:
            findret, corners = cv2.findChessboardCorners(
                img, self.board_corner, None, None)
            if findret:
                imgpoints.append(corners)
                objpoints.append(objpoint)
        
        cammatr = np.identity(3)
        rvecs = np.zeros(3)
        tvecs = np.zeros(3)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, imgshape[0:2], cammatr, None)
        
        img = self.imageloader[idx]
        
        dst = cv2.undistort(img, mtx, dist, None, None)
        
        dst = cv2.resize(dst, (512, 512))
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        
        return dst

class Assign2():

    def arBoard():
        pass

    def arVertical():
        pass

class Assign3():

    def disparityMap():
        pass

    def disparityValue():
        pass

class Assign4():

    def siftKeypoint():
        pass
    
    def siftMatch():
        pass

class Assign5():

    def showAugment():
        pass

    def showModel():
        pass

    def showAccuracyLoss():
        pass
    
    def predictLabel():
        pass
