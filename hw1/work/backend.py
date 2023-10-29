import os

import numpy as np

import cv2

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

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
    board_corner = (11, 8)
    fs_path = "../Dataset_CvDl_Hw1/Q2_Image/Q2_lib/alphabet_lib_onboard.txt"
    imageloader = None
    intr = None
    rvecs = []
    tvecs = []
    dist = None
    projection = []
    def __init__(self, imageloader):
        self.imageloader = imageloader
    
    def calcProjection(self):
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

        self.intr = mtx
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.dist = dist

    def arBoard(self, idx, string):
        if len(string) > 6:
            raise Exception("string too long")
        if len(string) == 0:
            raise Exception("empty string")

        self.calcProjection()
        fs = cv2.FileStorage(self.fs_path, cv2.FILE_STORAGE_READ)
        
        basecoord = np.array([[7, 5, 0], [4, 5, 0], [1, 5, 0],
                            [7, 2, 0], [4, 2, 0], [1, 2, 0]])
        
        linestart = []
        lineend = []
        for i in range(len(string)):
            ch = fs.getNode(string[i]).mat()
            for j in range(len(ch)):
                linestart.append(np.add(ch[j][0], basecoord[i]))
                lineend.append(np.add(ch[j][1], basecoord[i]))
        
        linestart = np.array(linestart, np.float32)
        lineend = np.array(lineend, np.float32)

        pointstart, _ = cv2.projectPoints(linestart, self.rvecs[idx],
                self.tvecs[idx], self.intr, self.dist)
        pointend, _ = cv2.projectPoints(lineend, self.rvecs[idx],
                self.tvecs[idx], self.intr, self.dist)

        img = self.imageloader[idx]
        for i in range(len(linestart)):
            start = pointstart[i][0].astype(int)
            end = pointend[i][0].astype(int)
            cv2.line(img, start, end, (0, 0, 255), 5)

        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def arVertical(self, idx, string):
        self.calcProjection()

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
