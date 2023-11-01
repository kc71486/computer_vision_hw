import os

import numpy as np

from multimethod import multimethod
from typing import Any, Union

import cv2

import hwutil

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


class Assign1:
    board_corner: tuple
    imageloader: hwutil.ImageLoader
    __ctr1: int
    __ctr5: int

    def __init__(self, imageloader: hwutil.ImageLoader):
        self.board_corner = (11, 8)
        self.imageloader = imageloader
        self.__ctr1 = 0
        self.__ctr5 = 0

    def loop1(self) -> np.ndarray:
        img = self.find_corner(self.__ctr1)
        self.__ctr1 = (self.__ctr1 + 1) % len(self.imageloader)
        return img

    def loop5(self) -> np.ndarray:
        img = self.show_undistorted(self.__ctr5)
        self.__ctr5 = (self.__ctr5 + 1) % len(self.imageloader)
        return img

    def find_corner(self, idx: int) -> np.ndarray:
        img = self.imageloader[idx]
        find, corners = cv2.findChessboardCorners(img, self.board_corner)
        if find:
            cv2.drawChessboardCorners(img, self.board_corner, corners, find)

        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def find_intrinsic(self) -> np.ndarray:
        imgshape = self.imageloader[0].shape
        cx = self.board_corner[0]
        cy = self.board_corner[1]

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        objpoint = np.zeros((cx * cy, 3), np.float32)
        objpoint[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2)
        imgpoint = None

        objpoints = []
        imgpoints = []

        for img in self.imageloader:
            findret, corners = cv2.findChessboardCorners(img, self.board_corner)
            if findret:
                imgpoints.append(corners)
                objpoints.append(objpoint)

        cammatr = np.identity(3)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, imgshape[0:2], cammatr, None)

        return mtx

    @multimethod
    def findExtrinsic(self, idx: int) -> np.ndarray:
        imgshape = self.imageloader[0].shape
        cx = self.board_corner[0]
        cy = self.board_corner[1]

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        objpoint = np.zeros((cx * cy, 3), np.float32)
        objpoint[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2)
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

    def find_distortion(self):
        imgshape = self.imageloader[0].shape
        cx = self.board_corner[0]
        cy = self.board_corner[1]

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        objpoint = np.zeros((cx * cy, 3), np.float32)
        objpoint[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2)
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

    def show_undistorted(self, idx: int):
        imgshape = self.imageloader[0].shape
        cx = self.board_corner[0]
        cy = self.board_corner[1]

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        objpoint = np.zeros((cx * cy, 3), np.float32)
        objpoint[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2)
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


class Assign2:
    board_corner: tuple[int, int]
    fs_path: str
    imageloader: hwutil.ImageLoader
    intrinsic: Any
    r_vectors: list
    t_vectors: list
    dist: Any
    projection: list

    @multimethod
    def __init__(self, imageloader: hwutil.ImageLoader):
        self.board_corner = (11, 8)
        self.fs_path = "../Dataset_CvDl_Hw1/Q2_Image/Q2_lib/alphabet_lib_onboard.txt"
        self.imageloader = imageloader
        self.intrinsic = None
        self.r_vectors = []
        self.t_vectors = []
        self.dist = None
        self.projection = []

    @multimethod
    def calcProjection(self):
        imgshape = self.imageloader[0].shape
        cx = self.board_corner[0]
        cy = self.board_corner[1]

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        objpoint = np.zeros((cx * cy, 3), np.float32)
        objpoint[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2)
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

        self.intrinsic = mtx
        self.r_vectors = rvecs
        self.t_vectors = tvecs
        self.dist = dist

    def ar_board(self, idx: int, string: str):
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

        pointstart, _ = cv2.projectPoints(linestart, self.r_vectors[idx],
                                          self.t_vectors[idx], self.intrinsic, self.dist)
        pointend, _ = cv2.projectPoints(lineend, self.r_vectors[idx],
                                        self.t_vectors[idx], self.intrinsic, self.dist)

        img = self.imageloader[idx]
        for i in range(len(linestart)):
            start = pointstart[i][0].astype(int)
            end = pointend[i][0].astype(int)
            cv2.line(img, start, end, (0, 0, 255), 10)

        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def ar_vertical(self, idx: int, string: str):
        self.calcProjection()


class Assign3:
    left_wrapper: hwutil.ImageWrapper
    right_wrapper: hwutil.ImageWrapper

    def __init__(self, leftwrapper: hwutil.ImageWrapper, rightwrapper: hwutil.ImageWrapper):
        self.left_wrapper = leftwrapper
        self.right_wrapper = rightwrapper

    def disparity_value(self, coord: tuple[Union[float, int], Union[float, int]]):
        outleft = self.left_wrapper.read()
        outright = self.right_wrapper.read()
        left = cv2.cvtColor(outleft, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(outright, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM_create()
        disp = stereo.compute(left, right)
        disparity = np.divide(disp.astype(np.float32), 16).astype(int)

        disval = disparity[coord[1]][coord[0]]
        if disval >= 0:
            newcoord = (coord[0] + disval, coord[1])

            cv2.circle(outright, newcoord, radius=5,
                       color=(0, 0, 255), thickness=-1)

        outleft = cv2.resize(outleft, (512, 512))
        outleft = cv2.cvtColor(outleft, cv2.COLOR_BGR2RGB)

        outright = cv2.resize(outright, (512, 512))
        outright = cv2.cvtColor(outright, cv2.COLOR_BGR2RGB)

        disp = np.maximum(disp, 0)
        disp = cv2.resize(disp, (512, 512))
        disp = cv2.merge((disp, disp, disp))

        return outleft, outright, disp


class Assign4:
    left_wrapper: hwutil.ImageWrapper
    right_wrapper: hwutil.ImageWrapper

    def __init__(self, left_wrapper, right_wrapper):
        self.left_wrapper = left_wrapper
        self.right_wrapper = right_wrapper

    def siftKeypoint(self):
        pass

    def siftMatch(self):
        pass


class Assign5:
    wrapper: hwutil.ImageWrapper

    def __init__(self, wrapper):
        self.left_wrapper = wrapper

    def showAugment(self):
        pass

    def showModel(self):
        pass

    def showAccuracyLoss(self):
        pass

    def predictLabel(self):
        pass
