import os

import threading

import numpy as np

from typing import Any, Optional, Sequence

import cv2

import hwutil

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


class Assign1:
    board_corner: tuple
    imageloader: hwutil.ImageLoader
    last_updated: float
    calc_thread: threading.Thread
    intrinsic: Optional[np.ndarray]
    extrinsic: Sequence[np.ndarray]
    distortion: Optional[np.ndarray]
    __ctr1: int
    __ctr5: int

    def __init__(self, imageloader: hwutil.ImageLoader):
        self.board_corner = (11, 8)
        self.imageloader = imageloader
        self.last_updated = -2
        self.calc_thread = threading.Thread(target=None)
        self.intrinsic = None
        self.extrinsic = []
        self.distortion = None
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

    def __calc_camera(self) -> None:
        image_shape = self.imageloader[0].shape
        cx, cy = self.board_corner

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        object_point = np.zeros((cx * cy, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2)

        object_points = []
        image_points = []

        for img in self.imageloader:
            found, corners = cv2.findChessboardCorners(
                img, self.board_corner, None)
            if found:
                image_points.append(corners)
                object_points.append(object_point)

        camera_matrix = np.identity(3)
        distortion = np.zeros(5)

        ret, mtx, dist, r_vectors, t_vectors = cv2.calibrateCamera(
            object_points, image_points, image_shape[0:2], camera_matrix, distortion)

        self.intrinsic = mtx
        self.distortion = dist
        extrinsic = []

        for idx in range(len(r_vectors)):
            r_vector = cv2.Rodrigues(r_vectors[idx])
            t_vector = t_vectors[idx]
            extrinsic.append(np.concatenate((r_vector[0], t_vector), axis=1))

        self.extrinsic = extrinsic
        self.last_updated = self.imageloader.last_updated

    def __start_calc(self) -> None:
        if self.calc_thread.is_alive():
            return
        if self.last_updated == self.imageloader.last_updated:
            return
        self.calc_thread = threading.Thread(target=self.__calc_camera)
        self.calc_thread.start()

    def find_corner(self, idx: int) -> np.ndarray:
        self.__start_calc()
        img = self.imageloader[idx]
        find, corners = cv2.findChessboardCorners(img, self.board_corner)
        if find:
            cv2.drawChessboardCorners(img, self.board_corner, corners, find)

        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def find_intrinsic(self) -> np.ndarray:
        self.__start_calc()
        while self.last_updated != self.imageloader.last_updated:
            pass

        return self.intrinsic

    def find_extrinsic(self, idx: int) -> np.ndarray:
        self.__start_calc()
        while self.last_updated != self.imageloader.last_updated:
            pass

        return self.extrinsic[idx]

    def find_distortion(self):
        self.__start_calc()
        while self.last_updated != self.imageloader.last_updated:
            pass

        return self.distortion

    def show_undistorted(self, idx: int):
        self.__start_calc()
        while self.last_updated != self.imageloader.last_updated:
            pass

        img = self.imageloader[idx]
        dst = cv2.undistort(img, self.intrinsic, self.distortion)

        dst = cv2.resize(dst, (512, 512))
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        return dst


class Assign2:
    board_corner: tuple[int, int]
    imageloader: hwutil.ImageLoader
    last_updated: float
    calc_thread: threading.Thread
    intrinsic: Optional[np.ndarray]
    r_vectors: Sequence[np.ndarray]
    t_vectors: Sequence[np.ndarray]
    dist: Any
    projection: list

    def __init__(self, imageloader: hwutil.ImageLoader):
        self.board_corner = (11, 8)
        self.imageloader = imageloader
        self.last_updated = -2
        self.calc_thread = threading.Thread(target=None)
        self.intrinsic = None
        self.r_vectors = []
        self.t_vectors = []
        self.dist = None
        self.projection = []

    def calc_projection(self):
        image_shape = self.imageloader[0].shape
        cx, cy = self.board_corner

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        object_point = np.zeros((cx * cy, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2)

        object_points = []
        image_points = []

        for img in self.imageloader:
            found, corners = cv2.findChessboardCorners(img, self.board_corner, None)
            if found:
                image_points.append(corners)
                object_points.append(object_point)

        camera_matrix = np.identity(3)

        ret, mtx, dist, r_vectors, t_vectors = cv2.calibrateCamera(
            object_points, image_points, image_shape[0:2], camera_matrix, None)

        self.intrinsic = mtx
        self.r_vectors = r_vectors
        self.t_vectors = t_vectors
        self.dist = dist

    def __get_fs_path(self, filename: str) -> str:
        dir_name = os.path.dirname(self.imageloader.files[0])
        return os.path.join(dir_name, 'Q2_lib', filename)

    def __get_ar(self, idx: int, input_string: str, filename: str, base_coord: np.ndarray):
        self.calc_projection()
        fs = cv2.FileStorage(self.__get_fs_path(filename), cv2.FILE_STORAGE_READ)

        line_start = []
        line_end = []
        for i in range(len(input_string)):
            ch = fs.getNode(input_string[i]).mat()
            for j in range(len(ch)):
                line_start.append(np.add(ch[j][0], base_coord[i]))
                line_end.append(np.add(ch[j][1], base_coord[i]))

        line_start = np.array(line_start, np.float32)
        line_end = np.array(line_end, np.float32)

        point_start, _ = cv2.projectPoints(line_start, self.r_vectors[idx],
                                           self.t_vectors[idx], self.intrinsic, self.dist)
        point_end, _ = cv2.projectPoints(line_end, self.r_vectors[idx],
                                         self.t_vectors[idx], self.intrinsic, self.dist)

        img = self.imageloader[idx]
        for i in range(len(line_start)):
            start = point_start[i][0].astype(int)
            end = point_end[i][0].astype(int)
            cv2.line(img, start, end, (0, 0, 255), 10)

        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def ar_board(self, idx: int, input_string: str):
        if len(input_string) > 6:
            raise Exception("string too long")
        if len(input_string) == 0:
            raise Exception("empty string")

        base_coord = np.array([[7, 5, 0], [4, 5, 0], [1, 5, 0],
                              [7, 2, 0], [4, 2, 0], [1, 2, 0]])

        return self.__get_ar(idx, input_string, "alphabet_lib_onboard.txt", base_coord)

    def ar_vertical(self, idx: int, input_string: str):
        if len(input_string) > 6:
            raise Exception("string too long")
        if len(input_string) == 0:
            raise Exception("empty string")

        base_coord = np.array([[7, 5, 0], [4, 5, 0], [1, 5, 0],
                               [7, 2, 0], [4, 2, 0], [1, 2, 0]])

        return self.__get_ar(idx, input_string, "alphabet_lib_vertical.txt", base_coord)


class Assign3:
    left_wrapper: hwutil.ImageWrapper
    right_wrapper: hwutil.ImageWrapper

    def __init__(self, left_wrapper: hwutil.ImageWrapper, right_wrapper: hwutil.ImageWrapper) -> None:
        self.left_wrapper = left_wrapper
        self.right_wrapper = right_wrapper

    def disparity_value(self, coord: tuple[int, int]) -> \
            tuple[np.ndarray, np.ndarray, np.ndarray]:
        out_left = self.left_wrapper.read()
        out_right = self.right_wrapper.read()
        left = cv2.cvtColor(out_left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(out_right, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM_create()
        disp = stereo.compute(left, right)
        disparity = np.divide(disp.astype(np.float32), 16).astype(int)

        disparity_val = disparity[coord[1]][coord[0]]
        if disparity_val >= 0:
            new_coord = (coord[0] + disparity_val, coord[1])

            cv2.circle(out_right, new_coord, radius=5,
                       color=(0, 0, 255), thickness=-1)

        out_left = cv2.resize(out_left, (512, 512))
        out_left = cv2.cvtColor(out_left, cv2.COLOR_BGR2RGB)

        out_right = cv2.resize(out_right, (512, 512))
        out_right = cv2.cvtColor(out_right, cv2.COLOR_BGR2RGB)

        disp = np.maximum(disp, 0)
        disp = cv2.resize(disp, (512, 512))
        disp = cv2.merge((disp, disp, disp))

        return out_left, out_right, disp


class Assign4:
    left_wrapper: hwutil.ImageWrapper
    right_wrapper: hwutil.ImageWrapper

    def __init__(self, left_wrapper, right_wrapper) -> None:
        self.left_wrapper = left_wrapper
        self.right_wrapper = right_wrapper

    def sift_keypoint(self):
        sift = cv2.SIFT.create()
        pass

    def sift_match(self):
        pass


class Assign5:
    wrapper: hwutil.ImageWrapper

    def __init__(self, wrapper):
        self.left_wrapper = wrapper

    def show_augment(self):
        pass

    def show_model(self):
        pass

    def show_accuracy_loss(self):
        pass

    def predict_label(self):
        pass
