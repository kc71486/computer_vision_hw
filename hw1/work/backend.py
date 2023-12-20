import os

import threading

import numpy as np

import pickle

from typing import Optional, Sequence, Final

import cv2

from PIL import Image

import torch
import torchvision
from torch import nn
import torchinfo

import hwutil

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


class Assign1:
    BOARD_CORNER: Final = (11, 8)
    resize_ratio = 0.25
    imageloader: hwutil.ImageLoader
    last_updated: float
    calc_thread: threading.Thread
    intrinsic: Optional[np.ndarray]
    extrinsic: Sequence[np.ndarray]
    distortion: Optional[np.ndarray]
    __ctr1: int
    __ctr5: int

    def __init__(self, imageloader: hwutil.ImageLoader):
        self.imageloader = imageloader
        self.last_updated = -2
        self.calc_thread = threading.Thread(target=None)
        self.intrinsic = None
        self.extrinsic = []
        self.distortion = None
        self.__ctr1 = 0
        self.__ctr5 = 0

    def loop1(self) -> np.ndarray:
        image = self.find_corner(self.__ctr1)
        self.__ctr1 = (self.__ctr1 + 1) % len(self.imageloader)
        return image

    def loop5(self) -> tuple[np.ndarray, np.ndarray]:
        images = self.show_undistorted(self.__ctr5)
        self.__ctr5 = (self.__ctr5 + 1) % len(self.imageloader)
        return images

    def __calc_camera(self) -> None:
        image_shape = self.imageloader[0].shape
        cx, cy = Assign1.BOARD_CORNER

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        object_point = np.zeros((cx * cy, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2)

        object_points = []
        image_points = []

        for img in self.imageloader:
            found, corners = cv2.findChessboardCorners(
                img, Assign1.BOARD_CORNER, None)
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
        find, corners = cv2.findChessboardCorners(img, Assign1.BOARD_CORNER)
        if find:
            cv2.drawChessboardCorners(img, Assign1.BOARD_CORNER, corners, find)

        imgsize = img.shape
        img = cv2.resize(img, (int(imgsize[1] * Assign1.resize_ratio), int(imgsize[0] * Assign1.resize_ratio)))
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

    def show_undistorted(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        self.__start_calc()
        while self.last_updated != self.imageloader.last_updated:
            pass

        img = self.imageloader[idx]
        dst = cv2.undistort(img, self.intrinsic, self.distortion)

        imgsize = img.shape
        img = cv2.resize(img, (int(imgsize[1] * Assign1.resize_ratio), int(imgsize[0] * Assign1.resize_ratio)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dst_size = dst.shape
        dst = cv2.resize(dst, (int(dst_size[1] * Assign1.resize_ratio), int(dst_size[0] * Assign1.resize_ratio)))
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        return img, dst


class Assign2:
    BOARD_CORNER: Final = (11, 8)
    resize_ratio = 0.25
    imageloader: hwutil.ImageLoader
    last_updated: float
    calc_thread: threading.Thread
    intrinsic: Optional[np.ndarray]
    r_vectors: Sequence[np.ndarray]
    t_vectors: Sequence[np.ndarray]
    distortion: Optional[np.ndarray]
    projection: Sequence
    __ctr1: int
    __ctr2: int

    def __init__(self, imageloader: hwutil.ImageLoader):
        self.imageloader = imageloader
        self.last_updated = -2
        self.calc_thread = threading.Thread(target=None)
        self.intrinsic = None
        self.r_vectors = []
        self.t_vectors = []
        self.distortion = None
        self.__ctr1 = 0
        self.__ctr2 = 0

    def loop1(self, input_string: str) -> np.ndarray:
        image = self.ar_board(self.__ctr1, input_string)
        self.__ctr1 = (self.__ctr1 + 1) % len(self.imageloader)
        return image

    def loop2(self, input_string: str) -> np.ndarray:
        image = self.ar_vertical(self.__ctr2, input_string)
        self.__ctr2 = (self.__ctr2 + 1) % len(self.imageloader)
        return image

    def __calc_projection(self) -> None:
        image_shape = self.imageloader[0].shape
        cx, cy = Assign2.BOARD_CORNER

        # grid, (0,0,0),(0,1,0),(0,2,0)...(1,0,0),(1,1,0)...(m,n,0)
        object_point = np.zeros((cx * cy, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2)

        object_points = []
        image_points = []

        for img in self.imageloader:
            found, corners = cv2.findChessboardCorners(img, Assign2.BOARD_CORNER, None)
            if found:
                image_points.append(corners)
                object_points.append(object_point)

        camera_matrix = np.identity(3)

        ret, mtx, dist, r_vectors, t_vectors = cv2.calibrateCamera(
            object_points, image_points, image_shape[0:2], camera_matrix, None)

        self.intrinsic = mtx
        self.r_vectors = r_vectors
        self.t_vectors = t_vectors
        self.distortion = dist
        self.last_updated = self.imageloader.last_updated

    def __start_calc(self) -> None:
        if self.calc_thread.is_alive():
            return
        if self.last_updated == self.imageloader.last_updated:
            return
        self.calc_thread = threading.Thread(target=self.__calc_projection)
        self.calc_thread.start()

    def __get_ar(self, idx: int, input_string: str, filename: str, base_coord: np.ndarray) -> np.ndarray:
        self.__start_calc()

        dir_name = os.path.dirname(self.imageloader.files[0])
        fs_filename = os.path.join(dir_name, 'Q2_lib', filename)
        fs = cv2.FileStorage(fs_filename, cv2.FILE_STORAGE_READ)

        line_start = []
        line_end = []
        for i in range(len(input_string)):
            ch = fs.getNode(input_string[i]).mat()
            for j in range(len(ch)):
                line_start.append(np.add(ch[j][0], base_coord[i]))
                line_end.append(np.add(ch[j][1], base_coord[i]))

        line_start = np.array(line_start, np.float32)
        line_end = np.array(line_end, np.float32)

        while self.last_updated != self.imageloader.last_updated:
            pass

        point_start, _ = cv2.projectPoints(line_start, self.r_vectors[idx],
                                           self.t_vectors[idx], self.intrinsic, self.distortion)
        point_end, _ = cv2.projectPoints(line_end, self.r_vectors[idx],
                                         self.t_vectors[idx], self.intrinsic, self.distortion)

        img = self.imageloader[idx]
        for i in range(len(line_start)):
            start = point_start[i][0].astype(int)
            end = point_end[i][0].astype(int)
            cv2.line(img, start, end, (0, 0, 255), 10)

        imgsize = img.shape
        img = cv2.resize(img, (int(imgsize[1] * Assign3.resize_ratio), int(imgsize[0] * Assign3.resize_ratio)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def ar_board(self, idx: int, input_string: str) -> np.ndarray:
        if len(input_string) > 6:
            raise Exception("string too long")
        if len(input_string) == 0:
            raise Exception("empty string")

        base_coord = np.array([[7, 5, 0], [4, 5, 0], [1, 5, 0],
                               [7, 2, 0], [4, 2, 0], [1, 2, 0]])

        return self.__get_ar(idx, input_string, "alphabet_lib_onboard.txt", base_coord)

    def ar_vertical(self, idx: int, input_string: str) -> np.ndarray:
        if len(input_string) > 6:
            raise Exception("string too long")
        if len(input_string) == 0:
            raise Exception("empty string")

        base_coord = np.array([[7, 5, 0], [4, 5, 0], [1, 5, 0],
                               [7, 2, 0], [4, 2, 0], [1, 2, 0]])

        return self.__get_ar(idx, input_string, "alphabet_lib_vertical.txt", base_coord)


class Assign3:
    resize_ratio = 0.25
    left_wrapper: hwutil.ImageWrapper
    right_wrapper: hwutil.ImageWrapper
    disparity: Optional[np.ndarray]

    def __init__(self, left_wrapper: hwutil.ImageWrapper, right_wrapper: hwutil.ImageWrapper) -> None:
        self.left_wrapper = left_wrapper
        self.right_wrapper = right_wrapper
        self.disparity = None

    def disparity_image(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out_left = self.left_wrapper.read()
        out_right = self.right_wrapper.read()
        left = cv2.cvtColor(out_left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(out_right, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM.create(numDisparities=256, blockSize=21)
        disp = stereo.compute(left, right)
        self.disparity = np.divide(disp.astype(np.float32), 16).astype(int)

        out_left_size = out_left.shape
        out_left = cv2.resize(out_left, (int(out_left_size[1] * Assign3.resize_ratio),
                                         int(out_left_size[0] * Assign3.resize_ratio)))
        out_left = cv2.cvtColor(out_left, cv2.COLOR_BGR2RGB)

        out_right_size = out_right.shape
        out_right = cv2.resize(out_right, (int(out_right_size[1] * Assign3.resize_ratio),
                                           int(out_right_size[0] * Assign3.resize_ratio)))
        out_right = cv2.cvtColor(out_right, cv2.COLOR_BGR2RGB)

        disp = np.maximum(disp, 0)

        disp_size = disp.shape
        disp = cv2.resize(disp, (int(disp_size[1] * Assign3.resize_ratio), int(disp_size[0] * Assign3.resize_ratio)))
        disp = cv2.merge((disp, disp, disp))
        return out_left, out_right, disp

    def disparity_value(self, coord: tuple[int, int]) -> tuple[np.ndarray, str]:
        out_right = self.right_wrapper.read()

        disparity = self.disparity

        out_str = str(coord) + ":"
        disparity_val = disparity[coord[1]][coord[0]]
        if disparity_val >= 0:
            new_coord = (coord[0] - disparity_val, coord[1])

            cv2.circle(out_right, new_coord, radius=20,
                       color=(0, 255, 0), thickness=-1)
        out_str = out_str + str(disparity_val)

        out_right_size = out_right.shape
        out_right = cv2.resize(out_right, (int(out_right_size[1] * Assign3.resize_ratio),
                                           int(out_right_size[0] * Assign3.resize_ratio)))
        out_right = cv2.cvtColor(out_right, cv2.COLOR_BGR2RGB)

        return out_right, out_str


class Assign4:
    resize_ratio = 0.25
    left_wrapper: hwutil.ImageWrapper
    right_wrapper: hwutil.ImageWrapper

    def __init__(self, left_wrapper, right_wrapper) -> None:
        self.left_wrapper = left_wrapper
        self.right_wrapper = right_wrapper

    def sift_keypoint(self) -> np.ndarray:
        img = self.left_wrapper.read()

        sift = cv2.SIFT.create()
        keypoint = sift.detect(img, None)

        new_img = cv2.drawKeypoints(img, keypoint, img, color=(0, 255, 0))
        new_img_size = new_img.shape
        new_img = cv2.resize(new_img, (int(new_img_size[1] * Assign4.resize_ratio),
                                       int(new_img_size[0] * Assign4.resize_ratio)))
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

        return new_img

    def sift_match(self):
        left = self.left_wrapper.read()
        right = self.right_wrapper.read()

        sift = cv2.SIFT.create()
        left_keypoint, left_des = sift.detectAndCompute(left, None)
        right_keypoint, right_des = sift.detectAndCompute(right, None)

        matches = cv2.BFMatcher().knnMatch(left_des, right_des, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        out_img = cv2.drawMatchesKnn(left, left_keypoint, right, right_keypoint, good, None)

        out_img_size = out_img.shape
        out_img = cv2.resize(out_img, (int(out_img_size[1] * Assign4.resize_ratio),
                                       int(out_img_size[0] * Assign4.resize_ratio)))
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

        return out_img


class Assign5:
    image_loader: hwutil.ImageLoader
    wrapper: hwutil.ImageWrapper
    weight_path: Final[str] = "VGG19_Epoch_50.pth"
    transform: torchvision.transforms.transforms
    model: Optional[nn.Module]
    weight_loaded: bool
    batches: Final[int]
    train_loss: Sequence
    train_accuracy: Sequence
    test_loss: Sequence
    test_accuracy: Sequence
    labels: Final[Sequence[str]] = ["airplane", "automobile", "bird", "cat", "deer",
                                    "dog", "frog", "horse", "ship", "truck"]

    def __init__(self, loader, wrapper: hwutil.ImageWrapper) -> None:
        self.image_loader = loader
        self.wrapper = wrapper
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomCrop(size=32, padding=2, pad_if_needed=True),
            torchvision.transforms.RandomRotation(degrees=30)
        ])
        self.model = None
        self.weight_loaded = False
        self.batches = 32
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

    def __yield_model(self) -> None:
        if self.model is None:
            self.model = VGG19()

    def __yield_stats(self) -> None:
        if len(self.test_accuracy) > 0:
            return
        with open("train_loss.p", "rb") as f:
            self.train_loss = pickle.load(f)
        with open("train_accuracy.p", "rb") as f:
            self.train_accuracy = pickle.load(f)
        with open("test_loss.p", "rb") as f:
            self.test_loss = pickle.load(f)
        with open("test_accuracy.p", "rb") as f:
            self.test_accuracy = pickle.load(f)

    def __yield_weight(self) -> None:
        if not self.weight_loaded:
            self.model.load_state_dict(torch.load(Assign5.weight_path, map_location=torch.device('cpu')))
            self.weight_loaded = True

    def show_augment(self) -> Sequence[Image.Image]:
        self.__yield_model()
        images = []
        for cv_img in self.image_loader:
            img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            images.append(self.transform(img))
        return images

    def show_model(self) -> torchinfo.ModelStatistics:
        self.__yield_model()
        return torchinfo.summary(self.model, input_size=(self.batches, 3, 32, 32), mode="eval", verbose=0)

    def show_accuracy_loss(self) -> tuple[Sequence, Sequence, Sequence, Sequence]:
        self.__yield_model()
        self.__yield_stats()

        return self.train_loss, self.train_accuracy, self.test_loss, self.test_accuracy

    def predict_label(self) -> tuple[Sequence, Sequence, str]:
        self.__yield_model()
        self.__yield_weight()

        cv_img = self.wrapper.read()
        image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        transform = torchvision.transforms.ToTensor()
        tensor = transform(image)
        tensor = torch.unsqueeze(tensor, dim=0)

        self.model.eval()
        with torch.no_grad():
            predicted = self.model(tensor)
        predicted = torch.squeeze(predicted)
        predicted = torch.nn.functional.softmax(predicted, dim=0)
        predicted_class = self.labels[torch.argmax(predicted).item()]

        return self.labels, predicted, predicted_class


class VGG19(torch.nn.Module):
    features: torch.nn.Module
    avg_pool: torch.nn.Module
    classifier: torch.nn.Module

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # set pool = (1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),  # set in_features = 512
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 10),  # set out = 10
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
