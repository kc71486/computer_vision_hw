import sys
import os

import numpy as np

from PyQt5 import QtWidgets, QtGui

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from typing import Any, Callable, Iterable

import backend
import hwutil

matplotlib.use("Qt5Agg")


class MainWindow:
    app: QtWidgets.QApplication
    cwd: str
    imageloader: Any
    left_wrapper: hwutil.ImageWrapper
    right_wrapper: hwutil.ImageWrapper
    assign: list[Any]
    main_panel: QtWidgets.QWidget
    image_window: 'ImageWindow'
    outfile: Any

    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.cwd = os.getcwd()
        self.imageloader = hwutil.ImageLoader()
        self.left_wrapper = hwutil.ImageWrapper()
        self.right_wrapper = hwutil.ImageWrapper()
        self.assign = [None] * 5

        self.main_panel = QtWidgets.QWidget()
        self.main_panel.setWindowTitle("Main Window")

        self.image_window = ImageWindow((1, 3))
        self.outfile = OutFile()

        main_layout = QtWidgets.QGridLayout(self.main_panel)
        main_layout.addWidget(self.get_group0(), 0, 0)
        main_layout.addWidget(self.get_group1(), 0, 1)
        main_layout.addWidget(self.get_group2(), 0, 2)
        main_layout.addWidget(self.get_group3(), 0, 3)
        main_layout.addWidget(self.get_group4(), 1, 1)
        main_layout.addWidget(self.get_group5(), 1, 2)

    def run(self) -> None:
        self.main_panel.setVisible(True)
        sys.exit(self.app.exec_())

    def get_group0(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Load Image")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("Load Folder")
        button1.clicked.connect(self.choose_folder)

        button2 = QtWidgets.QPushButton("Load First/Left Image")
        button2.clicked.connect(self.choose_left)

        button3 = QtWidgets.QPushButton("Load Second/Right Image")
        button3.clicked.connect(self.choose_right)

        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)

        return group

    def get_group1(self) -> QtWidgets.QGroupBox:
        self.assign[0] = backend.Assign1(self.imageloader)

        group = QtWidgets.QGroupBox("Calibration")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("1.1 Find corners")
        button1.clicked.connect(
            lambda: self.image_window.show_interval("1.1",
                                                    self.assign[0].loop1, 2))

        button2 = QtWidgets.QPushButton("1.2 Find intrinsic")
        button2.clicked.connect(
            lambda: self.outfile.print(self.assign[0].find_intrinsic()))

        group3 = QtWidgets.QGroupBox("Find extrinsic")
        layout3 = QtWidgets.QVBoxLayout(group3)

        spinbox3 = QtWidgets.QSpinBox()

        button3 = QtWidgets.QPushButton("1.3 Find extrinsic")
        button3.clicked.connect(
            lambda: self.outfile.print(self.assign[0].find_extrinsic(
                self.convert_index(spinbox3.value(), self.imageloader.files, "bmp"))))

        layout3.addWidget(spinbox3)
        layout3.addWidget(button3)

        button4 = QtWidgets.QPushButton("1.4 Find distortion")
        button4.clicked.connect(
            lambda: self.outfile.print(self.assign[0].find_distortion()))

        button5 = QtWidgets.QPushButton("1.5 Show result")
        button5.clicked.connect(
            lambda: self.image_window.show_interval("1.5",
                                                    self.assign[0].loop5, 2))

        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(group3)
        layout.addWidget(button4)
        layout.addWidget(button5)

        return group

    def get_group2(self) -> QtWidgets.QGroupBox:
        self.assign[1] = backend.Assign2(self.imageloader)

        group = QtWidgets.QGroupBox("Augmented Reality")
        layout = QtWidgets.QVBoxLayout(group)

        input0 = QtWidgets.QLineEdit()

        button1 = QtWidgets.QPushButton("2.1 Show words on board")
        button1.clicked.connect(
            lambda: self.image_window.show_image("2.1",
                                                 self.assign[1].ar_board(2, input0.text())))

        button2 = QtWidgets.QPushButton("2.2 Show words vertical")
        button2.clicked.connect(
            lambda: self.image_window.show_image("2.2",
                                                 self.assign[1].ar_vertical(2, input0.text())))

        layout.addWidget(input0)
        layout.addWidget(button1)
        layout.addWidget(button2)

        return group

    def get_group3(self) -> QtWidgets.QGroupBox:
        self.assign[2] = backend.Assign3(self.left_wrapper, self.right_wrapper)

        group = QtWidgets.QGroupBox("Stereo disparity map")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("3.1 Stereo disparity map")
        button1.clicked.connect(self.click_event_q3)

        layout.addWidget(button1)

        return group

    def get_group4(self) -> QtWidgets.QGroupBox:
        self.assign[3] = None

        group = QtWidgets.QGroupBox("SIFT")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("4.1 Keypoints")

        button2 = QtWidgets.QPushButton("4.2 Matched Keypoints")

        layout.addWidget(button1)
        layout.addWidget(button2)

        return group

    def get_group5(self) -> QtWidgets.QGroupBox:
        self.assign[4] = None

        group = QtWidgets.QGroupBox("VGG19")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("5.1 Show Augmented Images")

        button2 = QtWidgets.QPushButton("5.2 Show Model Structure")

        button3 = QtWidgets.QPushButton("5.3 Show Acc and Loss")

        button4 = QtWidgets.QPushButton("5.4 Inference")

        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)
        layout.addWidget(button4)

        return group

    @staticmethod
    def convert_index(value, filepaths: list[str], ext: str) -> int:
        dir_name = os.path.dirname(filepaths[0])
        file = os.path.join(dir_name, str(value) + "." + ext)
        return filepaths.index(file)

    def choose_folder(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self.main_panel, "choose folder", self.cwd)
        self.imageloader.set_path(chosen)

    def choose_left(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_panel, "choose file", self.cwd)
        self.left_wrapper.set_path(chosen)

    def choose_right(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_panel, "choose file", self.cwd)
        self.right_wrapper.set_path(chosen)

    def click_event_q3(self) -> None:
        self.image_window.show_images_func("3.1",
                                           lambda: self.assign[2].disparity_value((700, 400)))


class ImageWindow(QtWidgets.QWidget):
    labels: list[QtWidgets.QLabel]
    work_threads: list

    def __init__(self, grid_size: tuple[int, int]):
        super().__init__()
        self.labels = []
        self.work_threads = []
        layout = QtWidgets.QGridLayout(self)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                label = QtWidgets.QLabel(self)
                self.labels.append(label)
                label.mousePressEvent = None
                layout.addWidget(label, i, j)
        self.work_threads = []

    def __clear_thread(self) -> None:
        for thread in self.work_threads:
            thread.cancel()
        self.work_threads = []
        for index, label in enumerate(self.labels):
            if label.mousePressEvent is not None:
                self.lebel[index].mousePressEvent = None

    def __clear_label(self) -> None:
        for label in self.labels:
            label.clear()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        super().closeEvent(event)
        self.__clear_thread()

    def __post_img_idx(self, index: int, img: np.ndarray) -> None:
        h, w, d = img.shape
        pixmap = QtGui.QPixmap(QtGui.QImage(
            img.data, w, h, w * d, QtGui.QImage.Format_RGB888))
        self.labels[index].setPixmap(pixmap)

    def __post_img_func_idx(self, index: int, imgfunc: Callable) -> None:
        img = imgfunc()
        h, w, d = img.shape
        pixmap = QtGui.QPixmap(QtGui.QImage(
            img.data, w, h, w * d, QtGui.QImage.Format_RGB888))
        self.labels[index].setPixmap(pixmap)

    def __post_img_func(self, imgfunc: Callable) -> None:
        images = imgfunc()
        for img in images:
            h, w, d = img.shape
            pixmap = QtGui.QPixmap(QtGui.QImage(
                img.data, w, h, w * d, QtGui.QImage.Format_RGB888))
            self.labels[0].setPixmap(pixmap)

    def __display(self, title: str) -> None:
        super().setWindowTitle(title)
        super().setVisible(True)

    def show_images(self, title: str, images: Iterable[np.ndarray]) -> None:
        self.__clear_thread()
        self.__clear_label()
        for index, img in enumerate(images):
            self.__post_img_idx(index, img)
        self.__display(title)

    def show_images_func(self, title: str, imgfunc: Callable) -> None:
        self.__clear_thread()
        self.__clear_label()
        imgs = imgfunc()
        for index, img in enumerate(imgs):
            self.__post_img_idx(index, img)
        self.__display(title)

    def show_image(self, title: str, img: np.ndarray) -> None:
        self.__clear_thread()
        self.__clear_label()
        self.__post_img_idx(0, img)
        self.__display(title)

    def show_interval(self, title: str, imgfunc: Callable, interval: float | int) -> None:
        self.__clear_thread()
        self.__clear_label()
        self.work_threads.append(hwutil.SetInterval(interval,
                                                    lambda: self.__post_img_func_idx(0, imgfunc)))
        self.__display(title)

    def refresh(self, index: int, img: np.ndarray) -> None:
        self.__removeThread(index)
        self.__post_img_idx(index, img)

    def add_click_event(self, index: int, func: Callable) -> None:
        self.label[index].mousePressEvent = func


class OutFile:
    _fd: int

    def __init__(self) -> None:
        self._fd = 1

    def print(self, obj: Any) -> None:
        self._fd = 1
        print(obj)
