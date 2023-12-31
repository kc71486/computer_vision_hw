import sys
import os

import numpy as np

import cv2

from PyQt5 import QtWidgets, QtGui, QtCore

import matplotlib
from matplotlib import pyplot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from typing import Any, Callable, Iterable, Optional

import backend
import hwutil

matplotlib.use("Qt5Agg")


class MainWindow:
    app: QtWidgets.QApplication
    cwd: str
    imageloader: Any
    left_wrapper: hwutil.ImageWrapper
    right_wrapper: hwutil.ImageWrapper
    assign1: backend.Assign1
    assign2: backend.Assign2
    assign3: backend.Assign3
    assign4: backend.Assign4
    assign5: backend.Assign5
    previewLabel: Optional[QtWidgets.QLabel]
    predictLabel: Optional[QtWidgets.QLabel]
    main_panel: QtWidgets.QWidget
    image_window: 'ImageWindow'
    plot_window: 'CanvasWindow'
    outfile: Any

    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.cwd = os.getcwd()
        self.imageloader = hwutil.ImageLoader()
        self.left_wrapper = hwutil.ImageWrapper()
        self.right_wrapper = hwutil.ImageWrapper()

        self.main_panel = QtWidgets.QWidget()
        self.main_panel.setWindowTitle("Main Window")
        self.previewLabel = None
        self.predictLabel = None

        self.image_window = ImageWindow((2, 2))
        self.plot_window = CanvasWindow()
        self.outfile = OutFile()

        main_layout = QtWidgets.QGridLayout(self.main_panel)
        main_layout.addWidget(self.get_group0(), 0, 0)
        main_layout.addWidget(self.get_group1(), 0, 1)
        main_layout.addWidget(self.get_group2(), 0, 2)
        main_layout.addWidget(self.get_group3(), 0, 3)
        main_layout.addWidget(self.get_group4(), 1, 1)
        main_layout.addWidget(self.get_group5(), 1, 2)
        main_layout.addWidget(self.get_group6(), 1, 3)

    def run(self) -> None:
        self.main_panel.setVisible(True)
        sys.exit(self.app.exec_())

    def get_group0(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Load Image")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("Load Folder")
        button1.clicked.connect(self.choose_folder)

        button2 = QtWidgets.QPushButton("Load First/Left")
        button2.clicked.connect(self.choose_left)

        button3 = QtWidgets.QPushButton("Load Second/Right")
        button3.clicked.connect(self.choose_right)

        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)

        return group

    def get_group1(self) -> QtWidgets.QGroupBox:
        self.assign1 = backend.Assign1(self.imageloader)

        group = QtWidgets.QGroupBox("Calibration")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("1.1 Find corners")
        button1.clicked.connect(
            lambda: self.image_window.show_interval("1.1",
                                                    self.assign1.loop1, 2))

        button2 = QtWidgets.QPushButton("1.2 Find intrinsic")
        button2.clicked.connect(
            lambda: self.outfile.print(self.assign1.find_intrinsic()))

        group3 = QtWidgets.QGroupBox("Find extrinsic")
        layout3 = QtWidgets.QVBoxLayout(group3)

        spinbox3 = QtWidgets.QSpinBox()

        button3 = QtWidgets.QPushButton("1.3 Find extrinsic")
        button3.clicked.connect(
            lambda: self.outfile.print(self.assign1.find_extrinsic(
                self.convert_index(spinbox3.value(), self.imageloader.files, "bmp"))))

        layout3.addWidget(spinbox3)
        layout3.addWidget(button3)

        button4 = QtWidgets.QPushButton("1.4 Find distortion")
        button4.clicked.connect(
            lambda: self.outfile.print(self.assign1.find_distortion()))

        button5 = QtWidgets.QPushButton("1.5 Show result")
        button5.clicked.connect(
            lambda: self.image_window.show_interval_multi("1.5",
                                                          self.assign1.loop5, 2))

        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(group3)
        layout.addWidget(button4)
        layout.addWidget(button5)

        return group

    def get_group2(self) -> QtWidgets.QGroupBox:
        self.assign2 = backend.Assign2(self.imageloader)

        group = QtWidgets.QGroupBox("Augmented Reality")
        layout = QtWidgets.QVBoxLayout(group)

        input0 = QtWidgets.QLineEdit()

        button1 = QtWidgets.QPushButton("2.1 Show words on board")
        button1.clicked.connect(
            lambda: self.image_window.show_interval("2.1",
                                                    lambda: self.assign2.loop1(input0.text()),
                                                    2))

        button2 = QtWidgets.QPushButton("2.2 Show words vertical")
        button2.clicked.connect(
            lambda: self.image_window.show_interval("2.2",
                                                    lambda: self.assign2.loop2(input0.text()),
                                                    2))

        layout.addWidget(input0)
        layout.addWidget(button1)
        layout.addWidget(button2)

        return group

    def get_group3(self) -> QtWidgets.QGroupBox:
        self.assign3 = backend.Assign3(self.left_wrapper, self.right_wrapper)

        group = QtWidgets.QGroupBox("Stereo disparity map")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("3.1 Stereo disparity map")
        button1.clicked.connect(self.click_event_q3)

        layout.addWidget(button1)

        return group

    def get_group4(self) -> QtWidgets.QGroupBox:
        self.assign4 = backend.Assign4(self.left_wrapper, self.right_wrapper)

        group = QtWidgets.QGroupBox("SIFT")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("4.1 Keypoints")
        button1.clicked.connect(
            lambda: self.image_window.show_image("4.1", self.assign4.sift_keypoint()))

        button2 = QtWidgets.QPushButton("4.2 Matched Keypoints")
        button2.clicked.connect(
            lambda: self.image_window.show_image("4.2", self.assign4.sift_match()))

        layout.addWidget(button1)
        layout.addWidget(button2)

        return group

    def get_group5(self) -> QtWidgets.QGroupBox:
        self.assign5 = backend.Assign5(self.imageloader, self.left_wrapper)

        group = QtWidgets.QGroupBox("VGG19")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("5.1 Show Augmented Images")
        button1.clicked.connect(self.show_augmented_gui)

        button2 = QtWidgets.QPushButton("5.2 Show Model Structure")
        button2.clicked.connect(
            lambda: self.outfile.print(self.assign5.show_model()))

        button3 = QtWidgets.QPushButton("5.3 Show Acc and Loss")
        button3.clicked.connect(self.show_accuracy_plot)

        button4 = QtWidgets.QPushButton("5.4 Inference")
        button4.clicked.connect(self.show_predict_q5)

        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)
        layout.addWidget(button4)

        return group

    def get_group6(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Q5 extra")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("Load and Show Image")
        button1.clicked.connect(self.choose_and_show_left)

        label1 = QtWidgets.QLabel()
        label1.setFixedSize(128, 128)
        self.previewLabel = label1

        label2 = QtWidgets.QLabel()
        label2.setText("Predicted=")
        self.predictLabel = label2

        layout.addWidget(button1)
        layout.addWidget(label1)
        layout.addWidget(label2)

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
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_panel, "choose left", self.cwd)
        self.left_wrapper.set_path(chosen)

    def choose_right(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_panel, "choose right", self.cwd)
        self.right_wrapper.set_path(chosen)

    def click_event_q3(self) -> None:
        self.image_window.show_images_func("3.1",
                                           lambda: self.assign3.disparity_image())
        self.image_window.add_label_event(0, self.click_label_q3)

    def click_label_q3(self, event: QtGui.QMouseEvent) -> None:
        x_val = int(event.x() / backend.Assign3.resize_ratio)
        y_val = int(event.y() / backend.Assign3.resize_ratio)
        img, msg = self.assign3.disparity_value((x_val, y_val))
        self.image_window.refresh(1, img)
        self.outfile.print(msg)

    def choose_and_show_left(self) -> None:
        self.choose_left()
        preview = self.left_wrapper.read()

        preview = cv2.resize(preview, (128, 128))
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        h, w, d = preview.shape
        pixmap = QtGui.QPixmap(QtGui.QImage(
            preview.data.tobytes(), w, h, w * d, QtGui.QImage.Format_RGB888))
        self.previewLabel.setPixmap(pixmap)

    def show_accuracy_plot(self) -> None:
        train_loss, train_accuracy, valid_loss, valid_accuracy = self.assign5.show_accuracy_loss()
        epochs = len(train_loss)

        fig = pyplot.figure()
        pyplot.subplot(2, 1, 1)
        pyplot.ylabel("train / validation loss")
        pyplot.xlabel("epochs")
        pyplot.plot([*range(1, epochs + 1)], train_loss, "r",
                    [*range(1, epochs + 1)], valid_loss, "b")

        pyplot.subplot(2, 1, 2)
        pyplot.ylabel("train / validation accuracy")
        pyplot.xlabel("epochs")
        pyplot.plot([*range(1, epochs + 1)], train_accuracy, 'r',
                    [*range(1, epochs + 1)], valid_accuracy, 'b')

        pyplot.tight_layout()

        self.plot_window.show_plot("5.3", fig)

    def show_augmented_gui(self) -> None:
        images = self.assign5.show_augment()

        fig = pyplot.figure()
        for idx, img in enumerate(images):
            fig.add_subplot(3, 3, idx + 1)
            pyplot.imshow(img)
            pyplot.axis('off')

        self.plot_window.show_plot("5.1", fig)

    def show_predict_q5(self) -> None:
        labels, predicted, predicted_class = self.assign5.predict_label()

        fig = pyplot.figure()
        pyplot.bar(labels, predicted)

        self.plot_window.show_plot("5.4", fig)
        self.predictLabel.setText(f"Predicted={predicted_class}")


class ClickLabel(QtWidgets.QLabel):
    released = QtCore.pyqtSignal(QtGui.QMouseEvent)
    event_info: Optional[QtGui.QMouseEvent]
    connected_func: Optional[Callable]

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.event_info = None
        self.connected_func = None

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        self.event_info = event
        self.released.emit(event)
        self.event_info = None

    def connect_event(self, func: Callable) -> None:
        if self.connected_func is not None:
            self.released.disconnect()
        self.released.connect(func)
        self.connected_func = func

    def remove_event(self) -> None:
        if self.connected_func is not None:
            self.released.disconnect()
            self.connected_func = None


class CanvasWindow(QtWidgets.QWidget):
    canvas: Optional[FigureCanvas]
    layout: QtWidgets.QLayout

    def __init__(self) -> None:
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.canvas = None

    def __set_canvas(self, fig: matplotlib.pyplot.Figure):
        old_canvas = self.canvas
        if old_canvas is not None:
            new_canvas = FigureCanvas(fig)
            self.layout.replaceWidget(old_canvas, new_canvas)
            old_canvas.setParent(None)
            self.canvas = new_canvas
        else:
            new_canvas = FigureCanvas(fig)
            self.layout.addWidget(new_canvas)
            self.canvas = new_canvas

    def __display(self, title: str) -> None:
        super().setWindowTitle(title)
        super().setVisible(True)

    def show_plot(self, title: str, fig: matplotlib.pyplot.Figure) -> None:
        self.__set_canvas(fig)
        self.__display(title)


class ImageWindow(QtWidgets.QWidget):
    labels: list[ClickLabel]
    work_threads: list

    def __init__(self, grid_size: tuple[int, int]) -> None:
        super().__init__()
        self.labels = []
        self.work_threads = []
        layout = QtWidgets.QGridLayout(self)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                label = ClickLabel(self)
                self.labels.append(label)
                layout.addWidget(label, i, j)
        self.work_threads = []

    def __clear_thread(self) -> None:
        for thread in self.work_threads:
            thread.cancel()
        self.work_threads = []
        for index, label in enumerate(self.labels):
            self.labels[index].remove_event()

    def __clear_label(self) -> None:
        for label in self.labels:
            label.clear()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        super().closeEvent(event)
        self.__clear_thread()

    def __post_img_idx(self, index: int, img: np.ndarray) -> None:
        h, w, d = img.shape
        pixmap = QtGui.QPixmap(QtGui.QImage(
            img.data.tobytes(), w, h, w * d, QtGui.QImage.Format_RGB888))
        self.labels[index].setPixmap(pixmap)

    def __post_img_func_idx(self, index: int, img_func: Callable) -> None:
        img = img_func()
        h, w, d = img.shape
        pixmap = QtGui.QPixmap(QtGui.QImage(
            img.data.tobytes(), w, h, w * d, QtGui.QImage.Format_RGB888))
        self.labels[index].setPixmap(pixmap)

    def __post_img_func(self, img_func: Callable) -> None:
        images = img_func()
        for index, image in enumerate(images):
            h, w, d = image.shape
            pixmap = QtGui.QPixmap(QtGui.QImage(
                image.data.tobytes(), w, h, w * d, QtGui.QImage.Format_RGB888))
            self.labels[index].setPixmap(pixmap)

    def __display(self, title: str) -> None:
        super().setWindowTitle(title)
        super().setVisible(True)

    def show_images(self, title: str, images: Iterable[np.ndarray]) -> None:
        self.__clear_thread()
        self.__clear_label()
        for index, img in enumerate(images):
            self.__post_img_idx(index, img)
        self.__display(title)

    def show_images_func(self, title: str, img_func: Callable[[], Iterable[np.ndarray]]) -> None:
        self.__clear_thread()
        self.__clear_label()
        images = img_func()
        for index, img in enumerate(images):
            self.__post_img_idx(index, img)
        self.__display(title)

    def show_image(self, title: str, img: np.ndarray) -> None:
        self.__clear_thread()
        self.__clear_label()
        self.__post_img_idx(0, img)
        self.__display(title)

    def show_interval(self, title: str, img_func: Callable[[], np.ndarray], interval: float | int) -> None:
        self.__clear_thread()
        self.__clear_label()
        self.work_threads.append(hwutil.SetInterval(interval,
                                                    lambda: self.__post_img_func_idx(0, img_func)))
        self.__display(title)

    def show_interval_multi(self, title: str, img_func: Callable, interval: float | int) -> None:
        self.__clear_thread()
        self.__clear_label()
        self.work_threads.append(hwutil.SetInterval(interval,
                                                    lambda: self.__post_img_func(img_func)))
        self.__display(title)

    def refresh(self, index: int, img: np.ndarray) -> None:
        self.__post_img_idx(index, img)

    def add_label_event(self, index: int, func: Callable) -> None:
        self.labels[index].connect_event(func)

    def remove_label_event(self, index: int) -> None:
        self.labels[index].remove_event()


class OutFile:
    _fd: int

    def __init__(self) -> None:
        self._fd = 1

    def print(self, obj: Any) -> None:
        self._fd = 1
        print(obj)
