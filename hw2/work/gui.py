import sys
import os

import cv2
import numpy as np

from PyQt5 import QtWidgets, QtGui, QtCore

import matplotlib
from matplotlib import pyplot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from typing import Final, Callable, Iterable, Optional

import backend
import hwutil

matplotlib.use("Qt5Agg")


class MainWindow:
    app: QtWidgets.QApplication
    img_wrapper: hwutil.ImageWrapper
    vid_wrapper: hwutil.VideoWrapper
    assign1: Final[backend.Assign1]
    assign2: Final[backend.Assign2]
    assign3: Final[backend.Assign3]
    assign4: Final[backend.Assign4]
    assign5: Final[backend.Assign5]
    main_panel: QtWidgets.QWidget
    image_window: "ImageWindow"
    plot_window: "CanvasWindow"
    pix_area: "DrawBoard"
    preview_label: QtWidgets.QLabel
    result_label_5: QtWidgets.QLabel
    predict_label: QtWidgets.QLabel

    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.cwd = os.getcwd()
        self.img_wrapper = hwutil.ImageWrapper()
        self.vid_wrapper = hwutil.VideoWrapper()
        self.assign1 = backend.Assign1(self.vid_wrapper)
        self.assign2 = backend.Assign2(self.vid_wrapper)
        self.assign3 = backend.Assign3(self.img_wrapper)
        self.assign4 = backend.Assign4()
        self.assign5 = backend.Assign5(self.img_wrapper)

        self.image_window = ImageWindow((2, 3))
        self.plot_window = CanvasWindow()
        self.pix_area = DrawBoard(size=(256, 256),
                                  background=QtGui.QColor(0, 0, 0),
                                  pen_color=QtGui.QColor(255, 255, 255),
                                  pen_width=10)

        self.main_panel = QtWidgets.QWidget()
        self.main_panel.setWindowTitle("Main Window")

        main_layout = QtWidgets.QGridLayout(self.main_panel)
        main_layout.addWidget(self.get_group0(), 0, 0, 3, 1)
        main_layout.addWidget(self.get_group1(), 0, 1, 1, 1)
        main_layout.addWidget(self.get_group2(), 1, 1, 1, 1)
        main_layout.addWidget(self.get_group3(), 2, 1, 1, 1)
        main_layout.addWidget(self.get_group4(), 0, 2, 2, 1)
        main_layout.addWidget(self.get_group5(), 2, 2, 1, 1)

    def run(self) -> None:
        self.main_panel.setVisible(True)
        sys.exit(self.app.exec_())

    def get_group0(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Load")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("Load Image")
        button1.clicked.connect(self.__choose_img_and_read)
        button2 = QtWidgets.QPushButton("Load Video")
        button2.clicked.connect(self.__choose_vid)

        layout.addWidget(button1)
        layout.addWidget(button2)

        return group

    def get_group1(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("1.Background Subtraction")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("Background Subtraction")
        button1.clicked.connect(self.__handle_as1)

        layout.addWidget(button1)

        return group

    def get_group2(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("2.Optical Flow")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("Preprocessing")
        button1.clicked.connect(lambda: self.image_window.show_image("Preprocessing", 0,
                                                                     self.assign2.preprocess()))
        button2 = QtWidgets.QPushButton("Video tracking")
        button2.clicked.connect(self.__handle_track)

        layout.addWidget(button1)
        layout.addWidget(button2)

        return group

    def get_group3(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("3.PCA")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("Dimension Reduction")
        button1.clicked.connect(self.__handle_pca)

        layout.addWidget(button1)

        return group

    def get_group4(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("4.MNIST Classifier Using VGG19")
        layout_h = QtWidgets.QHBoxLayout(group)
        wid_1 = QtWidgets.QWidget()
        layout_h.addWidget(wid_1)
        layout_1 = QtWidgets.QVBoxLayout(wid_1)
        wid_2 = QtWidgets.QWidget()
        layout_h.addWidget(wid_2)
        layout_2 = QtWidgets.QVBoxLayout(wid_2)

        button1 = QtWidgets.QPushButton("4.1.Show Model Structure")
        button1.clicked.connect(lambda: print(self.assign4.show_model()))
        button2 = QtWidgets.QPushButton("4.2.Show Accuracy and loss")
        button2.clicked.connect(lambda: self.image_window.show_image("4.2", 0,
                                                                     self.assign4.show_accuracy_loss()))
        button3 = QtWidgets.QPushButton("4.3.Predict")
        button3.clicked.connect(self.__handle_inference_4)
        button4 = QtWidgets.QPushButton("4.4.Reset")
        button4.clicked.connect(self.__handle_reset)
        self.predict_label = label1 = QtWidgets.QLabel()
        label1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label1.setText("-")

        layout_1.addWidget(button1)
        layout_1.addWidget(button2)
        layout_1.addWidget(button3)
        layout_1.addWidget(button4)
        layout_1.addWidget(label1)
        layout_2.addWidget(self.pix_area)

        return group

    def get_group5(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("5.Resnet50")
        layout_h = QtWidgets.QHBoxLayout(group)
        wid_1 = QtWidgets.QWidget()
        layout_h.addWidget(wid_1)
        layout_1 = QtWidgets.QVBoxLayout(wid_1)
        wid_2 = QtWidgets.QWidget()
        layout_h.addWidget(wid_2)
        layout_2 = QtWidgets.QVBoxLayout(wid_2)

        button1 = QtWidgets.QPushButton("5.1.Show Images")
        button1.clicked.connect(self.__handle_show_5)
        button2 = QtWidgets.QPushButton("5.2.Show Model Structure")
        button2.clicked.connect(lambda: print(self.assign5.show_model()))
        button3 = QtWidgets.QPushButton("5.3.Show Comparison")
        button3.clicked.connect(self.__handle_comparison)
        button4 = QtWidgets.QPushButton("5.4.Inference")
        button4.clicked.connect(self.__handle_inference_5)
        self.preview_label = label1 = QtWidgets.QLabel()
        label1.setFixedSize(256, 256)
        self.result_label_5 = label2 = QtWidgets.QLabel()

        layout_1.addWidget(button1)
        layout_1.addWidget(button2)
        layout_1.addWidget(button3)
        layout_1.addWidget(button4)
        layout_2.addWidget(label1)
        layout_2.addWidget(label2)

        return group

    def __choose_img_and_read(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_panel, "choose image", self.cwd)
        self.img_wrapper.set_path(chosen)
        image = cv2.resize(cv2.cvtColor(self.img_wrapper.read(), cv2.COLOR_BGR2RGB), (256, 256))
        h, w, d = image.shape
        pixmap = QtGui.QPixmap(QtGui.QImage(
            image.data.tobytes(), w, h, w * d, QtGui.QImage.Format_RGB888))
        self.preview_label.setPixmap(pixmap)

    def __choose_vid(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_panel, "choose video", self.cwd)
        self.vid_wrapper.set_path(chosen)

    def __handle_as1(self) -> None:
        fps, videos = self.assign1.background_subtract()
        interval = 1 / fps
        for idx, vid in enumerate(videos):
            vid_iter = hwutil.CircularSequenceIter(vid)
            self.image_window.show_interval("background subtraction", idx, vid_iter.next, interval)

    def __handle_track(self) -> None:
        fps, vid = self.assign2.video_tracking()
        interval = 1 / fps
        vid_iter = hwutil.CircularSequenceIter(vid)
        self.image_window.show_interval("video tracking", 0, vid_iter.next, interval)

    def __handle_pca(self) -> None:
        outstr, outimg1, outimg2 = self.assign3.pca()
        print(outstr)
        self.image_window.show_image_all("pca", (outimg1, outimg2))

    def __handle_inference_4(self) -> None:
        q_image = self.pix_area.pixmap.toImage()
        width = q_image.width()
        height = q_image.height()
        ptr = q_image.bits()
        ptr.setsize(q_image.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)
        in_img = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        predicted, predicted_class = self.assign4.show_inference(in_img)
        fig = pyplot.figure()
        pyplot.bar([str(i) for i in range(10)], predicted)
        self.plot_window.show_plot("4.3", fig)
        self.predict_label.setText(predicted_class)

    def __handle_reset(self) -> None:
        self.pix_area.clear()
        self.predict_label.setText("-")

    def __handle_show_5(self) -> None:
        (cat, dog) = self.assign5.show_images()

        fig = pyplot.figure()
        cat_plot = fig.add_subplot(1, 2, 1)
        cat_plot.set_title("cat")
        pyplot.imshow(cv2.cvtColor(cv2.resize(cat, (224, 224)), cv2.COLOR_BGR2RGB))
        pyplot.axis("off")
        dog_plot = fig.add_subplot(1, 2, 2)
        dog_plot.set_title("dog")
        pyplot.imshow(cv2.cvtColor(cv2.resize(dog, (224, 224)), cv2.COLOR_BGR2RGB))
        pyplot.axis("off")
        self.plot_window.show_plot("5.1", fig)

    def __handle_comparison(self) -> None:
        result = self.assign5.show_comparison()
        fig, axes = pyplot.subplots()
        bars = axes.bar(("without random erasing", "with random erasing"), result)
        axes.bar_label(bars)
        self.plot_window.show_plot("5.3", fig)

    def __handle_inference_5(self) -> None:
        result = self.assign5.show_inference()
        self.result_label_5.setText(result)


class DrawBoard(QtWidgets.QWidget):
    pixmap: Final[QtGui.QPixmap]
    background: QtGui.QColor
    prev_point: Optional[QtCore.QPoint]
    cur_point: Optional[QtCore.QPoint]
    pen: QtGui.QPen

    def __init__(self, size: tuple[int, int],
                 background: QtGui.QColor = QtGui.QColor(255, 255, 255),
                 pen_color: QtGui.QColor = QtGui.QColor(0, 0, 0),
                 pen_width: float = 1.0) -> None:
        super().__init__()
        super().setMinimumSize(size[0], size[1])
        self.pixmap = QtGui.QPixmap(size[0], size[1])
        self.pixmap.fill(background)
        self.background = background
        self.prev_point = None
        self.cur_point = None
        self.pen = QtGui.QPen(pen_color, pen_width)
        super().setVisible(True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if self.prev_point is not None and self.cur_point is not None:
            painter = QtGui.QPainter(self.pixmap)
            painter.setPen(self.pen)
            painter.drawLine(self.prev_point, self.cur_point)
        main_painter = QtGui.QPainter(self)
        main_painter.drawPixmap(0, 0, self.pixmap)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.prev_point = self.cur_point
            self.cur_point = QtCore.QPoint(event.x(), event.y())
            super().update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        self.prev_point = None
        self.cur_point = None

    def clear(self) -> None:
        self.pixmap.fill(self.background)
        super().update()


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
        self.__clear_label()

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

    def show_image(self, title: str, index: int, img: np.ndarray) -> None:
        self.__post_img_idx(index, img)
        self.__display(title)

    def show_image_all(self, title: str, images: Iterable[np.ndarray]) -> None:
        self.__clear_thread()
        self.__clear_label()
        for index, img in enumerate(images):
            self.__post_img_idx(index, img)
        self.__display(title)

    def show_image_func(self, title: str, index: int, img_func: Callable[[], np.ndarray]) -> None:
        image = img_func()
        self.__post_img_idx(index, image)
        self.__display(title)

    def show_image_func_all(self, title: str, img_func: Callable[[], Iterable[np.ndarray]]) -> None:
        self.__clear_thread()
        self.__clear_label()
        images = img_func()
        for index, img in enumerate(images):
            self.__post_img_idx(index, img)
        self.__display(title)

    def show_interval(self, title: str, index: int, img_func: Callable[[], np.ndarray], interval: float | int) -> None:
        thread = hwutil.Interval(interval, lambda: self.__post_img_func_idx(index, img_func))
        self.work_threads.append(thread)
        thread.start_instant()

        self.__display(title)

    def show_interval_all(self, title: str, img_func: Callable, interval: float | int) -> None:
        self.__clear_thread()
        self.__clear_label()
        thread = hwutil.Interval(interval, lambda: self.__post_img_func(img_func))
        self.work_threads.append(thread)
        thread.start_instant()

        self.__display(title)

    def clear(self) -> None:
        self.__clear_thread()
        self.__clear_label()

    def display(self, title: str) -> None:
        self.__display(title)

    def refresh(self, index: int, img: np.ndarray) -> None:
        self.__post_img_idx(index, img)

    def add_label_event(self, index: int, func: Callable) -> None:
        self.labels[index].connect_event(func)

    def remove_label_event(self, index: int) -> None:
        self.labels[index].remove_event()
