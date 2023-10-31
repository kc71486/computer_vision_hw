import sys
import os

import numpy as np

from PyQt5 import QtWidgets, QtGui

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from multimethod import multimethod
from typing import Any, Union, Callable, Iterable


import backend
import hwutil

class MainWindow():
    app = None
    mainpanel = None
    imagewindow = None

    outfile = None
    cwd = None
    imageloader = None
    leftwrapper = None
    rightwrapper = None
    assign = [None] * 5

    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        
        self.cwd = os.getcwd()
        self.imageloader = hwutil.ImageLoader()
        self.leftwrapper = hwutil.ImageWrapper()
        self.rightwrapper = hwutil.ImageWrapper()

        self.mainpanel = QtWidgets.QWidget()
        self.mainpanel.setWindowTitle("Main Window")
    
        mainlayout = QtWidgets.QGridLayout(self.mainpanel)
        mainlayout.addWidget(self.getGroup0(), 0, 0)
        mainlayout.addWidget(self.getGroup1(), 0, 1)
        mainlayout.addWidget(self.getGroup2(), 0, 2)
        mainlayout.addWidget(self.getGroup3(), 0, 3)
        mainlayout.addWidget(self.getGroup4(), 1, 1)
        mainlayout.addWidget(self.getGroup5(), 1, 2)

        self.imagewindow = ImageWindow((1, 3))
        self.outfile = OutFile()

    def run(self):
        self.mainpanel.setVisible(True)
        sys.exit(self.app.exec_())
        
    def getGroup0(self):
        group = QtWidgets.QGroupBox("Load Image")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("Load Folder")
        button1.clicked.connect(self.chooseFolder)

        button2 = QtWidgets.QPushButton("Load Image_L")
        button2.clicked.connect(self.chooseLeft)

        button3 = QtWidgets.QPushButton("Load Image_R")
        button3.clicked.connect(self.chooseRight)

        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)

        return group

    def getGroup1(self):
        self.assign[0] = backend.Assign1(self.imageloader)

        group = QtWidgets.QGroupBox("Calibration")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("1.1 Find corners")
        button1.clicked.connect(
                lambda :self.imagewindow.showInterval("1.1",
                        self.assign[0].loop1, 2))

        button2 = QtWidgets.QPushButton("1.2 Find intrinsic")
        button2.clicked.connect(
                lambda :self.outfile.print(self.assign[0].findIntrinsic()))
    
        group3 = QtWidgets.QGroupBox("Find extrinsic")
        layout3 = QtWidgets.QVBoxLayout(group3)

        spinbox3 = QtWidgets.QSpinBox()
    
        button3 = QtWidgets.QPushButton("1.3 Find extrinsic")
        button3.clicked.connect(
                lambda :self.outfile.print(self.assign[0].findExtrinsic(
                        convertIndex(spinbox3.value(),self.imageloader.files, "bmp"))))
    
        layout3.addWidget(spinbox3)
        layout3.addWidget(button3)

        button4 = QtWidgets.QPushButton("1.4 Find distortion")
        button4.clicked.connect(
                lambda :self.outfile.print(self.assign[0].findDistortion()))
    
        button5 = QtWidgets.QPushButton("1.5 Show result")
        button5.clicked.connect(
                lambda :self.imagewindow.showInterval("1.5", 
                        lambda :self.assign[0].loop5(), 2))
    
        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(group3)
        layout.addWidget(button4)
        layout.addWidget(button5)

        return group

    def getGroup2(self):
        self.assign[1] = backend.Assign2(self.imageloader)

        group = QtWidgets.QGroupBox("Augmented Reality")
        layout = QtWidgets.QVBoxLayout(group)
       
        input0 = QtWidgets.QLineEdit()

        button1 = QtWidgets.QPushButton("2.1 Show words on board")
        button1.clicked.connect(
                lambda :self.imagewindow.show("2.1",
                        self.assign[1].arBoard(2, input0.text()))) 

        button2 = QtWidgets.QPushButton("2.2 Show words vertical")
        
        layout.addWidget(input0)
        layout.addWidget(button1)
        layout.addWidget(button2)

        return group

    def getGroup3(self) -> QtWidgets.QGroupBox:
        self.assign[2] = backend.Assign3(self.leftwrapper, self.rightwrapper)
        group = QtWidgets.QGroupBox("Stereo disparity map")
        layout = QtWidgets.QVBoxLayout(group)
        
        button1 = QtWidgets.QPushButton("3.1 Stereo disparity map")
        button1.clicked.connect(self.clickEventQ3)

        layout.addWidget(button1)

        return group

    def getGroup4(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("SIFT")
        layout = QtWidgets.QVBoxLayout(group)
        
        button1 = QtWidgets.QPushButton("Load Image1")

        button2 = QtWidgets.QPushButton("Load Image2")

        button3 = QtWidgets.QPushButton("4.1 Keypoints")
        
        button4 = QtWidgets.QPushButton("4.2 Matched Keypoints")
        
        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)
        layout.addWidget(button4)
        
        return group

    def getGroup5(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("VGG19")
        layout = QtWidgets.QVBoxLayout(group)
        
        button1 = QtWidgets.QPushButton("Load Image")

        button2 = QtWidgets.QPushButton("5.1 Show Agumented Images")

        button3 = QtWidgets.QPushButton("5.2 Show Model Structure")
        
        button4 = QtWidgets.QPushButton("5.3 Show Acc and Loss")
        
        button5 = QtWidgets.QPushButton("5.4 Inference")
        
        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)
        layout.addWidget(button4)
        layout.addWidget(button5)

        return group

    def chooseFolder(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self.mainpanel, "choose folder", self.cwd)
        self.imageloader.setPath(chosen)

    def chooseLeft(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self.mainpanel, "choose file", self.cwd)
        self.leftwrapper.setPath(chosen)

    def chooseRight(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self.mainpanel, "choose file", self.cwd)
        self.rightwrapper.setPath(chosen)

    def clickEventQ3(self) -> None:
        self.imagewindow.show("3.1", 
                    lambda :self.assign[2].disparityValue((700, 400)))

class ImageWindow(QtWidgets.QWidget):
    labels = []
    workthreads = []

    @multimethod
    def __init__(self, gridsize:tuple[int,int]):
        super().__init__()
        layout = QtWidgets.QGridLayout(self)
        for i in range(gridsize[0]):
            for j in range(gridsize[1]):
                label = QtWidgets.QLabel(self)
                self.labels.append(label)
                label.mousePressEvent = None
                layout.addWidget(label, i, j)
        self.workthreads = []
    
    @multimethod
    def __clearThread(self) -> None:
        for thread in self.workthreads:
            thread.cancel()
        self.workthreads = []
        for index, label in enumerate(self.labels):
            if label.mousePressEvent is not None:
                self.lebel[index].mousePressEvent = None

    @multimethod
    def closeEvent(self, event:QtGui.QCloseEvent) -> None:
        super().closeEvent(event)
        self.__clearThread()
    
    @multimethod
    def __postImg(self, index:int, img:np.ndarray) -> None:
        h, w, d = img.shape
        pixmap = QtGui.QPixmap(QtGui.QImage(
                img.data, w, h, w * d, QtGui.QImage.Format_RGB888))
        self.labels[index].setPixmap(pixmap)
    
    @multimethod
    def __postImgFunc(self, index:int, imgfunc:Callable[[Any],np.ndarray]) -> None:
        img = imgfunc()
        h, w, d = img.shape
        pixmap = QtGui.QPixmap(QtGui.QImage(
                img.data, w, h, w * d, QtGui.QImage.Format_RGB888))
        self.labels[index].setPixmap(pixmap)

    @multimethod
    def __postImg(self, imgfunc:Callable) -> None:
        imgs = imgfunc()
        for img in imgs:
            h, w, d = img.shape
            pixmap = QtGui.QPixmap(QtGui.QImage(
                    img.data, w, h, w * d, QtGui.QImage.Format_RGB888))
            self.labels[index].setPixmap(pixmap)
    
    @multimethod
    def __display(self, title:str) -> None: 
        super().setWindowTitle(title)
        super().setVisible(True)
   
    @multimethod
    def show(self, title:str, imgs:Iterable[np.ndarray]) -> None:
        self.__clearThread()
        for index, img in enumerate(imgs):
            self.__postImg(index, img)
        self.__display(title)
    
    @multimethod
    def show(self, title:str, imgfunc:Callable) -> None:
        self.__clearThread()
        imgs = imgfunc()
        for index, img in enumerate(imgs):
            self.__postImg(index, img)
        self.__display(title)

    @multimethod
    def show(self, title:str, img:np.ndarray) -> None:
        self.__clearThread()
        self.__postImg(0, img)
        self.__display(title)

    @multimethod
    def showInterval(self, title:str, imgfunc:Callable[[Any],np.ndarray], interval:float|int) -> None:
        self.__clearThread()
        self.workthreads.append(hwutil.SetInterval(interval, 
                lambda :self.__postImgFunc(0, imgfunc)))
        self.__display(title)

    @multimethod
    def refresh(self, index:int, img:np.ndarray) -> None:
        self.__removeThread(index)
        self.__postImg(index, img)

    @multimethod
    def addClickEvent(self, index:int, func:Callable) -> None:
        self.label[index].mousePressEvent = func

class OutFile():
    @multimethod
    def __init__(self) -> Any:
        pass

    @multimethod
    def print(self, obj:Any):
        print(obj)

@multimethod
def convertIndex(value, filepaths:Iterable[str], ext:str) -> Iterable[str]:
    dirname = os.path.dirname(filepaths[0])
    file = os.path.join(dirname, str(value) + "." + ext)
    return filepaths.index(file)

