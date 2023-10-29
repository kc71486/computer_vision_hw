import sys
import os
from PyQt5 import QtWidgets, QtGui

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import backend
import hwutil

class MainWindow():
    app = None
    mainpanel = None
    imagewindow = None
    outfile = None
    cwd = None
    imageloader = None
    assign = [None] * 5

    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        
        self.cwd = os.getcwd()
        self.imageloader = hwutil.ImageLoader()

        self.mainpanel = QtWidgets.QWidget()
        self.mainpanel.setWindowTitle("Main Window")
    
        mainlayout = QtWidgets.QGridLayout(self.mainpanel)
        mainlayout.addWidget(self.getGroup0(), 0, 0)
        mainlayout.addWidget(self.getGroup1(), 0, 1)
        mainlayout.addWidget(self.getGroup2(), 0, 2)
        mainlayout.addWidget(self.getGroup3(), 0, 3)
        mainlayout.addWidget(self.getGroup4(), 1, 1)
        mainlayout.addWidget(self.getGroup5(), 1, 2)

        self.imagewindow = ImageWindow()
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

        button3 = QtWidgets.QPushButton("Load Image_R")

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
                lambda :self.imagewindow.showInterval("1.1", 2, 
                        lambda :self.assign[0].loopthrough(self.assign[0].findCorner)))

        button2 = QtWidgets.QPushButton("1.2 Find intrinsic")
        button2.clicked.connect(
                lambda :self.outfile.print(self.assign[0].findIntrinsic()))
    
        group3 = QtWidgets.QGroupBox("Find extrinsic")
        layout3 = QtWidgets.QVBoxLayout(group3)

        spinbox3 = QtWidgets.QSpinBox()
    
        button3 = QtWidgets.QPushButton("1.3 Find extrinsic")
        button3.clicked.connect(
                lambda :self.outfile.print(self.assign[0].findExtrinsic(0)))
    
        layout3.addWidget(spinbox3)
        layout3.addWidget(button3)

        button4 = QtWidgets.QPushButton("1.4 Find distortion")
        button4.clicked.connect(
                lambda :self.outfile.print(self.assign[0].findDistortion()))
    
        button5 = QtWidgets.QPushButton("1.5 Show result")
        button5.clicked.connect(
                lambda :self.imagewindow.showInterval("1.5", 2, 
                        lambda :self.assign[0].loopthrough(self.assign[0].showUndistorted)))
    
        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(group3)
        layout.addWidget(button4)
        layout.addWidget(button5)

        return group

    def getGroup2(self):
        group = QtWidgets.QGroupBox("Augmented Reality")
        layout = QtWidgets.QVBoxLayout(group)
       
        input0 = QtWidgets.QLineEdit()

        button1 = QtWidgets.QPushButton("2.1 Show words on board")

        button2 = QtWidgets.QPushButton("2.2 Show words vertical")
        
        layout.addWidget(input0)
        layout.addWidget(button1)
        layout.addWidget(button2)

        return group

    def getGroup3(self):
        group = QtWidgets.QGroupBox("Stereo disparity map")
        layout = QtWidgets.QVBoxLayout(group)
        
        button1 = QtWidgets.QPushButton("3.1 Stereo disparity map")

        layout.addWidget(button1)

        return group

    def getGroup4(self):
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

    def getGroup5(self):
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

    def chooseFolder(self):
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self.mainpanel, "choose folder", self.cwd)
        self.imageloader.setpath(chosen)

class ImageWindow(QtWidgets.QWidget):
    label = None
    workthread = None

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel(self)

        layout.addWidget(self.label)

    def closeEvent(self, event):
        super().closeEvent(event)
        if self.workthread is not None:
            self.workthread.cancel()
            self.workthread = None

    def postimg(self, imgfunc):
        img = imgfunc()
        h, w, d = img.shape
        pixmap = QtGui.QPixmap(QtGui.QImage(
                img.data, w, h, w * d, QtGui.QImage.Format_RGB888))
        self.label.setPixmap(pixmap)

    def display(self, title): 
        super().setWindowTitle(title)
        super().setVisible(True)

    def showSingle(self, title, imgfunc):
        self.postimg(imgfunc)
        self.display(title)

    def showInterval(self, title, interval, imgfunc):
        if self.workthread is not None:
            self.workthread.cancel()
        self.workthread = hwutil.SetInterval(interval, 
                lambda :self.postimg(imgfunc))
        self.display(title)

class OutFile():
    def __init__(self):
        pass

    def print(self, obj):
        print(obj)
        
