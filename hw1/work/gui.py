import sys
from PyQt5 import QtWidgets, QtGui

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import backend

class MainWindow():
    def __init__():
        pass
    def run():
        app = QtWidgets.QApplication(sys.argv)

        mainpanel = QtWidgets.QWidget()
        mainpanel.setWindowTitle("Main Window")
    
        mainlayout = QtWidgets.QGridLayout(mainpanel)
        mainlayout.addWidget(getGroup0(), 0, 0)
        mainlayout.addWidget(getGroup1(), 0, 1)
        mainlayout.addWidget(getGroup2(), 0, 2)
        mainlayout.addWidget(getGroup3(), 0, 3)
        mainlayout.addWidget(getGroup4(), 1, 1)
        mainlayout.addWidget(getGroup5(), 1, 2)

        mainpanel.setVisible(true)
        sys.exit(app.exec_())
        
    def getGroup0():
        group = QtWidgets.QGroupBox("Load Image")
        layout = QtWidgets.QVBoxLayout(group)

        return group

    def getGroup1():
        assign = backend.assign1()

        group = QtWidgets.QGroupBox("Calibration")
        layout = QtWidgets.QVBoxLayout(group)

        button1 = QtWidgets.QPushButton("1.1 Find corners", group)
        button1.clicked.connect(lambda: showImageWindow("1.1", assign.findCorner()))

        button2 = QtWidgets.QPushButton("1.2 Find intrinsic", group)
    
        group3 = QtWidgets.QGroupBox("Find extrinsic", group)
        layout3 = QtWidgets.QVBoxLayout(group3)

        spinbox3 = QtWidgets.QSpinBox(group3)
    
        button3 = QtWidgets.QPushButton("1.3 Find extrinsic", group3)
    
        layout3.addWidget(spinbox3)
        layout3.addWidget(button3)

        button4 = QtWidgets.QPushButton("1.4 Find distortion", group)
    
        button5 = QtWidgets.QPushButton("1.5 Show result", group)
    
        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(group3)
        layout.addWidget(button4)
        layout.addWidget(button5)

        return group

def getGroup2():
    group = QtWidgets.QGroupBox("Augmented Reality")
    layout = QtWidgets.QVBoxLayout(group)
   
    input0 = QtWidgets.QLineEdit(group)

    button1 = QtWidgets.QPushButton("2.1 Show words on board", group)

    button2 = QtWidgets.QPushButton("2.2 Show words vertical", group)
    
    layout.addWidget(input0)
    layout.addWidget(button1)
    layout.addWidget(button2)

    return group

def getGroup3():
    group = QtWidgets.QGroupBox("Stereo disparity map")
    layout = QtWidgets.QVBoxLayout(group)
    
    button1 = QtWidgets.QPushButton("3.1 Stereo disparity map", group)

    layout.addWidget(button1)

    return group

def getGroup4():
    group = QtWidgets.QGroupBox("SIFT")
    layout = QtWidgets.QVBoxLayout(group)
    
    button1 = QtWidgets.QPushButton("Load Image1", group)

    button2 = QtWidgets.QPushButton("Load Image2", group)

    button3 = QtWidgets.QPushButton("4.1 Keypoints", group)
    
    button4 = QtWidgets.QPushButton("4.2 Matched Keypoints", group)
    
    layout.addWidget(button1)
    layout.addWidget(button2)
    layout.addWidget(button3)
    layout.addWidget(button4)
    
    return group

def getGroup5():
    group = QtWidgets.QGroupBox("VGG19")
    layout = QtWidgets.QVBoxLayout(group)
    
    button1 = QtWidgets.QPushButton("Load Image", group)

    button2 = QtWidgets.QPushButton("5.1 Show Agumented Images", group)

    button3 = QtWidgets.QPushButton("5.2 Show Model Structure", group)
    
    button4 = QtWidgets.QPushButton("5.3 Show Acc and Loss", group)
    
    button5 = QtWidgets.QPushButton("5.4 Inference", group)
    
    layout.addWidget(button1)
    layout.addWidget(button2)
    layout.addWidget(button3)
    layout.addWidget(button4)
    layout.addWidget(button5)
    return group

class ImageWindow():
    def __init__(self):
        self.panel = QtWidgets.QWidget()
        self.panel.setWindowTitle(title)
        self.layout = QtWidgets.QVBoxLayout(self.panel)

        self.label = QtWidgets.QLabel(self.panel)

        self.layout.addWidget(label)
    def showImage(title, cvimg):
    
        h, w, d = cvimg.shape
        #pixmap = QtGui.QPixmap(QtGui.QImage(
        #        cvimg.data, w, h, w * d, QtGui.QImage.Format_RGB888))
        pixmap = QtGui.QPixmap(QtGui.QImage("../Dataset_CvDl_Hw1/Q1_Image/1.bmp"))

        self.label.setPixmap(pixmap)
    
        self.widget.setVisible(True)

