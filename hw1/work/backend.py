import os
import cv2

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

class imageLoader():
    pass

class assign1():
    board_corner = (11, 8)
    def __init__(self):
        pass

    def findCorner(self):
        image = cv2.imread("../Dataset_CvDl_Hw1/Q1_Image/1.bmp")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("pressed")
        #cv2.drawChessboardCorners(img, self.board_corner, None)
        return image

    def findIntrinsic(self):
        pass

    def findExtrinsic():
        pass

    def findDistortion():
        pass

    def showUndistorted():
        pass

class assign2():

    def arBoard():
        pass

    def arVertical():
        pass

class assign3():

    def disparityMap():
        pass

    def disparityValue():
        pass

class assign4():

    def siftKeypoint():
        pass
    
    def siftMatch():
        pass

class assign5():

    def showAugment():
        pass

    def showModel():
        pass

    def showAccuracyLoss():
        pass
    
    def predictLabel():
        pass
