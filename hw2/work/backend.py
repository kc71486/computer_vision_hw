import os

import numpy as np
import random

from typing import Optional, Final

import cv2

from sklearn.decomposition import PCA

from PIL import Image

import torch
import torchvision
from torch import nn
import torchinfo

import hwutil

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


class Assign1:
    videowrapper: hwutil.VideoWrapper

    def __init__(self, videowrapper: hwutil.VideoWrapper) -> None:
        self.videowrapper = videowrapper

    def background_subtract(self) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        video = self.videowrapper.read()
        fps = self.videowrapper.get_fps()

        mask = np.empty(video.shape, dtype=np.uint8)
        mixed = np.empty(video.shape, dtype=np.uint8)

        bg_subtract = cv2.createBackgroundSubtractorKNN(history=500,
                                                        dist2Threshold=400,
                                                        detectShadows=True)

        for (idx, frame) in enumerate(video):
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            applied = bg_subtract.apply(blurred)

            mask[idx] = cv2.cvtColor(applied, cv2.COLOR_GRAY2RGB).astype(np.uint8)
            mixed[idx] = cv2.bitwise_and(video[idx], mask[idx])

        return fps, (video, mask, mixed)


class Assign2:
    resize_ratio = 0.5
    videowrapper: hwutil.VideoWrapper
    init_point: np.ndarray

    def __init__(self, videowrapper: hwutil.VideoWrapper) -> None:
        self.videowrapper = videowrapper
        self.init_point = np.array((-1, -1))

    def preprocess(self) -> np.ndarray:
        video = self.videowrapper.read()

        image = video[0]

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=1, qualityLevel=0.5,
                                          minDistance=7, blockSize=7)

        cx, cy = self.init_point = corners[0].ravel().astype(np.uint32)
        cv2.line(image, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 4)
        cv2.line(image, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 4)

        imgsize = image.shape
        image = cv2.resize(image, (int(imgsize[1] * Assign2.resize_ratio), int(imgsize[0] * Assign2.resize_ratio)))

        return image

    def video_tracking(self) -> tuple[float, np.ndarray]:
        video = self.videowrapper.read()
        fps = self.videowrapper.get_fps()

        if self.init_point[0] == -1:
            self.preprocess()

        points = []
        line_image = np.zeros(video[0].shape, dtype=np.uint8)
        mixed = []

        first_frame = True
        for curimg in video:
            if first_frame:
                newpoint = self.init_point.reshape((1, 1, 2)).astype(np.float32)
            else:
                curgray = cv2.cvtColor(curimg, cv2.COLOR_RGB2GRAY)
                newpoint = cv2.calcOpticalFlowPyrLK(prevgray, curgray, points[-1], None)
                newpoint = newpoint[0].reshape((1, 1, 2))

            crossimg = np.copy(curimg)
            cx, cy = newpoint[0].ravel().astype(np.int32)
            if cx >= 0 and cy >= 0:
                cv2.line(crossimg, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 4)
                cv2.line(crossimg, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 4)

            if not first_frame:
                ax, ay = points[-1][0].flatten().astype(np.int32)
                bx, by = newpoint[0].flatten().astype(np.int32)
                line_image = cv2.line(line_image, (ax, ay), (bx, by), [224, 100, 0], 4)
                imgsize = line_image.shape
                mixed.append(cv2.resize(cv2.bitwise_or(crossimg, line_image),
                                        (int(imgsize[1] * Assign2.resize_ratio),
                                         int(imgsize[0] * Assign2.resize_ratio))))
            points.append(newpoint)
            prevgray = cv2.cvtColor(curimg, cv2.COLOR_RGB2GRAY)
            first_frame = False
        mixed = np.array(mixed)
        return fps, mixed


class Assign3:
    image_wrapper: hwutil.ImageWrapper

    def __init__(self, image_wrapper: hwutil.ImageWrapper) -> None:
        self.image_wrapper = image_wrapper

    def pca(self) -> tuple[str, np.ndarray, np.ndarray]:
        image_i = self.image_wrapper.read()
        image = cv2.cvtColor(image_i, cv2.COLOR_BGR2GRAY).astype(np.float32)
        image_1 = image / 255
        new_image = image
        out_str = ""
        for n in range(70, 100):
            pca = PCA(n_components=n)
            new_image_1 = pca.inverse_transform(pca.fit_transform(image_1))
            new_image = new_image_1 * 255
            mse = np.mean((image - new_image) ** 2)
            out_str += f"component = {n}, mse = {mse}\n"
            if mse <= 3.0:
                out_str += f"n = {n}"
                break

        new_image_i = cv2.cvtColor(new_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return out_str, image_i, new_image_i


class VGG19(nn.Module):
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # set pool = (1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),  # set in_features = 512
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Assign4:
    weight_path: Final[str] = "VGG19_epoch_50.pth"
    transform: torchvision.transforms.transforms
    model: Optional[VGG19]
    batches: Final[int]
    device: Final[torch.device]
    figure_image: Final[hwutil.ImageWrapper]

    def __init__(self) -> None:
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(32, 32)),
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
        ])
        self.model = None
        self.batches = 32
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.figure_image = hwutil.ImageWrapper("vgg19.png")

    def __yield_model(self) -> None:
        if self.model is None:
            self.model = VGG19()
            self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))

    def show_model(self) -> torchinfo.ModelStatistics:
        self.__yield_model()
        return torchinfo.summary(self.model, input_size=(self.batches, 3, 32, 32), mode="eval", verbose=0)

    def show_accuracy_loss(self) -> np.ndarray:
        return self.figure_image.read()

    def show_inference(self, image: np.ndarray) -> tuple[torch.Tensor, str]:
        self.__yield_model()

        cv_img = cv2.resize(image, dsize=(32, 32))
        image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image)
        tensor = torch.unsqueeze(tensor, dim=0)

        self.model.eval()
        with torch.no_grad():
            predicted = self.model(tensor)
        predicted = torch.squeeze(predicted)
        predicted = torch.nn.functional.softmax(predicted, dim=0)
        predicted_class = str(torch.argmax(predicted).item())

        return predicted, predicted_class


class Assign5:
    weight_path: Final[str] = "resnet50_epoch_50.pth"
    inference_path: Final[str] = "../inference_dataset"
    image_wrapper: hwutil.ImageWrapper
    transform: torchvision.transforms.transforms
    model: Optional[torchvision.models.ResNet]
    batches: Final[int]
    device: Final[torch.device]
    __accuracy: Final[tuple[float, float]] = (0.9466, 0.9454)

    def __init__(self, image_wrapper: hwutil.ImageWrapper) -> None:
        self.image_wrapper = image_wrapper
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
        ])
        self.model = None
        self.batches = 32
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __yield_model(self) -> None:
        if self.model is None:
            self.model = torchvision.models.resnet50()
            self.model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 1),
                                                torch.nn.Sigmoid())
            self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))

    @staticmethod
    def show_images() -> tuple[np.ndarray, np.ndarray]:
        cat_paths = []
        dog_paths = []
        cat_dir = os.path.join(Assign5.inference_path, "Cat")
        dog_dir = os.path.join(Assign5.inference_path, "Dog")
        for file in os.listdir(cat_dir):
            cat_file = os.path.join(cat_dir, file)
            cat_paths.append(cat_file)
        for file in os.listdir(dog_dir):
            dog_file = os.path.join(dog_dir, file)
            dog_paths.append(dog_file)
        cat_img = cv2.imread(random.choice(cat_paths))
        dog_img = cv2.imread(random.choice(dog_paths))
        return cat_img, dog_img

    def show_model(self) -> torchinfo.ModelStatistics:
        self.__yield_model()
        return torchinfo.summary(self.model, input_size=(self.batches, 3, 32, 32), mode="eval", verbose=0)

    def show_comparison(self) -> tuple[float, float]:
        return self.__accuracy

    def show_inference(self) -> str:
        self.__yield_model()

        image = self.image_wrapper.read()
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image)
        tensor = torch.unsqueeze(tensor, dim=0)

        self.model.eval()
        with torch.no_grad():
            predicted = self.model(tensor)
        predicted = torch.squeeze(predicted)
        predicted = "Cat" if (predicted > 0.5) else "Dog"

        return predicted
