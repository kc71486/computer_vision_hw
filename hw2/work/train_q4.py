"""
generated and modified from colab
"""

import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision
import torchinfo

class FakeVGG19(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1)) # set pool = (1, 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 1 * 1, 4096), # set in_features = 512
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(4096, 10), # set out = 10
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batchloss = 0
    batchaccuracy = 0
    totalloss = 0
    totalcorrect = 0

    model.train()
    for batch, (X, Y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        Y = Y.to(device)
        pred = model(X)
        loss = loss_fn(pred, Y)
        batchloss += loss.item()
        totalloss += loss.item()
        accuracy = (pred.argmax(dim=1) == Y.argmax(dim=1)).type(torch.float).sum().item()
        batchaccuracy += accuracy
        totalcorrect += accuracy
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 99:
            batchloss /= 100
            batchaccuracy /= 3200
            current = (batch + 1) * len(X)
            print(f"Accuracy: {(100*batchaccuracy):>0.1f}%, loss: {batchloss:>7f}  [{current:>5d}/{size:>5d}]")
            batchloss = 0
            batchaccuracy = 0

    totalloss /= num_batches
    totalaccuracy = totalcorrect / size

    print(f"Train Error: \n Accuracy: {(100*totalaccuracy):>0.1f}%, Avg loss: {totalloss:>8f} \n")

    return totalloss, totalaccuracy

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, Y).item()
            # correct += (pred.argmax(1) == Y).type(torch.float).sum().item()
            correct += (pred.argmax(dim=1) == Y.argmax(dim=1)).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy = correct / size

    print(f"Validation Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, accuracy


def main():
    train_data = torchvision.datasets.MNIST(".",
                                            download=True,
                                            train=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(size=32),
                                                torchvision.transforms.RandomCrop(size=32, padding=2, pad_if_needed=True),
                                                torchvision.transforms.RandomRotation(degrees=30),
                                                torchvision.transforms.Grayscale(num_output_channels=3),
                                                torchvision.transforms.ToTensor(),
                                            ]),
                                            target_transform=torchvision.transforms.Compose([
                                                lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1),
                                            ])
                                            )
    test_data = torchvision.datasets.MNIST(".",
                                            download=False,
                                            train=False,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(size=32),
                                                torchvision.transforms.Grayscale(num_output_channels=3),
                                                torchvision.transforms.ToTensor(),
                                            ]),
                                            target_transform=torchvision.transforms.Compose([
                                                lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1),
                                            ])
                                            )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    model = FakeVGG19()
    # pth_filename = "VGG19_v3_Epoch_50.pth"
    # model.load_state_dict(torch.load(pth_filename))

    learning_rate = 0.001
    batch_size = 32
    epochs = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.to(device)

    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss, accuracy = train_loop(train_loader, model, loss_fn, optimizer)
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        loss, accuracy = test_loop(test_loader, model, loss_fn)
        test_loss.append(loss)
        test_accuracy.append(accuracy)
        if t % 5 == 4:
          torch.save(model.state_dict(), f"VGG19_v7_Epoch_{t+1}.pth")
    print("Done!")

    plt.subplot(2, 1, 1)
    plt.ylabel('train / validation loss')
    plt.xlabel('epochs')
    plt.plot([*range(1, epochs + 1)], train_loss, 'r', [*range(1, epochs + 1)], test_loss, 'b')

    plt.subplot(2, 1, 2)
    plt.ylabel('train / validation accuracy')
    plt.xlabel('epochs')
    plt.plot([*range(1, epochs + 1)], train_accuracy, 'r', [*range(1, epochs + 1)], test_accuracy, 'b')

    plt.tight_layout()
    plt.savefig('vgg19.png')


if __name__ == "__main__":
    main()
