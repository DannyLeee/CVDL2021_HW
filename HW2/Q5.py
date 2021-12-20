from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *

import torch
import torch.nn as nn
from pytorch_model_summary import summary
from torchvision.models import resnet50
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import Q5_ui


class Main(QMainWindow, Q5_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # link button click function
        self.Q5_1.clicked.connect(self.show_model)
        self.Q5_2.clicked.connect(self.show_curve)
        self.Q5_3.clicked.connect(self.test)
        self.Q5_4.clicked.connect(self.show_augment)

    def show_model(self):
        result = "Model Structure\n"
        result += summary(ResNet_w_cls(), torch.zeros((1, 3, 32, 32)), max_depth=2)
        self.result.setText(result)

    def show_curve(self):
        img = mpimg.imread('curve.png')
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def test(self):
        model.eval()
        idx = self.img_index.value() - 1
        img = test_set[idx][0]

        with torch.no_grad():
            logits = model(img.unsqueeze(0))
            logits = logits.squeeze()
            cls = torch.argmax(logits)

        cls = "cat" if cls == 0 else "dog"
        plt.title(f"Class: {cls}")
        plt.axis("off")
        img = img.permute(1, 2, 0)
        imshow(img)

        plt.show()

    def show_augment(self):
        acc = [78.76, 90.44]
        mode = ["Before Random-Erasing", "After Random-Erasing"]
        x = np.arange(len(mode))
        plt.bar(x, acc, width=0.3)
        plt.xticks(x, mode)
        plt.ylim(70, 100)
        plt.ylabel('Accuracy')
        plt.title('Q 5.4')
        plt.show()


# model from torchvision
class ResNet_w_cls(nn.Module):
    def __init__(self):
        super(ResNet_w_cls, self).__init__()
        self.m = nn.Sequential(
            resnet50(),
            nn.Linear(1000, 2)
        )

    def forward(self, x):
        return self.m(x)

# hyper-parameter
BATCH_SIZE = 16
EPOCH = 10
LR = 1e-3

# model
model = ResNet_w_cls()
model.load_state_dict(torch.load(f"model/aug/e_8.pt", map_location=torch.device('cpu')))

# Dataset from torchvision
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor()])
data_set = datasets.ImageFolder("Dataset/Q5_Image/PetImages", transform=transform)
_, _, test_set = random_split(data_set, [20000, 2498, 2500],
                              generator=torch.Generator().manual_seed(87))
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE * 2)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())
