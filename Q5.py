from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *

import pickle
import torch
import torch.nn as nn
from pytorch_model_summary import summary
from torchvision.models import vgg
from torchvision.datasets import CIFAR10
from torchvision import transforms

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

import Q5_ui


class Main(QMainWindow, Q5_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # link button click function
        self.Q5_1.clicked.connect(self.show_dataset)
        self.Q5_2.clicked.connect(self.show_hyperparams)
        self.Q5_3.clicked.connect(self.show_model)
        self.Q5_4.clicked.connect(self.show_curve)
        self.Q5_5.clicked.connect(self.test)

    def show_dataset(self):
        id_ls = [-1 for _ in range(10)]
        flag = 0
        for i in range(50):
            if id_ls[train_set[i][1]] == -1:
                id_ls[train_set[i][1]] = i
                flag += 1
            if flag == 10:
                break

        plt.figure(1)
        for i, idx in enumerate(id_ls[:-1]):
            plt.subplot(3, 3, i + 1)
            plt.title(label_ls[i])
            plt.axis("off")
            img = train_set[idx][0]
            img = img.transpose(0, 1).transpose(1, 2)
            imshow(img)
        plt.show()

    def show_hyperparams(self):
        result = f"Hyper Parameters:\n\nBatch Size: {BATCH_SIZE}\n"
        result += str(optimizer)
        self.result.setText(result)

    def show_model(self):
        result = "Model Structure\n"
        result += summary(VGG_w_cls(), torch.zeros((1, 3, 32, 32)), max_depth=2)
        self.result.setText(result)

    def show_curve(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.plot(epoch_loss)

        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.ylabel("Acc (%)")
        plt.xlabel("epoch")
        plt.plot(epoch_acc)

        plt.show()

    def test(self):
        idx = self.img_index.value() - 1
        img = test_set[idx][0]

        with torch.no_grad():
            logits = model(img.unsqueeze(0))
            softmax = nn.Softmax()
            logits = softmax(logits)
            logits = logits.squeeze()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.title("Test Image")
        plt.axis("off")
        img = img.transpose(0, 1).transpose(1, 2)
        imshow(img)

        plt.subplot(1, 2, 2)

        plt.title("Logits")
        plt.xlabel("classes")
        plt.ylabel("Probability")
        plt.ylim([0.0, 1.0])
        plt.xticks(range(10), label_ls)
        plt.bar(range(10), logits)

        plt.show()


# model from torchvision
class VGG_w_cls(nn.Module):
    def __init__(self):
        super(VGG_w_cls, self).__init__()
        self.m = nn.Sequential(
            vgg.vgg16(),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        return self.m(x)


# hyper-parameter
BATCH_SIZE = 64
EPOCH = 25
LR = 1e-3
MOMENTUM = 0.9

# model and label
label_ls = ["airplane", "car", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"]
model = VGG_w_cls()
model.load_state_dict(torch.load(f"model/e_25", map_location=torch.device('cpu')))

# Dataset from torchvision
to_tensor = transforms.ToTensor()
train_set = CIFAR10("Dataset/Q5", train=True, transform=to_tensor)
test_set = CIFAR10("Dataset/Q5", train=False, transform=to_tensor)

# optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

# curve
with open('model/epoch_loss.pkl', 'rb') as fp:
    epoch_loss = pickle.load(fp)
with open('model/epoch_acc.pkl', 'rb') as fp:
    epoch_acc = pickle.load(fp)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())