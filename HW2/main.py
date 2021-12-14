from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

import ui


class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # link button click function
        self.Q4_1.clicked.connect(self.img_reconstruct)

        # load image
        self.Q4_img = []
        for i in range(1, 31):
            img = cv2.imread(f"Dataset/Q4_Image/{i}.jpg")
            self.Q4_img += [img]

    # Q 4.1
    def img_reconstruct(self):
        img = []
        for i in range(30):
            img += [self.Q4_img[i].reshape(-1)]
        x = np.stack(img)
        mu = np.mean(x, axis=0)
        pca = PCA()
        pca.fit(x)

        n_comp = 25
        x_hat = np.dot(pca.transform(x)[:, :n_comp], pca.components_[:n_comp, :])
        x_hat += mu
        x_hat = x_hat.reshape(30, 400, 400, 3).astype(int)

        plt.figure(1)
        for idx, img in enumerate(self.Q4_img):
            plt.subplot(4, 15, idx + 1 + (idx // 15) * 15)
            if idx == 0 or idx == 15:
                plt.ylabel('origin')
            else:
                plt.axis("off")
            imshow(img[:, :, ::-1])

            plt.subplot(4, 15, idx + 1 + 15 + (idx // 15) * 15)
            if idx == 0 or idx == 15:
                plt.ylabel('reconstruction')
            else:
                plt.axis("off")
            imshow(x_hat[idx][:, :, ::-1])
        plt.show()


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())