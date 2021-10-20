from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import cv2
import numpy as np

import ui


def debug_log(msg):
    print(msg)


def SIFT_point(img):
    '''
        ref: https://www.itread01.com/content/1547651741.html
        created on 08:05:10 2018-11-20
        @author ren_dong

        使用DoG和SIFT進行特徵提取和描述

        cv2.SIFT.detectAndCompute(image, mask[, descriptors[, useProvidedKeypoints]]) → keypoints, descriptors

        cv2.drawKeypoints(image, keypoints[, outImage[, color[, flags]]]) → outImage

        首先建立了一個SIFT物件，SIFT物件會使用DoG檢測關鍵點，並且對每個關鍵點周圍區域計算特徵向量。
        detectAndCompute()函式會返回關鍵點資訊(每一個元素都是一個物件，有興趣的可以看一下OpenCV原始碼)和關鍵點的描述符。
        然後，我們在影象上繪製關鍵點，並顯示出來。
    '''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Color Space Conversions

    # 建立sift物件
    # sift得到的影象為128維的特徵向量集
    sift = cv2.xfeatures2d.SIFT_create()

    # 進行檢測和計算  返回特徵點資訊和描述符
    keypoints, descriptor = sift.detectAndCompute(gray, None) # list, ndarray(num of keypoint * 128)
    # keypoints：特徵點集合list，向量內每一個元素是一個KeyPoint物件，包含了特徵點的各種屬性資訊；

    temp = list(zip(keypoints, descriptor))
    temp.sort(key=lambda item: item[0].size, reverse=True)  # sort by keypoint size
    keypoints = [item[0] for item in temp][:200]
    descriptor = np.stack([item[1] for item in temp][:200])

    # 繪製關鍵點
    img_ = img.copy()
    cv2.drawKeypoints(img, keypoints=keypoints[:200], outImage=img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

    # cv2.imshow('sift_keypoints', img)

    return img, keypoints, descriptor


class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
            super().__init__()
            self.setupUi(self)

            # link button click function
            self.Q1_1.clicked.connect(self.corner_detection)
            self.Q1_2.clicked.connect(self.find_intrinsic)
            self.Q1_3.clicked.connect(self.find_extrinsic)
            self.Q1_4.clicked.connect(self.find_distortion)
            self.Q1_5.clicked.connect(self.undistorted)
            self.Q2_1.clicked.connect(self.word_lie)
            self.Q2_2.clicked.connect(self.word_stand)
            self.Q3_1.clicked.connect(self.disparity_map)
            # self.Q3_2.clicked.connect(self.disparity_match)
            self.Q4_1.clicked.connect(self.find_keypoints)
            self.Q4_2.clicked.connect(self.keypoints_match)
            self.Q4_3.clicked.connect(self.warp)

            # load image
            self.img1 = []
            for idx in range(1, 16):
                self.img1 += [cv2.imread(f'Dataset/Q1_Image/{idx}.bmp')]

            self.img4_1 = cv2.imread('Dataset/Q4_Image/Shark1.jpg')
            self.img4_2 = cv2.imread('Dataset/Q4_Image/Shark2.jpg')

            # global variable
            self.corners = []   # Q1


    # Q 1.1
    def corner_detection(self):
        # col or row numbers of corners
        pattern_size = (11, 8)

        for idx in range(0, 15):
            img = self.img1[idx]
            is_found, corners = cv2.findChessboardCorners(img, pattern_size)
            cv2.drawChessboardCorners(img, pattern_size, corners, is_found)
            self.corners += [corners]

            win_name = "Q 1.1"
            cv2.namedWindow(win_name, 0)
            cv2.resizeWindow(win_name, 512, 512)
            cv2.moveWindow(win_name, self.geometry().x() + 300, self.geometry().y())
            cv2.imshow(win_name, img)
            cv2.waitKey(500)    # wait for 500 ms

        cv2.waitKey(500)
        cv2.destroyWindow(win_name)

        # ref: https://docs.opencv.org/4.5.3/dc/dbb/tutorial_py_calibration.html
        # only need "pattern" to find distortion matrix then the intrinsic matrix (camera matrix) will also appear
        # so, the object points can use "pattern" form to describe the chess board

        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = [objp for _ in range(15)]   # 3d point in real world space
        _, self.intrinsic_mtx, self.dist_coeffs, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(objpoints, self.corners, self.img1[0].shape[:-1], None, None)

    # Q 1.2
    def find_intrinsic(self):
        print("Q 1.2 intrinsic matrix:")
        print(self.intrinsic_mtx)
        print("------------------------------------------------\n")

    # Q 1.3
    def find_extrinsic(self):
        idx = self.img_idx.value() - 1

        R, _ = cv2.Rodrigues(self.rvecs[idx])
        result = np.append(R, self.tvecs[idx], axis=1)
        print(f"Q 1.3 extrinsic matrix of image {idx + 1}")
        print(result)
        print("------------------------------------------------\n")

    # Q 1.4
    def find_distortion(self):
        print("Q 1.4 distortion matrix:")
        print(self.dist_coeffs)
        print("------------------------------------------------\n")
    
    # Q 1.5
    def undistorted(self):
        for idx in range(0, 15):
            img = self.img1[idx]
            h, w = img.shape[:2]

            img_ = cv2.undistort(img, self.intrinsic_mtx, self.dist_coeffs)

            win_name = "Q 1.5"
            cv2.namedWindow(win_name, 0)
            cv2.resizeWindow(win_name, 1024, 512)
            cv2.moveWindow(win_name, self.geometry().x() + 300, self.geometry().y())
            result = np.zeros((h, w + w, 3), dtype="uint8")
            result[0:h, 0:w] = img
            result[0:h, w:] = img_
            cv2.imshow(win_name, result)
            cv2.waitKey(500)    # wait for 500 ms

        cv2.waitKey(500)
        cv2.destroyWindow(win_name)

    # Q 2.1
    def word_lie(self):
        pass

    # Q 2.2
    def word_stand(self):
        pass

    # Q 3.1
    def disparity_map(self):
        pass

    # Q 3.2
    def disparity_match(self):
        pass

    # Q 4.1
    def find_keypoints(self):

        img, kps_1, feature_1 = SIFT_point(self.img4_1)
        cv2.imshow("Q 4.1_1", img)
        # Qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        # self.result.setPixmap(QtGui.QPixmap.fromImage(Qimg))

        img, kps_2, feature_2 = SIFT_point(self.img4_2)
        cv2.imshow("Q 4.1_2", img)

        # Qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        # self.result_2.setPixmap(QtGui.QPixmap.fromImage(Qimg))

        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.kps_1 = kps_1
        self.kps_2 = kps_2


    # Q 4.2
    def keypoints_match(self):
        # ref:https://chtseng.wordpress.com/2017/05/22/%E5%9C%96%E5%83%8F%E7%89%B9%E5%BE%B5%E6%AF%94%E5%B0%8D%E4%BA%8C-%E7%89%B9%E5%BE%B5%E9%BB%9E%E6%8F%8F%E8%BF%B0%E5%8F%8A%E6%AF%94%E5%B0%8D/

        # DescriptorMatcher_create是用來建立一個執行特徵點匹配運算的實體。如果兩個特徵點的distance愈小，我們就認為它們愈近似。
        matcher = cv2.DescriptorMatcher_create("BruteForce")

        # 使用KNN，從兩組Local features中兩兩最近似的成對放置為一組（K參數=2）。
        # matches為稍後要放置符合要求的keypoints陣列。
        rawMatches = matcher.knnMatch(self.feature_2, self.feature_1, 2)

        # 逐一取出已配對的keypoints，若距離差異小於0.8倍，則認定為符合的關鍵點
        # 放入matches陣列，此方式稱為David Lowe’s ratio test，可排除掉不適合的match。（一般建議為0.7~0.8倍）
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        self.matches = matches

        # 將first圖和second圖放置於同一張。
        (hA, wA) = self.img4_2.shape[:2]
        (hB, wB) = self.img4_1.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = self.img4_2
        vis[0:hB, wA:] = self.img4_1

        # 繪製所有放置於matches陣列中，兩點的直線
        for (trainIdx, queryIdx) in matches:

            ptA = (int(self.kps_2[queryIdx].pt[0]), int(self.kps_2[queryIdx].pt[1]))
            ptB = (int(self.kps_1[trainIdx].pt[0] + wA), int(self.kps_1[trainIdx].pt[1]))

            cv2.line(vis, ptA, ptB, 256, 1)

        cv2.imshow("Q 4.2", vis)

    # Q 4.3
    def warp(self):
        # ref:https://blog.csdn.net/qq_36387683/article/details/98446442

        (hA, wA) = self.img4_2.shape[:2]
        (hB, wB) = self.img4_1.shape[:2]

        src_pts = np.float32([self.kps_1[m[0]].pt for m in self.matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kps_2[m[1]].pt for m in self.matches]).reshape(-1, 1, 2)

        warpPerspective_mat, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # debug_log(warpPerspective_mat)

        # temp = np.zeros((hA, wA+wB, 3), dtype="uint8")
        # temp[0:hB, 0:wB] = self.img4_1


        temp = cv2.warpPerspective(self.img4_2, warpPerspective_mat, (wA+wB, hA))
        temp[0:hB, 0:wB] = self.img4_1
        cv2.imshow("", temp)
        # cv2.imshow("Q 4.3", temp_)

    # Q 5


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())