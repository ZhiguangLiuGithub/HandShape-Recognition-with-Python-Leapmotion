# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np

# 手の画像の読み込み
for hand_class in [0, 1, 2, 3, 4, 5]:

    # 画像番号0から999まで対応
    for i in range(900):
        if hand_class==0:
            filename = 'hand-learn2/img_zero{0:03d}.png'.format(i)
        elif hand_class==1:
            filename = 'hand-learn2/img_one{0:03d}.png'.format(i)
        elif hand_class==2:
            filename = 'hand-learn2/img_two{0:03d}.png'.format(i)
        elif hand_class==3:
            filename = 'hand-learn2/img_three{0:03d}.png'.format(i)
        elif hand_class==4:
            filename = 'hand-learn2/img_four{0:03d}.png'.format(i)
        elif hand_class==5:
            filename = 'hand-learn2/img_five{0:03d}.png'.format(i)

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        print('{0}を読み込んでいます'.format(filename))

        if hand_class==0:
            cv2.imwrite('hand-learn22/img_zero{0:03d}.png'.format(i+100),img)
        elif hand_class==1:
            cv2.imwrite('hand-learn22/img_one{0:03d}.png'.format(i+100),img)
        elif hand_class==2:
            cv2.imwrite('hand-learn22/img_two{0:03d}.png'.format(i+100),img)
        elif hand_class==3:
            cv2.imwrite('hand-learn22/img_three{0:03d}.png'.format(i+100),img)
        elif hand_class==4:
            cv2.imwrite('hand-learn22/img_four{0:03d}.png'.format(i+100),img)
        elif hand_class==5:
            cv2.imwrite('hand-learn22/img_five{0:03d}.png'.format(i+100),img)
