# -*- coding: utf-8 -*-
import cv2, Leap, math, ctypes
import sys
import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from decimal import Decimal
import matplotlib.pyplot as plt

# 学習に用いる縮小画像のサイズ
sw = 160
sh = 120

# 学習済ファイルの確認
if len(sys.argv)==2:
    savefile = sys.argv[1]
    try:
        clf = joblib.load(savefile)
    except IOError:
        print('学習済ファイル{0}を開けません'.format(savefile))
        sys.exit()
else:
    print('使用法: python ml-08-04-recognition.py 学習済ファイル.pkl')
    sys.exit()

def getImageVector(img):
    # 白い領域(ピクセル値が0でない領域)の座標を集める
    nonzero = cv2.findNonZero(img)
    # その領域を囲う四角形の座標と大きさを取得
    xx, yy, ww, hh = cv2.boundingRect(nonzero)
    # 白い領域を含む最小の矩形領域を取得
    img_nonzero = img[yy:yy+hh, xx:xx+ww]
    # 白い領域を(sw, sh)サイズに縮小するための準備
    img_small = np.zeros((sh, sw), dtype=np.uint8)
    # 画像のアスペクト比を保ったまま、白い領域を縮小してimg_smallにコピーする
    if 4*hh < ww*3 and hh > 0:
        htmp = int(sw*hh/ww)
        if htmp>0:
            img_small_tmp = cv2.resize(img_nonzero, (sw, htmp), interpolation=cv2.INTER_LINEAR)
            img_small[(sh-htmp)//2:(sh-htmp)//2+htmp, 0:sw] = img_small_tmp
    elif 4*hh >= ww*3 and ww > 0:
        wtmp = int(sh*ww/hh)
        if wtmp>0:
            img_small_tmp = cv2.resize(img_nonzero, (wtmp, sh), interpolation=cv2.INTER_LINEAR)
            img_small[0:sh, (sw-wtmp)//2:(sw-wtmp)//2+wtmp] = img_small_tmp
    # 0...1の範囲にスケーリングしてからリターンする
    return np.array([img_small.ravel()/255.])

# X:画像から計算したベクトル、y:正解データ
testX = np.empty((0,sw*sh), float)
testy = np.array([], int)

# 手の画像の読み込み
for hand_class in [0, 1, 2, 3, 4, 5]:

    # 画像番号0から999まで対応
    for i in range(1000):
        if hand_class==0:
            filename = 'hand-learn5/img_zero{0:03d}.png'.format(i)
        elif hand_class==1:
            filename = 'hand-learn5/img_one{0:03d}.png'.format(i)
        elif hand_class==2:
            filename = 'hand-learn5/img_two{0:03d}.png'.format(i)
        elif hand_class==3:
            filename = 'hand-learn5/img_three{0:03d}.png'.format(i)
        elif hand_class==4:
            filename = 'hand-learn5/img_four{0:03d}.png'.format(i)
        elif hand_class==5:
            filename = 'hand-learn5/img_five{0:03d}.png'.format(i)

        my_hand = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if my_hand is None:
            continue
        #print('{0}を読み込んでいます'.format(filename))

        # 以下、最も広い白領域のみを残すための計算
        # まず、白領域の塊（クラスター）にラベルを振る
        img_dist, img_label = cv2.distanceTransformWithLabels(255-my_hand, cv2.DIST_L2, 5)
        img_label = np.uint8(img_label) & my_hand
        # ラベル0は黒領域なので除外
        img_label_not_zero = img_label[img_label != 0]
        # 最も多く現れたラベルが最も広い白領域のラベル
        if len(img_label_not_zero) != 0:
            m = stats.mode(img_label_not_zero)[0]
        else:
            m = 0
        # 最も広い白領域のみを残す
        this_hand = np.uint8(img_label == m)*255

        #膨張
        #kernel = np.ones((4,4),np.uint8)
        #this_hand = cv2.erode(this_hand,kernel,iterations = 1)
        #this_hand = cv2.dilate(this_hand,kernel,iterations = 1)

        # 画像から、学習用ベクトルの取得
        img_vector = getImageVector(this_hand)
        # 学習用データの格納
        if img_vector.size > 0:
            testX = np.append(testX, img_vector, axis=0)
            testy = np.append(testy, hand_class)


print('評価中…')
# 学習済のニューラルネットワークから分類結果を取得
result = clf.predict(testX)

print classification_report(testy, result)
