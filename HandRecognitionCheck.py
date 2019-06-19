# -*- coding: utf-8 -*-
import cv2, Leap, math, ctypes
import sys
import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import time
from decimal import Decimal
import matplotlib.pyplot as plt

# 学習に用いる縮小画像のサイズ
sw = 160
sh = 120

# 手の認識用パラメータ（HチャンネルとSチャンネルとを二値化するための条件）
hmin = 0
hmax = 30 # 15-40程度にセット
smin = 50

hand_class =  ['0', '1', '2', '3', '4', '5']

#表示の設定
fig = plt.figure(figsize=(8,8))
checkbar = fig.add_subplot(2,2,1)
recog_image = fig.add_subplot(2,2,4)
recog_image.tick_params(bottom=False,
                left=False,
                right=False,
                top=False)
recog_image.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
prediction_num = fig.add_subplot(2,2,3)
prediction_num.tick_params(bottom=False,
                left=False,
                right=False,
                top=False)
prediction_num.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
#result_num = fig.add_subplot(2,2,2)
#result_num.tick_params(bottom=False,
#                left=False,
#                right=False,
#                top=False)
#result_num.tick_params(labelbottom=False,
#                labelleft=False,
#                labelright=False,
#                labeltop=False)

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
    #cv2.imshow('img_nonzero', img_nonzero)
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
    #cv2.imshow('img_small', img_small)
    return np.array([img_small.ravel()/255.])

def convert_distortion_maps(image):

    distortion_length = image.distortion_width * image.distortion_height
    xmap = np.zeros(distortion_length/2, dtype=np.float32)
    ymap = np.zeros(distortion_length/2, dtype=np.float32)

    for i in range(0, distortion_length, 2):
        xmap[distortion_length/2 - i/2 - 1] = image.distortion[i] * image.width
        ymap[distortion_length/2 - i/2 - 1] = image.distortion[i + 1] * image.height

    xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width/2))
    ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width/2))

    #resize the distortion map to equal desired destination image size
    resized_xmap = cv2.resize(xmap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)
    resized_ymap = cv2.resize(ymap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)

    #Use faster fixed point maps
    coordinate_map, interpolation_coefficients = cv2.convertMaps(resized_xmap,
                                                                 resized_ymap,
                                                                 cv2.CV_32FC1,
                                                                 nninterpolation = False)

    return coordinate_map, interpolation_coefficients

def undistort(image, coordinate_map, coefficient_map, width, height):
    destination = np.empty((width, height), dtype = np.ubyte)

    #wrap image data in numpy array
    i_address = int(image.data_pointer)
    ctype_array_def = ctypes.c_ubyte * image.height * image.width
    # as ctypes array
    as_ctype_array = ctype_array_def.from_address(i_address)
    # as numpy array
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    img = np.reshape(as_numpy_array, (image.height, image.width))

    #remap image to destination
    destination = cv2.remap(img,
                            coordinate_map,
                            coefficient_map,
                            interpolation = cv2.INTER_LINEAR)

    #resize output to desired destination size
    destination = cv2.resize(destination,
                             (width, height),
                             0, 0,
                             cv2.INTER_LINEAR)
    return destination

def run(controller):
    maps_initialized = False
    while(True):
        #sleepで更新速度を制御
        #time.sleep(0.01)
        frame = controller.frame()
        image = frame.images[0]
        if image.is_valid:
            if not maps_initialized:
                left_coordinates, left_coefficients = convert_distortion_maps(frame.images[0])
                right_coordinates, right_coefficients = convert_distortion_maps(frame.images[1])
                maps_initialized = True

            undistorted_left = undistort(image, left_coordinates, left_coefficients, 400, 400)
            undistorted_right = undistort(image, right_coordinates, right_coefficients, 400, 400)

            #画像を２値化（白黒に処理）
            ret,hand = cv2.threshold(undistorted_right,90,255,cv2.THRESH_BINARY)

            my_hand = hand[80:320,40:360]

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
            kernel = np.ones((4,4),np.uint8)
            #this_hand = cv2.erode(this_hand,kernel,iterations = 1)
            this_hand = cv2.dilate(this_hand,kernel,iterations = 1)

            # 輪郭を抽出
            contours,hierarchy = cv2.findContours(this_hand,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]

            #重心を求める
            mu = cv2.moments(this_hand, False)
            mx,my= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])

            #手首の位置を求める
            frame = controller.frame()
            righthand = frame.hands.rightmost
            arm = righthand.arm

            i_box = frame.interaction_box
            normalized_tip = i_box.normalize_point(arm.wrist_position)
            app_x = 160  * normalized_tip.x + 80
            app_y = 120 * (normalized_tip.z) + 60
            app = (int(app_x),int(app_y))

            #重心と手首の位置から回転させる
            angle = 90 + math.degrees(math.atan2(my-app[1],mx-app[0]))
            trans = cv2.getRotationMatrix2D((mx,my), angle , 1.0)
            this_hand = cv2.warpAffine(this_hand, trans, (360,240))


            # 最大の白領域からscikit-learnに入力するためのベクトルを取得
            hand_vector = getImageVector(this_hand)

            # 学習済のニューラルネットワークから分類結果を取得
            result = clf.predict(hand_vector)
            # 分類結果を表示
            #print(hand_class[result[0]])

            pp = clf.predict_proba(hand_vector)[0]
            hc = [0,1,2,3,4,5]

            recog_image.cla()
            checkbar.cla()
            prediction_num.cla()
            #result_num.cla()

            checkbar.bar(hc,pp)
            checkbar.set_xticks(hc,hand_class)
            checkbar.set_ylim([0,1])

            this_hand = cv2.cvtColor(this_hand, cv2.COLOR_GRAY2RGB)
            recog_image.imshow(this_hand)

            prediction_num.text(0.3,0.3,str(hand_class[result[0]]),size=100)

            #if pp[int(hand_class[result[0]])] > 0.9:
            #    result_num.text(0.3,0.3,str(hand_class[result[0]]),size=100)
            #else:
            #    result_num.text(0.3,0.3," ",size=100)

            plt.draw()

            plt.pause(0.001)


            #if cv2.waitKey(1) & 0xFF == ord('q'):
            if 0xFF == ord('q'):
                break

def main():
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

    print('認識を開始します')
    try:
        run(controller)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == '__main__':
    main()
