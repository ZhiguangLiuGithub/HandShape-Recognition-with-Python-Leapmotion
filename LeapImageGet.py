# -*- coding: utf-8 -*-
import cv2, Leap, math, ctypes
import numpy as np
import time
from scipy import stats

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
    count = 0
    while(True):
        #sleepで更新速度を制御
        time.sleep(1)
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
            ret,hand = cv2.threshold(undistorted_right,70,255,cv2.THRESH_BINARY)

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

            # 得られた二値化画像を画面に表示
            cv2.imshow('hand', this_hand)
            #cv2.imshow('hand', undistorted_right)
            cv2.imwrite("ImageData/5/img_five{0:03d}.png".format(count),this_hand)
            count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main():
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
    try:
        run(controller)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == '__main__':
    main()
