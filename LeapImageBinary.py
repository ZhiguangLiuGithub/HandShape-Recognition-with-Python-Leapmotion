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
    while(True):
        #sleepで更新速度を制御
        time.sleep(0.1)
        frame = controller.frame()
        image = frame.images[0]
        if image.is_valid:
            if not maps_initialized:
                left_coordinates, left_coefficients = convert_distortion_maps(frame.images[0])
                right_coordinates, right_coefficients = convert_distortion_maps(frame.images[1])
                maps_initialized = True

            undistorted_left = undistort(image, left_coordinates, left_coefficients, 400, 400)
            undistorted_right = undistort(image, right_coordinates, right_coefficients, 400, 400)

            hand = undistorted_right[80:320,40:360]

            #hand_color = cv2.cvtColor(hand,cv2.COLOR_GRAY2BGR)

            #hand_gray = cv2.cvtColor(hand,cv2.COLOR_BGR2GRAY)

            #画像を２値化（白黒に処理）
            ret,my_hand = cv2.threshold(hand,70,255,cv2.THRESH_BINARY)

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
            kernel = np.ones((5,5),np.uint8)
            this_hand = cv2.dilate(this_hand,kernel,iterations = 1)

            hand_color = cv2.cvtColor(this_hand,cv2.COLOR_GRAY2BGR)

            # 輪郭を抽出
            contours,hierarchy = cv2.findContours(this_hand,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            #hand_color = cv2.drawContours(hand_color, [cnt], 0, (0,255,0), 3)
            #hand_color = cv2.drawContours(hand_color, contours[0::4], 0, (0,255,0), 3)

            #hull = cv2.convexHull(cnt,returnPoints = False)
            #defects = cv2.convexityDefects(cnt,hull)

            #for i in range(defects.shape[0]):
            #    s,e,f,d = defects[i,0]
            #    start = tuple(cnt[s][0])
            #    end = tuple(cnt[e][0])
            #    far = tuple(cnt[f][0])
            #    cv2.line(hand_color,start,end,[0,255,0],2)
            #    cv2.circle(hand_color,far,5,[0,0,255],-1)

            #for p in cnt[0::6]:
                #point = tuple(p[0])
                #cv2.circle(hand_color,point,2,[0,0,255],-1)

            #重心を求めて表示
            mu = cv2.moments(this_hand, False)
            mx,my= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
            cv2.circle(hand_color, (mx,my),2,[0,0,255],-1)

            #手首の位置を求める
            frame = controller.frame()
            righthand = frame.hands.rightmost
            arm = righthand.arm

            i_box = frame.interaction_box
            normalized_tip = i_box.normalize_point(arm.wrist_position)
            app_x = 160  * normalized_tip.x + 80
            app_y = 120 * (normalized_tip.z) + 60
            app = (int(app_x),int(app_y))

            cv2.circle(hand_color,app,2,[0,255,0],-1)

            #cv2.imshow('hand', hand_color)

            #重心と手首の位置から回転させる
            angle = 90 + math.degrees(math.atan2(my-app[1],mx-app[0]))
            print(angle)
            trans = cv2.getRotationMatrix2D((mx,my), angle , 1.0)
            hand_color = cv2.warpAffine(hand_color, trans, (360,240))

            #cv2.circle(hand_color,app,2,[255,0,0],-1)

            #手首以下を黒にする
            rx = int((app[0]-mx)*math.cos(angle)-(app[1]-my)*math.sin(angle) + mx)
            ry = int((app[0]-mx)*math.sin(angle)+(app[1]-my)*math.cos(angle) + my)
            #cv2.circle(hand_color, (rx,ry),2,[255,0,0],-1)
            print(ry)
            #hand_color = cv2.rectangle(hand_color, (0, ry), (360, 240), color=(0, 0, 0), thickness=-1)


            # 得られた二値化画像を画面に表示
            cv2.imshow('hand', hand_color)
            #cv2.imshow('hand', undistorted_right)


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
