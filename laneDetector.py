#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#import imutils
import cv2
import numpy as np
from math import sqrt


def Highlight_lowpass(value, min=0, max=0.5, ratio=1.0):
    # 0 <= value <= 0.5
    new = ratio * value + (1 - ratio) * max

    return new


class VehicleDetector():

    def __init__(self):

        self.bgr_seg = 0.5  # 画面截取比例
        self.binary_lane_seg = 0.6  # 车道线检测线比例
        self.binary_pedroad_seg_01 = 0.4  # 斑马线检测线01比例
        self.binary_pedroad_seg_02 = 0.8  # 斑马线检测线02比例

        self.lane_with_redroad_W_threshold = 0.7
        self.pedroad_threshold = 0.25  # 斑马线白点阈值
        self.single_lane_threshold = 0.5  # 偏移量过大阈值
        self.theta = 0.15  # 车道线检测线宽度

        self.size = None

    def feedCap(self, bgr_img, debug=False):

        offset = None  # 车辆偏移量的估计值
        PdeRoad_bool = [False, False]  # 斑马线检测线01、02是否检测到斑马线

        # 提取摄像头画面下半部分二值图
        H, W = bgr_img.shape[:2]
        roi = cv2.cvtColor(bgr_img[int(H*self.bgr_seg):], cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)
        binary_img = cv2.medianBlur(binary_img, ksize=5)
        cv2.line(bgr_img, (0, int(H*self.bgr_seg)),
                 (W, int(H*self.bgr_seg)), (0, 0, 255), 2)
        
        if debug:
            cv2.imshow('bgr', bgr_img)
            cv2.imshow('roi', binary_img)

        # 人行横道检测01
        binary_pedroad_seg_H_01 = int(
            binary_img.shape[0]*self.binary_pedroad_seg_01)
        bgr_pedroad_seg_H_01 = int(H*self.bgr_seg) + binary_pedroad_seg_H_01
        pedroad_seg_mask_01 = binary_img[binary_pedroad_seg_H_01]

        # 人行横道检测02
        binary_pedroad_seg_H_02 = int(
            binary_img.shape[0]*self.binary_pedroad_seg_02)
        bgr_pedroad_seg_H_02 = int(H*self.bgr_seg) + binary_pedroad_seg_H_02
        pedroad_seg_mask_02 = binary_img[binary_pedroad_seg_H_02]

        # 是否有人行横道01
        pedroad_ratio_01 = float(
            np.sum(pedroad_seg_mask_01 == 255))/float(pedroad_seg_mask_01.shape[0])
        if pedroad_ratio_01 > self.pedroad_threshold:
            cv2.line(bgr_img, (0, bgr_pedroad_seg_H_01),
                     (W, bgr_pedroad_seg_H_01), (0, 255, 0), 2)
            PdeRoad_bool[0] = True
        else:
            cv2.line(bgr_img, (0, bgr_pedroad_seg_H_01),
                     (W, bgr_pedroad_seg_H_01), (255, 0, 0), 2)

        # 是否有人行横道02
        pedroad_ratio_02 = float(
            np.sum(pedroad_seg_mask_02 == 255))/float(pedroad_seg_mask_02.shape[0])
        if pedroad_ratio_02 > self.pedroad_threshold:
            cv2.line(bgr_img, (0, bgr_pedroad_seg_H_02),
                     (W, bgr_pedroad_seg_H_02), (0, 255, 0), 2)
            PdeRoad_bool[1] = True
            cv2.putText(bgr_img, 'Pedestrain Road!', (10, 80),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.line(bgr_img, (0, bgr_pedroad_seg_H_02),
                     (W, bgr_pedroad_seg_H_02), (255, 0, 0), 2)

        # 车道线检测
        binary_lane_seg_H = int(binary_img.shape[0]*self.binary_lane_seg)
        bgr_lane_seg_H = int(H*self.bgr_seg) + binary_lane_seg_H

        start = -int(binary_img.shape[0]*self.theta) + binary_lane_seg_H
        end = int(binary_img.shape[0]*self.theta) + binary_lane_seg_H

        if self.size is None:
            self.size = (bgr_img.shape[1], bgr_img.shape[0])
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            # self.videoWriter = cv2.VideoWriter(
            #     '/home/pi/result.mp4', fourcc, 5, self.size)
            self.videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, 30, self.size)

        # 是否有车道线
        if np.sum(binary_img[start:end] == 255) == 0:
            cv2.putText(bgr_img, 'No Lane Detected!', (10, 40),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
            cv2.rectangle(bgr_img, (0, start+int(H*self.bgr_seg)),
                          (W, end+int(H*self.bgr_seg)), (0, 0, 200), -1)
            self.videoWriter.write(bgr_img)
            return binary_img, bgr_img, offset, PdeRoad_bool, 0

        if PdeRoad_bool[1] or PdeRoad_bool[0]:
            # 如果有斑马线，只检测右侧车道
            seg_W = int(binary_img.shape[1]*self.lane_with_redroad_W_threshold)
            try:
                lane_seg_mask = np.where(binary_img[start:end, seg_W:] == 255)[1]
                lane_left = np.min(lane_seg_mask) + seg_W
                lane_right = lane_left
                lane_center = lane_left
                cv2.rectangle(bgr_img, (seg_W, start+int(H*self.bgr_seg)),
                            (W, end+int(H*self.bgr_seg)), (0, 255, 0), -1)
                cv2.rectangle(bgr_img, (0, start+int(H*self.bgr_seg)),
                            (seg_W, end+int(H*self.bgr_seg)), (0, 0, 200), -1)
            except Exception as e:
                return binary_img, bgr_img, offset, PdeRoad_bool, 0
            if np.sum(binary_img[start:end, seg_W:] == 255):
                cv2.putText(bgr_img, 'No Lane Detected!', (10, 40),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
                cv2.rectangle(bgr_img, (0, start+int(H*self.bgr_seg)),
                            (W, end+int(H*self.bgr_seg)), (0, 0, 200), -1)
                return binary_img, bgr_img, offset, PdeRoad_bool, 0
            
        else:
            lane_seg_mask = np.where(binary_img[start:end] == 255)[1]
            lane_left = np.min(lane_seg_mask)
            lane_right = np.max(lane_seg_mask)
            lane_center = (lane_left + lane_right)/2
            cv2.rectangle(bgr_img, (0, start+int(H*self.bgr_seg)),
                          (W, end+int(H*self.bgr_seg)), (0, 255, 0), -1)

        cv2.circle(bgr_img, (lane_left, bgr_lane_seg_H), 3, (0, 0, 255), 2)
        cv2.circle(bgr_img, (lane_right, bgr_lane_seg_H), 3, (0, 0, 255), 2)
        cv2.circle(bgr_img, (int(lane_center), bgr_lane_seg_H),
                   5, (0, 0, 255), -1)

        if lane_right - lane_left < W * self.single_lane_threshold:
            offset = float(lane_center - W/2)/float(W)
            if offset > 0:
                offset = Highlight_lowpass(0.5 - offset)
                #offset = sqrt(offset * 2) / 2
            else:
                offset = -(Highlight_lowpass(0.5 - abs(offset)))
                #offset = - sqrt(offset * 2) / 2
            if lane_center > W/2:
                text = 'Too lane_right:{:.4f}'.format(offset)
            else:
                text = 'Too lane_left:{:.4f}'.format(offset)
            cv2.putText(bgr_img, text, (10, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
        else:
            offset = float(lane_center - W/2)/float(W)
            cv2.putText(bgr_img, 'lane_center Offset:{:.4f}'.format(offset), (10, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

        ratio_info = 'ratio_1:{:.3f} ratio_2:{:.3f}'.format(pedroad_ratio_01, pedroad_ratio_02)
        cv2.putText(bgr_img, ratio_info, (10, 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

        self.videoWriter.write(bgr_img)
        return binary_img, bgr_img, offset, PdeRoad_bool, 1


if __name__ == '__main__':

    name = 'demo'
    path = 'Video_2.avi'

    det = VehicleDetector()
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    size = None

    while True:

        # try:
        _, bgr_img = cap.read()
        if bgr_img is None:
            break

        bgr_img = cv2.resize(bgr_img, None, fx=0.5, fy=0.5)
        result, bgr_img, offset, _, _ = det.feedCap(bgr_img, debug=True)
        result = cv2.merge([result, result, result])
        result = np.vstack([bgr_img, result])
        cv2.imshow(name, result)
        cv2.waitKey(1)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            break

    cap.release()