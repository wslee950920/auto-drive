#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2, time, os
import tensorflow as tf

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from Filter import Filter
from PID import PID

from Deep import Deep

Width = 640
Height = 480

def lane_detection(image, binary):
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    for i in range(1, cnt): # 각각의 객체 정보에 들어가기 위해 반복문. 범위를 1부터 시작한 이유는 배경을 제외
        (x, y, w, h, area) = stats[i]
        #print('area', area)

        # 노이즈 제거
        if area < 1000:
            continue

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255))

    cv2.imshow('image', image)


image = None
bridge = CvBridge()

def img_callback(data):
    global image

    image = bridge.imgmsg_to_cv2(data, "bgr8")
    #print('image', image, np.array(image).size)

# You are to publish "steer_anlge" following load lanes
if __name__ == '__main__':
    rospy.init_node('lane_detection')
    rospy.Subscriber("/usb_cam/image_raw", Image, img_callback)

    deep=Deep()
    while not rospy.is_shutdown():
        if np.array(image).size<Width*Height*3:
            continue

        binary=deep.auto_drive(image)
        cv2.imshow('binary', binary)

        lane_detection(image, binary)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
