#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from std_msgs.msg import Int32MultiArray

class Parallel:
    def __init__(self):
        self.image = None
        self.bridge = CvBridge()

        rospy.init_node('parallel_parking')
        rospy.Subscriber("/usb_cam/image_raw", Image, self.img_callback)
        rospy.Subscriber('xycar_ultrasonic', Int32MultiArray, self.ultra_callback)

        self.Width=640
        self.Height=480

        self.pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
        self.xm = xycar_motor()
        self.rate = rospy.Rate(10)

        self.ultra = [0,0,0,0,0]

    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        #print('image', image, np.array(image).size)

    def ultra_callback(self, msg):
        for _ in msg.data:
            #left
            self.ultra[0] = msg.data[0]
            #right
            self.ultra[1] = msg.data[4]
            #rear right
            self.ultra[2] = msg.data[5]
            #rear mid
            self.ultra[3] = msg.data[6]
            #rear left
            self.ultra[4] = msg.data[7]

    def parallel_parking(self):
        if np.array(self.image).size<self.Width*self.Height*3:
            return

        for _ in range(30):
            self.xm.angle=50
            self.xm.speed=-20

            self.pub.publish(self.xm)

            self.rate.sleep()

        for _ in range(30):
            self.xm.angle=-50
            self.xm.speed=-20

            self.pub.publish(self.xm)

            self.rate.sleep()

        

        

        