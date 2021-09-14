#!/usr/bin python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import tensorflow as tf
import os.path as ops

from lanenet_model import lanenet

from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.lanenet_cfg
weights_path='/root/xycar_ws/src/lane_detection/src/model/tusimple_train_miou=0.5487.ckpt-104'

class Deep:
    def __init__(self):
        self.width=640
        self.height=480

        self.input_tensor=tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        self.net=lanenet.LaneNet(phase='test', cfg=CFG)
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='LaneNet') 

        sess_config=tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction=CFG.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth=CFG.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type='BFC'
        self.sess=tf.Session(config=sess_config)

        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=weights_path)

    def auto_drive(self, image):      
        with self.sess.as_default():
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            binary_seg_image, _ = self.sess.run(
                [self.binary_seg_ret, self.instance_seg_ret],
                feed_dict={self.input_tensor: [image]}
            )

            binary=cv2.resize(np.array(binary_seg_image[0] * 255, np.uint8), (640, 480), interpolation=cv2.INTER_LINEAR)
            
            return binary

