#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2, time
#from xycar_msgs.msg import xycar_motor

from Filter import Filter
from PID import PID

Width = 640
Height = 480
Offset = 330
Gap=30

left_fitx=[0 for _ in range(Height)]
right_fitx=[Width for _ in range(Height)]
def lane_detection(binary):
    global left_fitx, right_fitx

    margin = 50
    minpix = 5

    nwindows = 16
    window_height = np.int(Height // nwindows)
    ploty = np.linspace(0, Height-1, Height) 

    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    histogram = np.sum(binary[Height//2:,:], axis=0)   #x축 기준 히스토그램
    out_img = np.dstack((binary, binary, binary))   #이진 영상에 색깔을 입힐 수 있게끔 3차원(BGR)으로 만들어 준다.
    mid_point=Width//2

    centerx_base=np.argmax(histogram[mid_point-150:mid_point+150])+mid_point-150 if np.argmax(histogram[mid_point-150:mid_point+150])!=0 else Width//2
    centerx_current=centerx_base

    center_lane=[]
    cdiff=0
    for window in range(nwindows):
        win_y_low = Height - (window + 1) * window_height
        win_y_high = Height - window * window_height

        win_xcenter_low = centerx_current - margin
        win_xcenter_high = centerx_current + margin

        cv2.rectangle(out_img, (win_xcenter_low, win_y_low), (win_xcenter_high, win_y_high), (0, 255, 255), 4)

        good_center = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xcenter_low) & (
            nonzerox < win_xcenter_high)).nonzero()[0]
        center_lane.append(good_center)
        #print(len(good_center))
        if len(good_center) > minpix:
            centerx_current = np.int(np.mean(nonzerox[good_center]))    #인덱스 배열로 y좌표에 해당하는 x좌표 평균

        if abs(centerx_current-centerx_base)>abs(cdiff):
            cdiff=centerx_current-centerx_base

    center_lane = np.concatenate(center_lane)
    #print(len(center_lane))
    if len(center_lane)>0:
        centerx = nonzerox[center_lane]
        centery = nonzeroy[center_lane]

        center_fit = np.polyfit(centery, centerx, 2)
        cp = np.poly1d(center_fit)

        center_lane_inds = ((nonzerox > (cp(nonzeroy) - margin)) & (nonzerox < (cp(nonzeroy) + margin))) 

        binary[nonzeroy[center_lane_inds], nonzerox[center_lane_inds]] = 0
        out_img[nonzeroy[center_lane_inds], nonzerox[center_lane_inds]] = [0, 0, 0]
    #cv2.imshow('binary', binary)

    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    histogram = np.sum(binary[Height//2:,:], axis=0)

    leftx_base = np.argmax(histogram[:mid_point])
    rightx_base = np.argmax(histogram[mid_point:])+mid_point if np.argmax(histogram[mid_point:])!=0 else Width 

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane = []
    right_lane = []  

    ldiff=0
    rdiff=0
    for window in range(nwindows):
        win_y_low = Height - (window + 1) * window_height
        win_y_high = Height - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 0, 255), 4)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 4)

        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        left_lane.append(good_left)
        if len(good_left) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left]))    #인덱스 배열로 y좌표에 해당하는 x좌표 평균

        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        right_lane.append(good_right)
        if len(good_right) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right]))
        #print(len(good_left), len(good_right))

        if abs(leftx_current-leftx_base)>abs(ldiff):
            ldiff=leftx_current-leftx_base
        if abs(rightx_current-rightx_base)>abs(rdiff):
            rdiff=rightx_current-rightx_base

    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)
    #print(len(left_lane), len(right_lane))

    if len(left_lane)>5000:
        leftx = nonzerox[left_lane]
        lefty = nonzeroy[left_lane]

        left_fit = np.polyfit(lefty, leftx, 2)
        lp = np.poly1d(left_fit)

        left_lane_inds = ((nonzerox > (lp(nonzeroy) - margin)) & (nonzerox < (lp(nonzeroy) + margin))) 
    
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        lp = np.poly1d(left_fit)

        left_fitx = lp(ploty)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    
    if len(right_lane)>5000:
        rightx = nonzerox[right_lane] 
        righty = nonzeroy[right_lane]

        right_fit = np.polyfit(righty, rightx, 2)
        rp = np.poly1d(right_fit)

        right_lane_inds = ((nonzerox > (rp(nonzeroy) - margin)) & (nonzerox < (rp(nonzeroy) + margin)))
    
        rightx = nonzerox[right_lane_inds] 
        righty = nonzeroy[right_lane_inds]

        right_fit = np.polyfit(righty, rightx, 2)
        rp = np.poly1d(right_fit)

        right_fitx = rp(ploty)

        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]
    
    diff=[ldiff, cdiff, rdiff]
    index=[abs(d) for d in diff].index(max([abs(d) for d in diff]))

    cv2.imshow('out_img', out_img)
    return out_img, left_fitx, right_fitx, ploty, diff[index]


def calibrate_image(frame):
    global Width, Height
    
    mtx = np.array([
        [422.037858, 0.0, 245.895397], 
        [0.0, 435.589734, 163.625535], 
        [0.0, 0.0, 1.0]
    ])
    dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])
    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (Width, Height), 1, (Width, Height))
    
    tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    tf_image = tf_image[y:y+h, x:x+w]

    return cv2.resize(tf_image, (Width, Height))


# You are to find "left and light position" of road lanes
def process_image(frame):
    global Offset

    cal=calibrate_image(frame)
    #cv2.imshow('cal', cal)

    gray=cv2.cvtColor(cal, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1)
    alpah = 4.0
    sharp = np.clip((1+alpah)*gray - alpah*blurred, 0, 255).astype(np.uint8)
    ret, s_binary = cv2.threshold(sharp, 145, 255, cv2.THRESH_BINARY)
    #cv2.imshow('s_binary', s_binary)

    canny=cv2.Canny(np.uint8(blurred), 60, 70)
    #cv2.imshow('canny', canny)
    _, b_thresh = cv2.threshold(cal[:, :, 0], 90, 255, cv2.THRESH_BINARY)
    #cv2.imshow('b_thresh', b_thresh)

    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    s_dil=cv2.dilate(s_binary, kernel, iterations=2)
    b_dil=cv2.dilate(b_thresh, kernel, iterations=2)
    #cv2.imshow('s_dil', s_dil)
    #cv2.imshow('b_dil', b_dil)

    binary=cv2.bitwise_and(s_dil, canny)
    binary=cv2.bitwise_and(binary, b_dil)
    #cv2.imshow('binary', binary)

    '''cv2.circle(cal, (Width, 380), 5, (0, 0, 255))
    cv2.circle(cal, (490, Offset-40), 5, (0, 0, 255))
    cv2.circle(cal, (120, Offset-40), 5, (0, 0, 255))
    cv2.circle(cal, (0, 380), 5, (0, 0, 255))
    cv2.imshow('cal', cal)'''
    #영상 roi
    srcQuad=np.float32([
        (Width, 380), 
        (490, Offset-40), 
        (120, Offset-40), 
        (0, 380)
    ])
    dstQuad=np.float32([
        (Width, Height),
        (Width, 0),
        (0, 0),
        (0, Height)
    ])
    #아래는 실제 자이카 D모델 roi
    '''srcQuad=np.float32([
        (Width, 360), 
        (450, Offset-45), 
        (150, Offset-45), 
        (0, 360)
    ])
    dstQuad=np.float32([
        (Width, Height),
        (Width, 0),
        (0, 0),
        (0, Height)
    ])'''
    pers=cv2.getPerspectiveTransform(srcQuad, dstQuad)
    warp=cv2.warpPerspective(binary, pers, (Width, Height))
    #cv2.imshow('warp', warp)

    out_img, left_fitx, right_fitx, ploty, diff=lane_detection(warp)

    pts_left = np.array([np.transpose(np.vstack([
                         left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([
                          right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(out_img, np.int_([pts]), (0,255, 0))
    newpers=cv2.getPerspectiveTransform(dstQuad, srcQuad)
    newwarp = cv2.warpPerspective(out_img, newpers, (Width, Height))

    result = cv2.addWeighted(cal, 1, newwarp, 0.3, 0)

    return diff, result


# You are to publish "steer_anlge" following load lanes
if __name__ == '__main__':
    cap = cv2.VideoCapture('kmu_track.mkv')
    time.sleep(3)

    filter=Filter(10)
    pid=PID(0.55, 0.0, 0.4)

    #rospy.init_node('line_drive')
    #pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    #rate = rospy.Rate(10)

    lpos=0
    rpos=Width
    center=Width//2
    while not rospy.is_shutdown():
        ret, image = cap.read()
        if not ret:
            continue

        diff, result = process_image(image)
        filter.add_sample(diff)
        error=filter.get_wmm()
        angle=(pid.pid_control(error))

        cv2.putText(result, 'angle : {0:0.2f}`'.format(angle), (25, 50), 0, 1, (0, 255, 0), 2)
        cv2.imshow('result', result)
        
        #xm=xycar_motor()
        if abs(angle)<10:
            pid.Kp=0.3
            pid.Ki=0.0
            pid.Kd=0.1
            
            #xm.speed=10
        else:
            pid.Kp=0.4
            pid.Ki=0.0
            pid.Kd=0.1

            #xm.speed=10

        #xm.angle=angle
        #pub.publish(xm)
        #rate.sleep()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
