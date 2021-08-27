#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import time
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import xycar_motor
from ar_track_alvar_msgs.msg import AlvarMarkers
from std_msgs.msg import Int32MultiArray
from tf.transformations import euler_from_quaternion
from PID import PID
from env.xycarRL import *
from rosModule import *
from darknet_ros_msgs.msg import BoundingBoxes

lidar_points = None
ar_flag=0
prevId=0
ultra = [0,0,0,0,0]
arData = {"DX":0.0, "DY":0.0, "DZ":0.0, "AX":0.0, "AY":0.0, "AZ":0.0, "AW":0.0}
prev = [0, 0, 1000, 0, 0]
box_data = BoundingBoxes()
class_name = ''
yolo_angle = 0

def yolo_callback(msg):
  global box_data
  
  box_data = msg

def ultra_callback(msg):
  global ultra

  for j in msg.data:
    #print(msg.data)
    #print(msg.data[0])
    #left
    ultra[0] = msg.data[0]
    #right
    ultra[1] = msg.data[4]
    #rear right
    ultra[2] = msg.data[5]
    #rear mid
    ultra[3] = msg.data[6]
    #rear left
    ultra[4] = msg.data[7]

def lidar_callback(data):
	global lidar_points

	lidar_points = data.ranges

def ar_callback(msg):
	global ar_flag, arData, class_name

	for i in msg.markers:
		#print(prevId, i.id)
		if prevId==0 and i.id==1:
			ar_flag=1

		elif prevId==1 and i.id==2:
			ar_flag=2

		elif prevId==2 and i.id==3:
			ar_flag=3

		elif prevId==3 and i.id==9:
			ar_flag=9

		elif prevId==3 and i.id==6:
			ar_flag=6
		
			class_name='pottedplant'
	
		elif prevId==3 and i.id==5:
			ar_flag=5

			class_name='bicycle'
		

		

		arData["DX"] = i.pose.pose.position.x+0.1 if i.pose.pose.position.x>0 else i.pose.pose.position.x-0.1
            	arData["DY"] = i.pose.pose.position.z
            	#self.arData["DZ"] = i.pose.pose.position.z

            	arData["AX"] = i.pose.pose.orientation.x
            	arData["AY"] = i.pose.pose.orientation.y
            	arData["AZ"] = i.pose.pose.orientation.z
            	arData["AW"] = i.pose.pose.orientation.w

	#ar_flag=1

ros_module = rosmodule()
rospy.Subscriber("/scan", LaserScan, lidar_callback)
rospy.Subscriber('ar_pose_marker', AlvarMarkers, ar_callback)
rospy.Subscriber('xycar_ultrasonic', Int32MultiArray, ultra_callback)
rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, yolo_callback)

pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
data = xycar_motor()
rate = rospy.Rate(10)
angle=0

def ar_parking():
	global data, angle

        (roll, pitch, yaw)=euler_from_quaternion((arData["AX"], arData["AY"], arData["AZ"], arData["AW"]))
        #print(math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
	
        #roll = math.degrees(roll)
        #pitch = math.degrees(pitch)
        yaw = math.degrees(pitch)
        distance = math.sqrt(pow(arData["DX"],2) + pow(arData["DY"],2))
        #print(self.arData["DX"], self.arData["DY"])
        #print(distance)
        atan = math.degrees(math.atan2(arData["DX"], arData["DY"]))

        #print('front', 'atan', atan, 'yaw', yaw, 'distance', distance)
        speed=20
        angle=atan

        if distance>1.8:
            speed=15

        elif distance>1.2:
            speed=15  

        elif distance>0.8:
            speed=15

        elif distance>0.33:
            speed=10
        
        else:
            speed=0

            #디버깅 요망
            #print(yaw, atan, distance)
            if abs(yaw)>8 and abs(atan)>8:
                print('back')
                back_drive()
            else:
                angle=0


        data.angle=angle
        data.speed=speed
        pub.publish(data)

        rate.sleep()


def next_state_rtn(laser_msg, angle): 
     ratio = 220
     increment = 0.714285708755853066
     idx = [275, 315, 0, 45, 90] # fr:31, fm:64 , fl:224    
 
     current_ipt = []
     for i in range(len(idx)):
         real_idx = int(round(float(idx[i]) / increment))
         if idx[i] == 0:
             tmp = [laser_msg[real_idx], laser_msg[real_idx+1], laser_msg[real_idx+2], laser_msg[real_idx+3], laser_msg[real_idx+4]]
         else:
             tmp = [laser_msg[real_idx-2], laser_msg[real_idx-1], laser_msg[real_idx], laser_msg[real_idx+1], laser_msg[real_idx+2]]
         current_ipt.append(max(tmp))
 
     current_ipt.append(angle)
     rtn = np.array(current_ipt)
 
     for j in range(len(current_ipt)-1):
         if rtn[j] == 0:
             rtn[j] = prev[j]
         else:
             prev[j] = rtn[j]
         rtn[j] *= ratio
 
     return rtn

def dqn2xycar():
     xycar = learning_xycar(False) 
  
     hidden_layer = ros_module.get_hidden_size()
     lidar_cnt = ros_module.get_use_lidar_cnt()
     xycar.set_lidar_cnt(lidar_cnt)
     xycar.set_hidden_size(hidden_layer)
  
     state_select = {
         "car sensor" : True,
         "car yaw" : False,
         "car position" : False,
         "car steer" : True
     }           
  
     xycar.state_setup(state_select)
     xycar.ML_init("DQN")
  
     view_epi = ros_module.get_view_epi()
     xycar.load_model(view_epi)
  
     angle = 0
     max_angle = 30
     handle_weights = 6.6
  
     state = xycar.episode_init_ros()
  
     while ar_flag==3 and ros_module.get_ros_shutdown_chk():
         action = xycar.get_action_viewer(state)
         if action == 2:
             angle += handle_weights
         elif action == 0:
             angle -= handle_weights
         elif action == 1:
             angle = 0
         
         angle = max(-max_angle, min(angle, max_angle))
         ros_module.auto_drive(int(float(angle)*(5.0/3.0)), 13)
         next_state = next_state_rtn(ros_module.get_laser_msg(), angle)
         steer = next_state[4]   
 
         state = next_state
         rate.sleep()

def drive_180():
    global data
    cnt=0

    #print("180--------------------------")
    while True:
	cnt+=1
        if(ultra[2]<= 20 and ultra[3] <=20):
          print("obstacle!")

          break

        data.angle = 50
        data.speed = -30
        pub.publish(data)

        rate.sleep()
    
    for _ in range(cnt//2):
	data.angle=-50
	data.speed=15

	pub.publish(data)

	rate.sleep()

    while ar_flag==2:		
	algorithm2xycar()


def back_drive():
	global data
        yaw=math.degrees(euler_from_quaternion((arData["AX"], arData["AY"], arData["AZ"], arData["AW"]))[1])
        back=-yaw
        cnt=0

        while abs(yaw)>5.0:
            cnt+=1

            #print('first', yaw)
            data.angle=back
            #print(self.id, self.xycar_msg.angle)
            data.speed=-30
            pub.publish(data)

            yaw=math.degrees(euler_from_quaternion((arData["AX"], arData["AY"], arData["AZ"], arData["AW"]))[1])

            rate.sleep()

        print('finish first')

        for _ in range(cnt):
            data.angle=back
            data.speed=-30
            pub.publish(data)

            self.r.sleep()
        print('finish second')

        yaw=math.degrees(euler_from_quaternion((arData["AX"], arData["AY"], arData["AZ"], arData["AW"]))[1])
        #print('second', yaw)
        while abs(yaw)>2.0:
            #print('second', yaw)
            data.angle=-back
            data.speed=-30
            pub.publish(data)

            yaw=math.degrees(euler_from_quaternion((arData["AX"], arData["AY"], arData["AZ"], arData["AW"]))[1])

            rate.sleep()
        print('finish third')


def ar_follow():
	global angle, data, ar_flag

	(roll, pitch, yaw)=euler_from_quaternion((arData["AX"], arData["AY"], arData["AZ"], arData["AW"]))
	yaw = math.degrees(pitch)

	distance = math.sqrt(pow(arData["DX"],2) + pow(arData["DY"],2))
	atan = math.degrees(math.atan2(arData["DX"], arData["DY"]))
	angle=atan

	speed=15
	#print(distance)
	if distance>1.8:
            algorithm2xycar()
            
            return

        elif distance>0.9:
            speed=15    

        elif distance>0.6:
            speed=10

        elif distance>0.4:
            speed=10
        
        else:
            drive_180()

        data.angle=angle
        data.speed=speed
        pub.publish(data)

        rate.sleep()

def algorithm2xycar():
	global data, angle
	
	sensor_value = np.concatenate((lidar_points[380:], lidar_points[0:125]))
        l=[s for s in sensor_value[100:150] if s>0.0]
        if not l:
            for i in range(10):
                #print('for1', i)
                data.angle=angle
                data.speed=-30

                pub.publish(data)
                rate.sleep()

            return

        m = min(l)
	#n = max(l)
	#print('min', m, 'max', n)
        #print('min1', m)
	idx=np.concatenate((np.arange(270, 360, 0.72), np.arange(0, 90, 0.72)))
	cos_list=[math.cos(math.radians(d)) for d in idx]
    	#print('cos list', cos_list)
    	sin_list=[math.sin(math.radians(d)) for d in idx]
    	#print('sin list', sin_list)	

	avg_x=0
    	avg_y=0

        for i in range(len(idx)):
        	avg_x+=sin_list[i]*float(sensor_value[i])
        	avg_y+=cos_list[i]*float(sensor_value[i])

	avg_x=float(avg_x)/5.0
    	avg_y=float(avg_y)/5.0
    	avg_distance=float(math.sqrt(float(avg_x**2)+float(avg_y**2)))

	angle=math.degrees(math.asin(avg_x/avg_distance))
    	if angle>50:
        	angle=50.0
    	elif angle<-50:
        	angle=-50.0

        if m<=0.3:
            while True:
                sensor_value=np.concatenate((lidar_points[380:], lidar_points[0:125]))
                l=[s for s in sensor_value[100:150] if s>0.0]
                if not l:
                    for i in range(20):
                        #print('for2', i)
                        data.angle=angle
                        data.speed=-30
			#print('data3', data)

                        pub.publish(data)
                        rate.sleep()

                    return
                m=min(l)
                #print('min2', m)
                if m>0.6:
                    return

                data.speed=-30
                data.angle=angle
                #print('data2', data)

                pub.publish(data)
                rate.sleep()
        else:
            pid=PID(1.0, 0.0, 0.0) 
	    data.angle=-pid.pid_control(angle) 
            data.speed=15

        #print('data1', data)
        pub.publish(data)
	rate.sleep()

def yolo_main():
	global data, yolo_angle

    	boxes = box_data

    	for i in range(len(boxes.bounding_boxes)):      
		print('detect')
      		if boxes.bounding_boxes[i].Class == class_name :
        		center = (boxes.bounding_boxes[i].xmax + boxes.bounding_boxes[i].xmin)/2
        		yolo_angle = int(50.0 *((center - 320.0)/320.0))
        
        		####### pottedplant###
          		data.angle=yolo_angle
			data.speed=10

			pub.publish(data)
          
          		rate.sleep()    
        
    	if len(boxes.bounding_boxes) == 0:		
		data.angle=-(yolo_angle)
		data.speed=10
		
		pub.publish(data)
 
      		rate.sleep()

while not rospy.is_shutdown():
	global prevId, angle

	if ar_flag==0:
		continue

	elif ar_flag==1:
		if lidar_points == None:
			continue

		algorithm2xycar()
		prevId=1

	elif ar_flag==2:
		prevId=2
		ar_follow()

	elif ar_flag==3:
		prevId=3
		dqn2xycar()

	elif ar_flag==9:
		ar_parking()		
	elif ar_flag==6 or ar_flag==5:
		print('start yolo')

		if box_data is None:
			continue

		yolo_main()
