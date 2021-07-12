#!/usr/bin/env python
#-- coding:utf-8 --

import rospy, time, cv2, math
import numpy as np

from xycar_msgs.msg import xycar_motor
from ar_track_alvar_msgs.msg import AlvarMarkers
from tf.transformations import euler_from_quaternion

class AR_PARKING:
    arData = {"DX":0.0, "DY":0.0, "DZ":0.0, "AX":0.0, "AY":0.0, "AZ":0.0, "AW":0.0}

    motor_pub = rospy.Publisher("/xycar_motor", xycar_motor, queue_size=1)
    xycar_msg = xycar_motor()

    def callback(self, msg):
        for i in msg.markers:
            self.arData["DX"] = i.pose.pose.position.x+0.1 if i.pose.pose.position.x>0 else i.pose.pose.position.x-0.1
            self.arData["DY"] = i.pose.pose.position.z
            #self.arData["DZ"] = i.pose.pose.position.z

            self.arData["AX"] = i.pose.pose.orientation.x
            self.arData["AY"] = i.pose.pose.orientation.y
            self.arData["AZ"] = i.pose.pose.orientation.z
            self.arData["AW"] = i.pose.pose.orientation.w
    

    def listener(self):
        rospy.init_node('ar_parking', anonymous=False)
        rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.callback)


    def back_drive(self):
        yaw=euler_from_quaternion((self.arData["AX"], self.arData["AY"], self.arData["AZ"], self.arData["AW"]))[1]
        back=math.degrees(-yaw)

        while abs(yaw)>0.02:
            self.xycar_msg.angle=back
            self.xycar_msg.speed=-30
            self.motor_pub.publish(self.xycar_msg)

            yaw=euler_from_quaternion((self.arData["AX"], self.arData["AY"], self.arData["AZ"], self.arData["AW"]))[1]
        print('finish first')

        while abs(back)-abs(math.degrees(yaw))>1.0:
            self.xycar_msg.angle=back-math.degrees(yaw)
            self.xycar_msg.speed=-30
            self.motor_pub.publish(self.xycar_msg)

            yaw=euler_from_quaternion((self.arData["AX"], self.arData["AY"], self.arData["AZ"], self.arData["AW"]))[1]
        print('finish second')

        while abs(yaw)>0.02:
            self.xycar_msg.angle=-back
            self.xycar_msg.speed=-30
            self.motor_pub.publish(self.xycar_msg)

            yaw=euler_from_quaternion((self.arData["AX"], self.arData["AY"], self.arData["AZ"], self.arData["AW"]))[1]
        print('finish third')

    def ar_parking(self):
        (roll, pitch, yaw)=euler_from_quaternion((self.arData["AX"], self.arData["AY"], self.arData["AZ"], self.arData["AW"]))
        #print(math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
	
        #roll = math.degrees(roll)
        #pitch = math.degrees(pitch)
        yaw = math.degrees(pitch)

        img = np.zeros((100, 500, 3))

        img = cv2.line(img,(25,65),(475,65),(0,0,255),2)
        img = cv2.line(img,(25,40),(25,90),(0,0,255),3)
        img = cv2.line(img,(250,40),(250,90),(0,0,255),3)
        img = cv2.line(img,(475,40),(475,90),(0,0,255),3)

        #이걸로 ratio조정 해보자...
        ratio=150
        point = int(self.arData["DX"]*ratio) + 250

        if point > 475:
            point = 475

        elif point < 25: 
            point = 25	

        img = cv2.circle(img,(point,65),15,(0,255,0),-1)  
  
        distance = math.sqrt(pow(self.arData["DX"],2) + pow(self.arData["DY"],2))
        #print(self.arData["DX"], self.arData["DY"])
        #print(distance)
        atan = math.degrees(math.atan2(self.arData["DX"], self.arData["DY"]))

        cv2.putText(img, str(round(distance, 2))+"m", (350,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

        #print('dx', self.arData["DX"], 'dy', self.arData["DY"])
        dx_dy_yaw_atan = "DX:"+str(int(self.arData["DX"]))+" DY:"+str(int(self.arData["DY"]))\
                +" Yaw:"+ str(round(yaw,1))+" Atan:"+str(round(atan, 1)) 
        cv2.putText(img, dx_dy_yaw_atan, (20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255))

        cv2.imshow('AR Tag Position', img)
        cv2.waitKey(1)

        #print('front', 'atan', atan, 'yaw', yaw, 'distance', distance)
        angle=atan
        speed=40
        if distance>1.8:
            speed=30

        elif distance>0.9:
            speed=25    

        elif distance>0.6:
            speed=20

        elif distance>0.3:
            speed=15
        
        else:
            speed=0

            #디버깅 요망
            #print(yaw, atan, distance)
            if abs(yaw)>5 and abs(atan)>25:
                print('back')
                self.back_drive()
            else:
                angle=0

        self.xycar_msg.angle=angle
        self.xycar_msg.speed=speed
        self.motor_pub.publish(self.xycar_msg)


if __name__ == '__main__':
    parking = AR_PARKING()
    parking.listener()
    r = rospy.Rate(10)

    while not rospy.is_shutdown():
        parking.ar_parking()

        r.sleep()

