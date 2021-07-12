#! /usr/bin/env python
#-- coding:utf-8 --

import pygame
import math
import numpy as np
import os

screen_width = 1860
screen_height = 1020
check_point = [(1750, 920), (120, 1020)]

class Map(pygame.sprite.Sprite):
    #생성자함수
    def __init__(self, screen):
        super(Map, self).__init__()
        self.screen = screen
	    #지도 이미지 불러오기
        #convert_alpha()를 통해 RGB 채널을 RGBA 채널로 전환한다. 
        current_path = os.path.dirname(__file__)
        self.image = pygame.image.load(os.path.join(current_path, 'hard2.png')).convert_alpha()
        self.rect = self.image.get_rect()
        
        #Mask 충돌체크를 위한 mask 생성
        self.mask = pygame.mask.from_surface(self.image)
    
    #Map 업데이트 함수
    def update(self):
        self.mask = pygame.mask.from_surface(self.image)
	    #이미지를 (0,0)에 위치하도록 출력된다.
        self.screen.blit(self.image, (0, 0))

class Car(pygame.sprite.Sprite):
    def __init__(self, car_file, map, pos):
        super(Car, self).__init__()

        self.angle = 90.0
        current_path = os.path.dirname(__file__)
        self.image = pygame.image.load(os.path.join(current_path, car_file)).convert_alpha()
        self.map = map
        self.rotate_surface = pygame.transform.rotate(self.image, self.angle)
        self.pos = pos
        self.speed = 100.0
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.current_check = 0
        self.goal = False
        self.distance = 0

        #자동차 휠베이스 (축거 : 앞바퀴축과 뒷바퀴축 사이의 거리)
        self.wheel_base = 84.0
        #조향각
        self.steering_angle = 0.0
        #각속도
        self.angular_velocity = 0.0
        #자동차 이미지 좌표 (가로x세로 128x64 픽셀의 자동차 그림파일. car.png)
        self.car_x_ori = [-64,-64, 64, 64] # 왼쪽 위아래, 오른쪽 위아래 포인트 총4개
        self.car_y_ori = [-32, 32,-32, 32] # 왼쪽 위아래, 오른쪽 위아래 포인트 총4개
        
        #자동차 이미지의 새로운 이미지 좌표를 계산하기 위한 리스트를 선언한다. 
        car_x = [0,0,0,0]
        car_y = [0,0,0,0]

        #자동차 이미지의 왼쪽상단, 오른쪽상단, 왼쪽하단, 오른쪽하단의 좌표를 이용해서 자동차가 회전한 변위각에 현재 위치를 더하여 자동차의 이동한 위치를 계산한다. 
        for i in range(4):
            car_x[i] = self.car_x_ori[i] * np.cos(-math.radians(self.angle)) - self.car_y_ori[i] * np.sin(-math.radians(self.angle)) + self.pos[0]
            car_y[i] = self.car_x_ori[i] * np.sin(-math.radians(self.angle)) + self.car_y_ori[i] * np.cos(-math.radians(self.angle)) + self.pos[1]

	    #새로운 이미지 좌표 리스트(x, y 각각)에서 가장 작은 값을 반올림한 후 정수로 변환하여 자동차 이미지의 새로운 좌표를 지정한다.
        self.car_img_x = int(round(min(car_x)))
        self.car_img_y = int(round(min(car_y)))

        self.center = self.pos
        
        self.rect = self.rotate_surface.get_rect()
        #회전 시킨 이미지로 다시 mask를 생성한다. 
        self.mask = pygame.mask.from_surface(self.rotate_surface)

        for d in range(90, -100, -15):
            self.check_radar(d)

        for d in range(90, -100, -15):
            self.check_radar_for_draw(d)

    def draw(self, screen):
        #print(self.car_img_x, self.car_img_y)
        screen.blit(self.rotate_surface, [self.car_img_x, self.car_img_y])


    def draw_radar(self, screen):
        for r in self.radars_for_draw:
            pos, dist = r
            
            pygame.draw.line(screen, (255, 0, 0), self.center, pos, 2)


    def check_collision(self):
        #print('collision', pygame.sprite.collide_mask(self.map, self))
        self.is_alive = (pygame.sprite.collide_mask(self.map, self) == None)


    def check_radar(self, degree):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        '''if degree==90:
            print('firsh', self.map.image.get_at((x, y)), len)'''
        while x>0 and x<screen_width and y>0 and y<screen_height and not self.map.image.get_at((x, y)) == (0, 0, 0, 255) and len < 600:
            len = len + 1

            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

            '''if degree==90:
                print(self.map.image.get_at((x, y)), len)'''

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])


    def check_radar_for_draw(self, degree):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while x>0 and x<screen_width and y>0 and y<screen_height and not self.map.image.get_at((x, y)) == (0, 0, 0, 255) and len < 600:
            len = len + 1

            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars_for_draw.append([(x, y), dist])


    def check_checkpoint(self):
        p = check_point[self.current_check]
        dist = get_distance(p, self.center)

        if dist < 100:
            self.pos[1]=check_point[self.current_check][1]-100
            self.angle=self.angle-180.0

            self.current_check += 1

            if self.current_check >= len(check_point):
                self.current_check = 0


    def update(self, dt):
        #print('update', dt)
        #각속도
        self.angular_velocity = 0.0
        
	    #조향각이 0이 아니라면
        if self.steering_angle != 0.0:
	    #각속도를 계산한다. 각속도=(선속도/회전반지름)
            self.angular_velocity = (self.speed / self.wheel_base) * np.tan(np.radians(self.steering_angle))

        #각변위를 계산해 angle 값에 더해준다. (각속도x시간=각변위)
        self.angle += (np.degrees(self.angular_velocity) * dt)
        #print(self.steering_angle, self.angular_velocity, self.angle)
        #print('----------------------------------------------------')
	    #이동변위를 계산해 spatium(이동거리) 값에 적용한다. (선속도x시간=이동변위)
        self.spatium = self.speed * dt
        self.distance+=self.spatium

        #삼각비를 이용해 x,y 좌표를 구해준다.
        self.pos[0] += (self.spatium * np.cos(np.radians(-self.angle)))
        self.pos[1] += (self.spatium * np.sin(np.radians(-self.angle)))
        #print(self.pos, self.spatium, self.angle, (self.spatium * np.cos(np.radians(-self.angle))), (self.spatium * np.sin(np.radians(-self.angle))))

        #자동차 이미지의 새로운 이미지 좌표를 계산하기 위한 리스트를 선언한다. 
        car_x = [0,0,0,0]
        car_y = [0,0,0,0]

        #자동차 이미지의 왼쪽상단, 오른쪽상단, 왼쪽하단, 오른쪽하단의 좌표를 이용해서 자동차가 회전한 변위각에 현재 위치를 더하여 자동차의 이동한 위치를 계산한다. 
        for i in range(4):
            car_x[i] = self.car_x_ori[i] * np.cos(-math.radians(self.angle)) - self.car_y_ori[i] * np.sin(-math.radians(self.angle)) + self.pos[0]
            car_y[i] = self.car_x_ori[i] * np.sin(-math.radians(self.angle)) + self.car_y_ori[i] * np.cos(-math.radians(self.angle)) + self.pos[1]

	    #새로운 이미지 좌표 리스트(x, y 각각)에서 가장 작은 값을 반올림한 후 정수로 변환하여 자동차 이미지의 새로운 좌표를 지정한다.
        self.car_img_x = int(round(min(car_x)))
        self.car_img_y = int(round(min(car_y)))
        self.center = self.pos

        self.rotate_surface = pygame.transform.rotate(self.image, self.angle)
        self.rect = pygame.Rect(self.car_img_x, self.car_img_y, self.rotate_surface.get_rect().w, self.rotate_surface.get_rect().h)

        #회전 시킨 이미지로 다시 mask를 생성한다. 
        self.mask = pygame.mask.from_surface(self.rotate_surface)
    

class PyGame2D:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)

        self.map=Map(self.screen)
        self.car = Car('car.png', self.map, [120, 900])

        self.game_speed = 60
        self.dt=0.0

    def action(self, action):
        #print('action', action)
        if action == 0:
            self.car.steering_angle += 5
        elif action == 1:
            self.car.steering_angle -= 5
        elif action == 2:
            pass

        if self.car.steering_angle>30:
            self.car.steering_angle=30.0
        elif self.car.steering_angle<-30:
            self.car.steering_angle=-30.0
        #print('steering angle', self.car.steering_angle)

        self.car.update(self.dt)
        self.car.check_collision()
        self.car.check_checkpoint()

        self.car.radars=[]
        for d in range(90, -100, -15):
            self.car.check_radar(d)

    def evaluate(self):
        return reward_cal(self.car.radars, self.car.steering_angle)

    def is_done(self):
        if not self.car.is_alive:
            self.car.current_check = 0
            self.car.distance = 0

            return True

        return False

    def observe(self):
        # return state
        #print(self.car.radars)
        radars = self.car.radars
        ret = [0 for _ in range(13)]
        for i, r in enumerate(radars):
            ret[i] = r[1]

        #print(ret)
        ret.append(self.car.steering_angle)
        #print(ret, self.car.steering_angle)
        return tuple(ret)

    def view(self):
        self.dt = float(self.clock.get_time()) / 1000.0
        #print('view', self.dt)

        # draw game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        #windows 화면을 흰색으로 칠한다. 
        self.screen.fill((255, 255, 255))
        self.map.update()

        self.car.radars_for_draw=[]
        for d in range(90, -100, -15):
            self.car.check_radar_for_draw(d)

        #print(check_point[self.car.current_check])
        #pygame.draw.circle(self.screen, (0, 255, 0), check_point[self.car.current_check], 100, 3)
        self.car.draw_radar(self.screen)
        self.car.draw(self.screen)

        pygame.display.update()
        self.clock.tick(self.game_speed)


def get_distance(p1, p2):
	return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))


def reward_cal(data1, data2):
    idx=[90, 75, 60, 45, 30, 15, 0, 345, 330, 315, 300, 285, 270]
    cos_list=[math.cos(math.radians(d)) for d in idx]
    #print('cos list', cos_list)
    sin_list=[math.sin(math.radians(d)) for d in idx]
    #print('sin list', sin_list)

    avg_x=0
    avg_y=0

    sensor_value_tmp=[r[1] for r in data1]
    steering_tmp=data2

    sensor_value_cal=np.array(sensor_value_tmp)
    #print(sensor_value_cal)
    steering=np.array(steering_tmp)

    for i in range(len(idx)):
        avg_x+=sin_list[i]*float(sensor_value_cal[i])
        avg_y+=cos_list[i]*float(sensor_value_cal[i])

    avg_x=float(avg_x)/5.0
    avg_y=float(avg_y)/5.0
    avg_distance=float(math.sqrt(float(avg_x**2)+float(avg_y**2)))

    result_angle=math.degrees(math.asin(avg_x/avg_distance))

    if result_angle>30:
        result_angle=30.0
    elif result_angle<-30:
        result_angle=-30.0
    #print(steering, result_angle)

    reward=1-abs(float(steering-result_angle))/60.0
    #print(reward)
    return float(reward)



