from picarx import Picarx
import time
import cv2
import numpy as np

car_speed = 65
prev_Mode = "Detection"

def drive(distance, curvature):

    try:
        px = Picarx()
        while True:
            px.forward(30)
            if curvature < 0:
                for angle in range(curvature, 0):
                    px.set_dir_servo_angle(angle)
                    time.sleep(0.01)
            else:
                for angle in range(0, curvature):
                    px.set_dir_servo_angle(angle)
                    time.sleep(0.01)

    finally:
        px.forward(0)


def move(distance, curvature, frame, Mode, Tracked_class):

    px = Picarx()

    angle_of_car , current_speed = beInLane(int(frame.shape[1]/4), distance, curvature , Mode , Tracked_class, px)
                
    angle_speed_str = "[ Angle ,Speed ] = [ " + str(int(angle_of_car)) + " , " + str(int(current_speed)) + " ] "
    cv2.putText(frame,str(angle_speed_str),(20,20),cv2.FONT_HERSHEY_DUPLEX,0.4,(0,0,255),1)

    return frame

def beInLane(Max_Sane_dist, distance, curvature, Mode, Tracked_class, px):

    IncreaseTireSpeedInTurns = True
    global car_speed,prev_Mode
    if((Tracked_class!=0) and (prev_Mode == "Tracking") and (Mode == "Detection")):
        if  (Tracked_class =="speed_sign_70"):
            car_speed = 70
        elif(Tracked_class =="speed_sign_80"):
            car_speed = 80
        elif(Tracked_class =="stop"):
            car_speed = 0
        
    prev_Mode = Mode # Set prevMode to current Mode
    
    Max_turn_angle = 90
    Max_turn_angle_neg = -90

    CarTurn_angle = 0

    if( (distance > Max_Sane_dist) or (distance < (-1 * Max_Sane_dist) ) ):
        # Max sane distance reached ---> Max penalize (Max turn Tires)
        if(distance > Max_Sane_dist):
            #Car offseted left --> Turn full wheels right
            CarTurn_angle = Max_turn_angle + curvature
        else:
            #Car Offseted right--> Turn full wheels left
            CarTurn_angle = Max_turn_angle_neg + curvature
    else:
        # Within allowed distance limits for car and lane
        # Interpolate distance to Angle Range
        Turn_angle_interpolated = np.interp(distance,[-Max_Sane_dist,Max_Sane_dist],[-90,90])
        print("Turn_angle_interpolated = ", Turn_angle_interpolated)
        CarTurn_angle = Turn_angle_interpolated + curvature

    # Handle Max Limit [if (greater then either limits) --> set to max limit]
    if( (CarTurn_angle > Max_turn_angle) or (CarTurn_angle < (-1 *Max_turn_angle) ) ):
        if(CarTurn_angle > Max_turn_angle):
            CarTurn_angle = Max_turn_angle
        else:
            CarTurn_angle = -Max_turn_angle

    angle = np.interp(CarTurn_angle,[-Max_turn_angle,Max_turn_angle],[30,120])

    curr_speed = car_speed

    px.forward(curr_speed)
    
    if curvature < 0:
        for angle in range(curvature, 0):
            px.set_dir_servo_angle(angle)
            time.sleep(0.01)
    else:
        for angle in range(0, curvature):
            px.set_dir_servo_angle(angle)
            time.sleep(0.01)

    return angle , curr_speed