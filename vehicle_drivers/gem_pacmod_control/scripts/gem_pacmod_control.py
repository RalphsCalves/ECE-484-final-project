#!/usr/bin/env python3

#==============================================================================
# File name          : gem_pacmod_control.py                                                                  
# Description        : pacmod interface                                                             
# Author             : Hang Cui
# Email              : hangcui3@illinois.edu                                                                     
# Date created       : 08/08/2022                                                                 
# Date last modified : 08/18/2022                                                          
# Version            : 1.0                                                                    
# Usage              : rosrun gem_pacmod_control gem_pacmod_control.py                                                                   
# Python version     : 3.8                                                             
#==============================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

# ROS Headers
# import rospy
# import alvinxy.alvinxy as axy 
from ackermann_msgs.msg import AckermannDrive
# from std_msgs.msg import String, Bool, Float32, Float64
# from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# ROS Headers
import alvinxy.alvinxy as axy # Import AlvinXY transformation module
import rospy

# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt

# from MP4 main
from gem_vision.scripts.gem_vision import VehiclePerception
from controller import VehicleController
from decision import VehicleDecision

""" for Final Project """
class PACMod(object):
    
    def __init__(self):

        self.rate = rospy.Rate(25)

        self.stanley_sub = rospy.Subscriber('/gem/stanley_gnss_cmd', AckermannDrive, self.stanley_gnss_callback)

        self.ackermann_msg_gnss                         = AckermannDrive()
        self.ackermann_msg_gnss.steering_angle_velocity = 0.0
        self.ackermann_msg_gnss.acceleration            = 0.0
        self.ackermann_msg_gnss.jerk                    = 0.0
        self.ackermann_msg_gnss.speed                   = 0.0 
        self.ackermann_msg_gnss.steering_angle          = 0.0

        # -------------------- PACMod setup --------------------

        self.gem_enable    = False
        self.pacmod_enable = False

        # GEM vehicle enable
        self.enable_sub = rospy.Subscriber('/pacmod/as_rx/enable', Bool, self.pacmod_enable_callback)
        # self.enable_cmd = Bool()
        # self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        # GEM vechile turn signal control
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 2.0 # radians/second


    # Get outputs of Stanley controller based on GNSS
    def stanley_gnss_callback(self, msg):
        self.ackermann_msg_gnss.acceleration = round(msg.acceleration ,2)
        self.ackermann_msg_gnss.steering_angle = round(msg.steering_angle ,2)
    
    # Conversion of front wheel to steering wheel
    def front2steer(self, f_angle):
      # sanity check on f_angle
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35

        # set steering angle based on front wheel angle_angle
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
            
        else:
            steer_angle = 0.0
        return steer_angle

    # Conversion to -pi to pi
    def pi_2_pi(self, angle):

        if angle > np.pi:
            return angle - 2.0 * np.pi

        if angle < -np.pi:
            return angle + 2.0 * np.pi

        return angle

    # Get vehicle states: x, y, yaw                                                     [ADDED from gem_gnss_tracker_stanley_rtk]
    def get_gem_state(self):

        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # rct_errorerence point is located at the center of GNSS antennas
        local_x_curr, local_y_curr = self.wps_to_local_xy_stanley(self.lon, self.lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = self.heading_to_yaw_stanley(self.heading) 

        # rct_errorerence point is located at the center of front axle
        curr_x = local_x_curr + self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr + self.offset * np.sin(curr_yaw)

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

    # Start PACMod interface
    def start_pacmod(self):
        
      while not rospy.is_shutdown():

        if(self.pacmod_enable == True):
          
          if (self.gem_enable == False):

              # ---------- Enable PACMod ----------

              # enable forward gear
              self.gear_cmd.ui16_cmd = 3

              # enable brake
              self.brake_cmd.enable  = True
              self.brake_cmd.clear   = False
              self.brake_cmd.ignore  = False
              self.brake_cmd.f64_cmd = 0.0

              # enable gas 
              self.accel_cmd.enable  = True
              self.accel_cmd.clear   = False
              self.accel_cmd.ignore  = False
              self.accel_cmd.f64_cmd = 0.0

              self.gear_pub.publish(self.gear_cmd)
              print("Foward Engaged!")

              self.turn_pub.publish(self.turn_cmd)
              print("Turn Signal Ready!")
              
              self.brake_pub.publish(self.brake_cmd)
              print("Brake Engaged!")

              self.accel_pub.publish(self.accel_cmd)
              print("Gas Engaged!")

              self.gem_enable = True

          else: 
            model_name = "ralphs_model"
            
            # Get the current position and orientation of the vehicle
            perceptionModule = VehiclePerception(model_name)
            # currState =  perceptionModule.gpsReading()      # currState is not necessary
            front_dist = perceptionModule.lidarReading()      # float
            lateral_error, lane_theta = perceptionModule.cameraReading()

            # get the 
            decisionModule = VehicleDecision()
            target_v, lateral_error, lane_theta = decisionModule.get_ref_state(currState = None, front_dist, lateral_error, lane_theta)

            controlModule = VehicleController(model_name)
            throttle_percent, steering_angle = controlModule.execute(target_v, lateral_error, lane_theta)

            # self.ackermann_msg_gnss.acceleration = throttle_percent
            # self.ackermann_msg_gnss.steering_angle = steering_angle # in degrees


            if (self.ackermann_msg_gnss.steering_angle <= 45 and self.ackermann_msg_gnss.steering_angle >= -45):
              self.turn_cmd.ui16_cmd = 1
            elif(self.ackermann_msg_gnss.steering_angle > 45):
              self.turn_cmd.ui16_cmd = 2 # turn left
            else:
              self.turn_cmd.ui16_cmd = 0 # turn right

            # self.accel_cmd.f64_cmd = self.ackermann_msg_gnss.acceleration # assumes we are using gnss
            # self.steer_cmd.angular_position = np.radians(self.ackermann_msg_gnss.steering_angle) # assumes we are using gnss
            self.accel_cmd.f64_cmd = throttle_percent
            self.steer_cmd.angular_position = np.radians(steering_angle)
            self.accel_pub.publish(self.accel_cmd)
            self.steer_pub.publish(self.steer_cmd)
                  

          self.rate.sleep()


def pacmod_run():

    rospy.init_node('pacmod_control_node', anonymous=True)
    pacmod = PACMod()

    try:
        pacmod.start_pacmod()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pacmod_run()


