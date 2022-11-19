import rospy
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from util.util import euler_to_quaternion, quaternion_to_euler

# from gem_gnss_tracker_pp and gem_gnss_tracker_stanlry_rtk
 
import math
from numpy import linalg as la
import scipy.signal as signal

""" from gem_gnss_tracker_pp and gem_gnss_tracker_stanlry_rtk """
class PID(object):

  def __init__(self, kp, ki, kd, wg=None):

    self.iterm  = 0
    self.last_t = None
    self.last_e = 0
    self.kp     = kp # 0.5
    self.ki     = ki # 0.0
    self.kd     = kd # 0.1
    self.wg     = wg # 20
    self.derror = 0

  def reset(self):
    self.iterm  = 0
    self.last_e = 0
    self.last_t = None

  def get_control(self, t, e, fwd=0):

    if self.last_t is None:
      self.last_t = t
      de = 0
    else:
      de = (e - self.last_e) / (t - self.last_t)

    if abs(e - self.last_e) > 0.5:
      de = 0

    self.iterm += e * (t - self.last_t)

    # take care of integral winding-up
    if self.wg is not None:
      if self.iterm > self.wg:
        self.iterm = self.wg
      elif self.iterm < -self.wg:
        self.iterm = -self.wg

    self.last_e = e
    self.last_t = t
    self.derror = de

    return fwd + self.kp * e + self.ki * self.iterm + self.kd * de

""" from gem_gnss_tracker_pp and gem_gnss_tracker_stanlry_rtk """
class Onlinct_errorilter(object):

  def __init__(self, cutoff, fs, order):
      
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    # Get the filter coct_errorficients 
    self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # Initialize
    self.z = signal.lfilter_zi(self.b, self.a)
  
  def get_data(self, data):

    filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
    return filted

""" from MP4 """
class VehicleController():

  def __init__(self, model_name='gem'):
    # Publisher to publish the control input to the vehicle model
    self.controlPub    = rospy.Publisher("/" + model_name + "/ackermann_cmd", AckermannDrive, queue_size = 1)
    self.model_name    = model_name

    self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
    self.speed_filter  = Onlinct_errorilter(1.2, 30, 4)
    self.desired_speed = 0.60 # m/s
    self.max_throttle  = 0.50
    # self.max_accel     = 0.48 # % of acceleration

    self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
    self.speed      = 0.0

    self.throttle_percent = 0
    self.steering_angle = 0
    self.velocity = 0
    self.turn_at_max_steer = np.radians() # radian turn at max steering angular position (2*pi) input as degrees

  def get_throttle_percent(self,target_velocity):
    
    
    current_time = rospy.get_time()
    filt_vel = np.squeeze(self.speed_filter.get_data(self.speed))
    acceleration_expected = self.pid_speed.get_control(current_time, target_velocity - filt_vel)
    
    """ set throttle percent range between [,]"""
    throttle_percent = (acceleration_expected+2.3501) / 7.3454
    
    """ sanity check on acceleration """
    if acceleration_expected > 0.64 :
      throttle_percent = 0.5
    if acceleration_expected < 0.0 :
      throttle_percent = 0.0

    """ sanity check on throttle """
    if throttle_percent > self.max_throttle:
      throttle_percent = self.max_throttle
    if throttle_percent < 0.3:
      throttle_percent = 0.37

    self.throttle_percent = throttle_percent

  def get_steering_angle(self,curr_steering_angle, crosstrack_error, heading_error): # meters, rads
    # radians, -: clockwise, +: counter-clockwise
    angular_position  = curr_steering_angle
    angular_position += 2*np.pi*np.sin(heading_error / (2*self.turn_at_max_steer))
    angular_position += 2*np.pi*np.sin(np.arctan(crosstrack_error/...))

    if angular_position > self.turn_at_max_steer:
      angular_position = self.turn_at_max_steer
    if angular_position < -self.turn_at_max_steer:
      angular_position = -self.turn_at_max_steer
    
    self.steering_angle = angular_position

  def print_data(self,target_v, lateral_error, lane_theta):
    print("\n")
    print("targ_vel  : ", np.round(self.velocity,4), "\t(m/s)")
    print("steer_ang : ", np.round(np.degrees(self.steering_angle),4), "\t(deg)")
    print("lat_err   : ", np.round(lateral_error,3), "\t(m)")
    print("lane_theta: ", np.round(np.degrees(lane_theta),4), "\t(deg)")
    pass

  def execute(self, target_v, lateral_error, lane_theta):
    """
    This function takes the current state of the vehicle and
    the target state to compute low-level control input to the vehicle
    Inputs:
        target_v: The desired velocity of the vehicle
        lateral_error: The lateral tracking error from the center line of the current lane
        lane_theta: The current lane heading
    """

    # TODO: compute errors and send computed control input to vehicle
    self.get_steering_angle  (lateral_error, lane_theta)
    self.get_throttle_percent(target_v)
    self.print_data(target_v, lateral_error, lane_theta)

    return self.throttle_percent, self.steering_angle
