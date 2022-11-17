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
    self.kp     = kp
    self.ki     = ki
    self.kd     = kd
    self.wg     = wg
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
    self.desired_speed = 0.6  # m/s
    self.max_accel     = 0.48 # % of acceleration
    self.speed_filter  = Onlinct_errorilter(1.2, 30, 4)

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
    set_speed = target_v
    crosstrack_error = lateral_error
    heading_error = lane_theta # rads

    

    # ----------------- tuning this part as needed -----------------
    k       = 0.41 
    wheelbase = .61
    angle_i = math.atan((k * 2 * self.wheelbase * np.sin(alpha)) / L) 
    angle   = angle_i*2
    # ----------------- tuning this part as needed -----------------

    f_delta        = round(heading_error + np.arctan2(crosstrack_error*0.4, filt_vel), 3)
    f_delta        = round(np.clip(f_delta, -0.61, 0.61), 3)
    f_delta_deg    = np.degrees(f_delta)
    steering_angle = self.front2steer(f_delta_deg)

    print("Crosstrack Error: " + str(round(crosstrack_error,3)) + ", Heading Error: "  + str(heading_error))
    print("Desired Speed: " + str(target_v) + ", Desired Str Ang: " + str(steering_angle))

    newAckermannCmd = AckermannDrive()
    newAckermannCmd.speed = set_speed
    newAckermannCmd.steering_angle = steering_angle

    # Publish the computed control input to vehicle model
    self.controlPub.publish(newAckermannCmd)

    filt_vel = np.squeeze(self.speed_filter.get_data(self.speed))
    a_expected = self.pid_speed.get_control(rospy.get_time(), self.desired_speed - filt_vel)

    """ sanity check on throttle """
    if a_expected > 0.64 :
      throttle_percent = 0.5

    if a_expected < 0.0 :
      throttle_percent = 0.0

    throttle_percent = (a_expected+2.3501) / 7.3454

    if throttle_percent > self.max_accel:
      throttle_percent = self.max_accel

    if throttle_percent < 0.3:
      throttle_percent = 0.37


    return throttle_percent, steering_angle
