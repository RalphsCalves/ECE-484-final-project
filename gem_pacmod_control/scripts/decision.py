import pickle
import numpy as np


""" from MP4 """
class VehicleDecision():
  def __init__(self):

    # vehicle state variable
    # 0: lane keeping
    # 1: turning
    # 2: off course
    # 2: emergency stop
    self.vehicle_state = 0

    self.state_verbose = ["lane keeping", "turning", "turning hard", "emergency stop"]
    self.sensing_distance = 15
    self.lane_width = 1.2
    self.wheelbase = 2*(0.61)
    self.max_steer = 2*np.pi
    self.max_turn  = (np.pi/180) * 90 # (deg) max front turn (of wheels)
    self.max_steering_angle_for_straight = (np.pi/180) * 90
    self.str_ang_straight_bounds = [np.radians(-10), np.radians(10)]


  def get_ref_state(self, front_dist, lateral_error, lane_theta):
    """
    Get the reference state for the vehicle according to the current state and result from perception module
    Inputs:
        currState: ModelState, the current state of vehicle
        front_dist: float, the current distance between vehicle and obstacle in frontla
        lateral_error: float, the current lateral tracking error from the center line
        lane_theta: the current lane heading with respect to the vehicle
    Outputs: reference velocity, lateral tracking error, and lane heading
    """

    # TODO: Implement decision module
    # --- emergency braking ---
    if front_dist < self.sensing_distance or front_dist != None:
      self.vehicle_state = 3
    
    else:
      # --- off course ---
      if   lateral_error >  ((self.lane_width - self.wheelbase)/ 2) or lane_theta > np.radians(self.max_turn):
        self.vehicle_state = 2
      # --- on course --- 
      elif lateral_error <= ((self.lane_width - self.wheelbase)/ 2) or lane_theta < np.radians(self.max_turn):

        # --- yes hard turn ---
        if lane_theta < 0 + self.str_ang_straight_bounds[0] or lane_theta > 0 + self.str_ang_straight_bounds[1]:
          self.vehicle_state = 1
        # --- no hard turn ---
        if lane_theta > 0 + self.str_ang_straight_bounds[0] and lane_theta < 0 + self.str_ang_straight_bounds[1]:
          self.vehicle_state = 0
        

    print("curr vehicle state:", self.state_verbose[self.vehicle_state])
    print("target speed:", ref_v)
    print("front_dist", front_dist)
    return ref_v, lateral_error, lane_theta
