import pickle
import numpy as np


""" from MP4 """
class VehicleDecision():
  def __init__(self):
    self.lane_width = 4.4

    # vehicle state variable
    # 0: lane keeping
    # 1: lane changing - turning stage
    # 2: lane changing - stabilizing stage
    # 3: emergency stop
    self.vehicle_state = 0

    self.state_verbose = ["lane keeping", "start lane changing", "stabilizing the car", "emergency stop"]

    # current lane
    self.current_lane = 'left'

  def get_ref_state(self, currState, front_dist, lateral_error, lane_theta):
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
    """
    1. sanity check on current dist betw GEM and obstacle
    2. 
    Lets hard code a velocity
    """
    # 1. sanity check on current dist betw GEM and obstacle
    EMERGENCY_BRAKING_DISTANCE = 9 # m
    D_TURNING     = 4 # m
    D_STABILIZING = 2 #

    V_GOAL_MAINTENANCE = 1.5 # m/s
    V_TURNING          = 1.0 # m/s
    V_STABILIZING      = 0.5 # m/s
    V_STOPPING         = 0.0 # m/s

    if front_dist < EMERGENCY_BRAKING_DISTANCE: # [min sensing distance]
      self.vehicle_state = 3
    if front_dist < D_TURNING:
      self.vehicle_state = 2
    if front_dist < D_STABILIZING:
      self.vehicle_state = 1
    else:
      self.vehicle_state = 0
    
    
    if self.vehicle_state == 0:
      ref_v = V_GOAL_MAINTENANCE
    elif self.vehicle_state == 1:
      ref_v = V_TURNING
    elif self.vehicle_state == 2:
      ref_v = V_STABILIZING
    elif self.vehicle_state == 3:
      ref_v = V_STOPPING
    
    # print("current state: ", self.state_verbose[self.vehicle_state])

    print("curr vehicle state:", self.state_verbose[self.vehicle_state], "target speed:", ref_v)
    return ref_v, lateral_error, lane_theta
