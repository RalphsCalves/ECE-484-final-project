#!/usr/bin/env python3

#================================================================
# File name: gem_vision.py                                                                  
# Description: show image from ZED camera                                                          
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 05/20/2021                                                                
# Date last modified: 08/24/2022                                                        
# Version: 0.1                                                                    
# Usage: rosrun gem_vision gem_vision.py                                                                      
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

import sys
import copy
import time
import rospy
import rospkg

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from std_msgs.msg import String, Float64
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# also on studentVision (mp1)
import math                                                     # ADDED
from line_fit import line_fit, tune_fit, bird_fit, final_viz    # ADDED
from Line import Line                                           # ADDED
from std_msgs.msg import Header, Float32                        # ADDED
from skimage import morphology                                  # ADDED

# also in perception (mp4)
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from gazebo_msgs.srv import GetModelState, GetModelStateResponse

from Line import Line
from line_fit import line_fit, tune_fit, bird_fit, final_viz


from pacmod_msgs.msg import SystemRptFloat, VehicleSpeedRpt # ADDED for velocity control

""" from MP4 perception.py"""
class VehiclePerception:
  def __init__(self, model_name='gem', resolution=0.1, side_range=(-20., 20.),
        fwd_range=(-20., 20.), height_range=(-1.6, 0.5)):
    # self.lane_detector = lanenet_detector()
    self.lane_detector = ImageConverter()
    self.lidar = LidarProcessing(resolution=resolution, side_range=side_range, fwd_range=fwd_range, height_range=height_range)

    self.bridge = CvBridge()
    self.model_name = model_name

  def cameraReading(self):
    # Get processed reading from the camera on the vehicle
    # Input: None
    # Output:
    # 1. Lateral tracking error from the center line of the lane
    # 2. The lane heading with respect to the vehicle
    return self.lane_detector.lateral_error, self.lane_detector.lane_theta

  def lidarReading(self):
    # Get processed reading from the Lidar on the vehicle
    # Input: None
    # Output: Distance between the vehicle and object in the front
    res = self.lidar.processLidar()
    return res

  def gpsReading(self):
    # Get the current state of the vehicle
    # Input: None
    # Output: ModelState, the state of the vehicle, contain the
    #   position, orientation, linear velocity, angular velocity
    #   of the vehicle
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        modelState = serviceResponse(model_name=self.model_name)
    except rospy.ServiceException as exc:
        rospy.loginfo("Service did not process request: "+str(exc))
        modelState = GetModelStateResponse()
        modelState.success = False
    return modelState

""" from MP4 perception.py"""
class LidarProcessing:
  def __init__(self, resolution=0.1, side_range=(-20., 20.), fwd_range=(-20., 20.), height_range=(-1.6, 0.5)):
    self.resolution = resolution
    self.side_range = side_range
    self.fwd_range = fwd_range
    self.height_range = height_range

    self.cvBridge = CvBridge()

    # empty initial image
    self.birdsEyeViewPub = rospy.Publisher("/mp4/BirdsEye", Image, queue_size=1)
    self.pointCloudSub = rospy.Subscriber("/velodyne_points", PointCloud2, self.__pointCloudHandler, queue_size=10)
    x_img = np.floor(-0 / self.resolution).astype(np.int32)
    self.vehicle_x = x_img - int(np.floor(self.side_range[0] / self.resolution))

    y_img = np.floor(-0 / self.resolution).astype(np.int32)
    self.vehicle_y = y_img + int(np.ceil(self.fwd_range[1] / self.resolution))


    self.x_front = float('nan')
    self.y_front = float('nan')

  def __pointCloudHandler(self, data):
    """
    Callback function for whenever the lidar point clouds are detected

    Input: data - lidar point cloud

    Output: None

    Side Effects: updates the birds eye view image
    """
    gen = point_cloud2.readgen = point_cloud2.read_points(cloud=data, field_names=('x', 'y', 'z', 'ring'))

    lidarPtBV = []
    for p in gen:
      lidarPtBV.append((p[0],p[1],p[2]))

    self.construct_birds_eye_view(lidarPtBV)

  def construct_birds_eye_view(self, data):
    """
    Call back function that get the distance between vehicle and nearest wall in given direction
    The calculated values are stored in the class member variables

    Input: data - lidar point cloud
    """
    # create image from_array
    x_max = 1 + int((self.side_range[1] - self.side_range[0]) / self.resolution)
    y_max = 1 + int((self.fwd_range[1] - self.fwd_range[0]) / self.resolution)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    if len(data) == 0:
      return im

    # Reference: http://ronny.rest/tutorials/module/pointclouds_01/point_cloud_birdseye/
    data = np.array(data)

    x_points = data[:, 0]
    y_points = data[:, 1]
    z_points = data[:, 2]

    # Only keep points in the range specified above
    x_filter = np.logical_and((x_points >= self.fwd_range[0]), (x_points <= self.fwd_range[1]))
    y_filter = np.logical_and((y_points >= self.side_range[0]), (y_points <= self.side_range[1]))
    z_filter = np.logical_and((z_points >= self.height_range[0]), (z_points <= self.height_range[1]))

    filter = np.logical_and(x_filter, y_filter)
    filter = np.logical_and(filter, z_filter)
    indices = np.argwhere(filter).flatten()

    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    def scale_to_255(a, min_val, max_val, dtype=np.uint8):
      a = (((a-min_val) / float(max_val - min_val) ) * 255).astype(dtype)
      tmp = copy.deepcopy(a)
      a[:] = 0
      a[tmp>0] = 255
      return a

    # clip based on height for pixel Values
    pixel_vals = np.clip(a=z_points, a_min=self.height_range[0], a_max=self.height_range[1])

    pixel_vals = scale_to_255(pixel_vals, min_val=self.height_range[0], max_val=self.height_range[1])

    # Getting sensor reading for front
    filter_front = np.logical_and((y_points>-2), (y_points<2))
    filter_front = np.logical_and(filter_front, x_points > 0)
    filter_front = np.logical_and(filter_front, pixel_vals > 128)
    indices = np.argwhere(filter_front).flatten()

    self.x_front = np.mean(x_points[indices])
    self.y_front = np.mean(y_points[indices])

    # convert points to image coords with resolution
    x_img = np.floor(-y_points / self.resolution).astype(np.int32)
    y_img = np.floor(-x_points / self.resolution).astype(np.int32)

    # shift coords to new original
    x_img -= int(np.floor(self.side_range[0] / self.resolution))
    y_img += int(np.ceil(self.fwd_range[1] / self.resolution))

    # Generate a visualization for the perception result
    im[y_img, x_img] = pixel_vals

    img = im.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    center = (self.vehicle_x, self.vehicle_y)
    cv2.circle(img, center, 5, (0,0,255), -1, 8, 0)

    center = self.convert_to_image(self.x_front, self.y_front)
    cv2.circle(img, center, 5, (0,255,0), -1, 8, 0)
    if not np.isnan(self.x_front) and not np.isnan(self.y_front):
        cv2.arrowedLine(img, (self.vehicle_x,self.vehicle_y), center, (255,0,0))

    x1, y1 = self.convert_to_image(20,2)
    x2, y2 = self.convert_to_image(0,-2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0))

    birds_eye_im = self.cvBridge.cv2_to_imgmsg(img, 'bgr8')

    self.birdsEyeViewPub.publish(birds_eye_im)


  def convert_to_image(self, x, y):
    """
    Convert point in vehicle frame to position in image frame
    Inputs:
        x: float, the x position of point in vehicle frame
        y: float, the y position of point in vehicle frame
    Outputs: Float, the x y position of point in image frame
    """

    x_img = np.floor(-y / self.resolution).astype(np.int32)
    y_img = np.floor(-x / self.resolution).astype(np.int32)

    x_img -= int(np.floor(self.side_range[0] / self.resolution))
    y_img += int(np.ceil(self.fwd_range[1] / self.resolution))
    return (x_img, y_img)

  def processLidar(self):
    """
    Compute the distance between vehicle and object in the front
    Inputs: None
    Outputs: Float, distance between vehicle and object in the front
    """
    front = np.sqrt(self.x_front**2+self.y_front**2)

    return front

  def get_lidar_reading(self):
    pass

""" for Final Project """
class ImageConverter:

  def __init__(self):

    # Create the cv_bridge object
    self.bridge = CvBridge()

    self.node_name = "gem_vision"
    rospy.init_node(self.node_name)
    rospy.on_shutdown(self.cleanup)

    self.image_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color", Image, self.image_callback)
    self.image_pub = rospy.Publisher("/front_camera/image_processed", Image, queue_size=1)

    self.pub_image_annotated       = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)                # [ADDED from COMPLETED studentVision]
    self.pub_bird                  = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)                # [ADDED from COMPLETED studentVision & perception]
    self.pub_color_thresh          = rospy.Publisher("lane_detection/color_thresh", Image, queue_size=1)            # [ADDED from COMPLETED studentVision]
    self.pub_gradient_thresh       = rospy.Publisher("lane_detection/gradient_thresh", Image, queue_size=1)         # [ADDED from COMPLETED studentVision]
    self.pub_perspective_transform = rospy.Publisher("lane_detection/perspective_transform", Image, queue_size=1)   # [ADDED from COMPLETED studentVision]

    # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
    # self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)

    self.left_line = Line(n=5)   # [ADDED from studentVision]
    self.right_line = Line(n=5)  # [ADDED from studentVision]
    self.detected = False        # [ADDED from studentVision]
    self.hist = True             # [ADDED from studentVision]

    """ ADDED from perception/lane_detector of MP4 """
    # initialization for the lateral tracking error and lane heading
    self.lateral_error = 0.0
    self.lane_theta = 0.0

    # determine the meter-to-pixel ratio (TODO: TUNE THIS)
    lane_width_meters = 4.4
    lane_width_pixels = 265.634
    self.meter_per_pixel = lane_width_meters / lane_width_pixels
    self.pixel_per_meter = lane_width_pixels / lane_width_meters

    # GEM vehicle speed / steer subscriber [added from gnss_tracker_pp and stanley]
    self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
    self.speed      = 0.0
    self.steer_sub = rospy.Subscriber("/pacmod/parsed_tx/steer_rpt", SystemRptFloat, self.steer_callback)
    self.steer = 0.0 # rads

    # --- tune this ---
    self.top_col_offset = 50
    self.top_row = 576 # [0,720] -> higher as you go gown
    self.bot_row = 720 # [0,720] -> higher as you go gown
    self.mid_col = 1280/2 
    self.perc_lane = 429/231 # tune
    self.top_col_offset = 0
    self.img_cols = 1280
    self.img_rows = 720
    
    # --- take measurement of 3m and have it tightly fit ---
    # ppm: pix/met, mpp: met/pix
    self.ppm_vert = (720) / 3
    self.mpp_vert = 3 / (720)

    self.row_3 = self.top_row # D_react + 3m forw
    self.row_2 = ... # D_react + 2m forw
    self.row_1 = ... # D_react + 1m forw
    self.row_0 = self.bot_row # D_react + 0m forw

    self.ppm_hori_3 = self.lanewid_pix_3 / self.lanewid_met # D_react + 3m forw
    self.ppm_hori_2 = self.lanewid_pix_2 / self.lanewid_met # D_react + 2m forw
    self.ppm_hori_1 = self.lanewid_pix_1 / self.lanewid_met # D_react + 1m forw
    self.ppm_hori_0 = self.lanewid_pix_0 / self.lanewid_met # D_react + 0m forw



    

  ''' ADDED some stuff studentVision '''
  def image_callback(self, ros_image):

    # Use cv_bridge() to convert the ROS image to OpenCV format
    try:
      frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
      rospy.logerr("CvBridge Error: {0}".format(e))

    # ----------------- Imaging processing code starts here ----------------

    pub_image = np.copy(frame)
    # mask_image, bird_image = self.detection(pub_image)  # [ADDED from Completed studentVision MP1]
    test_img = cv2.imread(pub_image)                      # [ADDED from Completed studentVision MP1]
    print("Image shape:")                                 # [ADDED from Completed studentVision MP1]
    print(test_img.shape)                                 # [ADDED from Completed studentVision MP1]


    # ----------------------------------------------------------------------

    mask_image, bird_image, lateral_error, lane_theta = self.detection(pub_image) # [ADDED from perception MP4]

    # publish the lateral tracking error and lane heading                         # [ADDED from perception MP4]
    if lateral_error is not None:
      self.lateral_error = lateral_error

    if lane_theta is not None:
      self.lane_theta = lane_theta


    try:  
        # Convert OpenCV image to ROS image and publish
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(pub_image, "bgr8"))
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    
    bird_image_test, M, Minv = self.perspective_transform(self.combinedBinaryImage(test_img))     # [ADDED from COMPLETED studentVision]
    out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image_test, '8UC1')                             # [ADDED from COMPLETED studentVision]
    self.pub_perspective_transform.publish(out_bird_msg)                                          # [ADDED from COMPLETED studentVision]

    color_image_test = self.color_thresh(test_img, (75, 205))                                     # [ADDED from COMPLETED studentVision]
    out_color_msg = self.bridge.cv2_to_imgmsg(color_image_test, '8UC1')                           # [ADDED from COMPLETED studentVision]
    self.pub_color_thresh.publish(out_color_msg)                                                  # [ADDED from COMPLETED studentVision]

    gradient_image_test = self.gradient_thresh(test_img)                                          # [ADDED from COMPLETED studentVision]
    out_gradient_msg = self.bridge.cv2_to_imgmsg(gradient_image_test, '8UC1')                     # [ADDED from COMPLETED studentVision]
    self.pub_gradient_thresh.publish(out_gradient_msg)                                            # [ADDED from COMPLETED studentVision]

    # [from studentVision]
    if mask_image is not None and bird_image is not None:                                         # [ADDED from UNCOMPLETED studentVision]
      # Convert an OpenCV image into a ROS image message
      out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')                                 # [ADDED from UNCOMPLETED studentVision]
      out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')                                # [ADDED from UNCOMPLETED studentVision]

      # Publish image message in ROS
      self.pub_image_annotated.publish(out_img_msg)                                               # [ADDED from UNCOMPLETED  studentVision]
      self.pub_bird.publish(out_bird_msg)                                                         # [ADDED from UNCOMPLETED  studentVision]
  
  ''' ADDED from COMPLETED studentVision '''
  def gradient_thresh(self, img, thresh_min=100, thresh_max=150):
    """
    Apply sobel edge detection on input image in x, y direction
    """
    if img is None:
      print("no image present")
      return -1
    

    #1. Convert the image to gray scale
    gray_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #2. Gaussian blur the image
    mat_size = 5
    gaussian_blurred = cv2.GaussianBlur(gray_scaled, (mat_size, mat_size), 0)

    #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
    ddepth = -1
    derivative_x = cv2.Sobel(gaussian_blurred, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    derivative_y = cv2.Sobel(gaussian_blurred, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    
    #4. Use cv2.addWeighted() to combine the results
    abs_gradx = cv2.convertScaleAbs(derivative_x)
    abs_grady = cv2.convertScaleAbs(derivative_y)

    # --------------- TUNE THIS ---------------  
    # Uncomment for Gazebo
    # grad = cv2.addWeighted(abs_gradx, 0.5, abs_grady, 0.5, 0)

    # Uncomment for rosbags
    # grad = cv2.addWeighted(abs_gradx, 0.75, abs_grady, 0.25, 0)

    # Uncomment for GEM e2 Hardware
    grad = cv2.addWeighted(abs_gradx, 0.75, abs_grady, 0.25, 0)
    # --------------- END TUNING ---------------  

    #5. Convert each pixel to unint8, then apply threshold to get binary image
    cvuint8 = cv2.convertScaleAbs(grad)
    ret, th = cv2.threshold(cvuint8, thresh_min, thresh_max, cv2.THRESH_BINARY)
    ####

    return th

  ''' ADDED from COMPLETED studentVision '''
  def color_thresh(self, img, thresh=(200, 255)):
    """
    Convert RGB to HSL and threshold to binary image using S/L channel
    """
    #1. Convert the image from RGB to HSL
    RGB_to_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # --------------- TUNE THIS ---------------  
    #2. Apply threshold on S/L channel to get binary image
    l_channel = RGB_to_HLS[:,:,1]
    ret, binary_output = cv2.threshold(l_channel, thresh[0], thresh[1], cv2.THRESH_BINARY)
    # s_channel = RGB_to_HLS[:,:,2]
    # ret, saturation_output = cv2.threshold(s_channel, thresh[0], thresh[1], cv2.THRESH_BINARY)
    #Hint: threshold on H to remove green grass
    # h_channel = RGB_to_HLS[:,:,0]
    # ret, hue_output = cv2.threshold(h_channel, 0, 70, cv2.THRESH_BINARY)
    # binary_output = np.logical_or(l_channel, h_channel).astype('uint8') * 255
    ####
    # --------------- END TUNING ---------------  

    return binary_output

  ''' ADDED from COMPLETED studentVision '''
  def combinedBinaryImage(self, img):
    """
    Get combined binary image from color filter and sobel filter
    """
    #1. Apply sobel filter and color filter on input image
    
    # # Uncomment for GEM
    # gem_gradient_thresh = (100, 150)
    # gem_color_thresh = (75, 205)
    # # SobelOutput = self.gradient_thresh(img)
    # SobelOutput = self.gradient_thresh(img, gem_gradient_thresh)
    # ColorOutput = self.color_thresh(img, gem_color_thresh)

    # Uncomment for rosbags
    # bag_gradient_thresh = (100, 150)
    # bag_color_thresh = (200, 255)
    # SobelOutput = self.gradient_thresh(img, bag_gradient_thresh)
    # ColorOutput = self.color_thresh(img, bag_color_thresh)

    # Uncomment for GEM e2 Hardware
    GEM_e2_hw_gradient_thresh = (100, 150)
    GEM_e2_hw_color_thresh = (200, 255)
    SobelOutput = self.gradient_thresh(img, GEM_e2_hw_gradient_thresh)
    ColorOutput = self.color_thresh(img, GEM_e2_hw_color_thresh)


    #2. Combine the outputs
    ## Here you can use as many methods as you want.
    #     
    binaryImage = np.zeros_like(SobelOutput)
    binaryImage = np.logical_or(SobelOutput, ColorOutput)
    # binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1

    #3. Remove noise from binary image
    binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)
    
    # return binaryImage.astype('uint8') * 255
    return binaryImage.astype('uint8')

  ''' ADDED from COMPLETED studentVision '''
  def perspective_transform(self, img, verbose=False):
    """
    Get bird's eye view from input image
    """
    #1. Visually determine 4 source points and 4 destination points
    #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
    #3. Generate warped image in bird view using cv2.warpPerspective()
    """
    (188, 500) (188, 750)
    (375, 248) (375, 992)
    """
    # Uncomment for rosbags
    # bag_src = np.float32([[450, 250], [0, 375], [1242, 375], [750, 250]])
    # bag_dst = np.float32([[0, 0], [0, 375], [1242, 375], [1242, 0]])
    # bag_warp = (1242, 375)
    # M = cv2.getPerspectiveTransform(bag_src, bag_dst)
    # Minv = np.linalg.inv(M)
    # warped_img = cv2.warpPerspective(img, M, bag_warp)

    # # Uncomment for Gazebo
    # gem_src = np.float32([[250, 260], [0, 400], [640, 400], [375, 260]])
    # gem_dst = np.float32([[0, 0], [0, 480], [640, 480], [640, 0]])
    # gem_warp = (640, 480)
    # M = cv2.getPerspectiveTransform(gem_src, gem_dst)
    # Minv = np.linalg.inv(M)
    # warped_img = cv2.warpPerspective(img, M, gem_warp)
    
    # Uncomment for GEM e2 Hardware
    # --------------- TUNE THIS ---------------  
    '''
    MAT = (1280, 720) (across, vertical)
    (188, 500) (188, 750)
    (375, 248) (375, 992)
    '''
    # choose row < 390:
    top_l_col  = self.mid_col - self.top_col_offset
    top_r_col = self.mid_col + self.top_col_offset

    GEM_e2_bag_src = np.float32([[top_l_col,self.top_row], [0,self.bot_row], [635,self.bot_row], [top_r_col,self.top_row]])
    GEM_e2_bag_dst = np.float32([[0, 0],    [0,self.img_rows], [self.img_cols,self.img_rows], [self.img_cols,0]])

    # --------------- END TUNING --------------- 
    GEM_e2_warp = (1280, 720)
    M = cv2.getPerspectiveTransform(GEM_e2_bag_src, GEM_e2_bag_dst)
    Minv = np.linalg.inv(M)
    warped_img = cv2.warpPerspective(img, M, GEM_e2_warp)

    return warped_img, M, Minv

  ''' ADDED ALGORITHM FROM MY HEAD, while high asf '''
  def get_weighted_err(self, ret, curr_v):

    A_l, B_l, C_l = ret['left_fit']
    A_r, B_r, C_r = ret['right_fit']

    steps = np.array([0.5, 1.0, 1.5])
    rows  = np.array([np.floor(step*curr_v*self.pixel_per_meter) for step in steps])

    # ----------------------------- Velocity Control Script START -----------------------------
    # col_l = np.array([(A_l*(y**2) + B_l*(y**1) + C_l*(y**0) ) for y in rows])
    # col_r = np.array([(A_r*(y**2) + B_r*(y**1) + C_r*(y**0) ) for y in rows])
    # col_ave = (col_l - col_r)*0.5
    # lat_errs = np.array([(self.mid_col - col)*self.meter_per_pixel for col in col_ave])
    # weights = (col_ave / np.linalg.norm(col_ave,2))
    # weighted_lat_err = np.sum(np.dot(weights,lat_errs)) / 3

    # dline_l = np.array([(2*A_l*(y**1) + B_l*(y**0) ) for y in rows])
    # dline_r = np.array([(2*A_r*(y**1) + B_r*(y**0) ) for y in rows])
    # dline_ave = (dline_l - dline_r)*0.5
    # dangles = np.array([(np.arctan(dline)) for dline in dline_ave])
    # weights = (dline_ave / np.linalg.norm(dline_ave,2))
    # weighted_lane_theta = np.sum(np.dot(weights,dangles)) / 3
    # ----------------------------- Velocity Control Script END -----------------------------

    # ----------------------------- Original Script START -----------------------------
    y = self.img_rows / 2 # middle row

    """ fuck it, lets find some lateral tracking error """
    # get the 2 sets of col (l+r), given a horizantal
    col_l = A_l*(y**2) + B_l*(y**1) + C_l*(y**0)
    col_r = A_r*(y**2) + B_r*(y**1) + C_r*(y**0) 

    """ fuck it, lets find some heading error """
    # get the derivative of the polynomial
    dp_l = 2*A_l*y + B_l
    dp_r = 2*A_r*y + B_r

    dangle_l = np.arctan(dp_l)
    dangle_r = np.arctan(dp_r)

    col_mid_ave = (col_l + col_r) / 2
    dangle_ave = (dangle_l + dangle_r) / 2

    lateral_error = (self.mid_col - col_mid_ave) * self.meter_per_pixel
    lane_theta = dangle_ave

    weighted_lat_err = lateral_error
    weighted_lane_theta = lane_theta
    # ----------------------------- Original Script END -----------------------------

    return weighted_lat_err, weighted_lane_theta

  # Get vehicle speed
  def speed_callback(self, msg):
      self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

  # Get value of steering wheel
  def steer_callback(self, msg):
      self.steer = round(np.degrees(msg.output),1)

  ''' ADDED from COMPLETED studentVision '''
  def detection(self, img):
    binary_img = self.combinedBinaryImage(img)
    img_birdeye, M, Minv = self.perspective_transform(binary_img)

    ''' sanity check on history? add a new set of Line '''
    if not self.hist:
      # Fit lane without previous result
      ret = line_fit(img_birdeye)
      left_fit = ret['left_fit']
      right_fit = ret['right_fit']
      nonzerox = ret['nonzerox']
      nonzeroy = ret['nonzeroy']
      left_lane_inds = ret['left_lane_inds']
      right_lane_inds = ret['right_lane_inds']

    else:
      # Fit lane with previous result
      if not self.detected:

        ret = line_fit(img_birdeye)

        if ret is not None:
          left_fit = ret['left_fit']
          right_fit = ret['right_fit']
          nonzerox = ret['nonzerox']
          nonzeroy = ret['nonzeroy']
          left_lane_inds = ret['left_lane_inds']
          right_lane_inds = ret['right_lane_inds']

          left_fit = self.left_line.add_fit(left_fit)
          right_fit = self.right_line.add_fit(right_fit)

          self.detected = True

        else:
          left_fit = self.left_line.get_fit()
          right_fit = self.right_line.get_fit()
          ret = tune_fit(img_birdeye, left_fit, right_fit) # get the polynomial (Ay^2 + By + C) using polyfit of the left/right lines

          if ret is not None:
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

            left_fit = self.left_line.add_fit(left_fit)
            right_fit = self.right_line.add_fit(right_fit)

          else:
            self.detected = False

      """ Annotate original image &&&&& return lane detection results (Ralph from MP4) """
      bird_fit_img = None
      combine_fit_img = None
      lateral_error = None
      lane_theta = None
      if ret is not None:
        bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
        combine_fit_img = final_viz(img, left_fit, right_fit, Minv)

        # instead of estimating the lateral tracking error and the lane heading separately at
        # different locations in the bird's eye view image as shown below, one can set a point
        # along the center line in front of the vehicle as a reference point and use a similar
        # controller as in MP2.

        # TODO :calculate the lateral tracking error from the center line
        # Hint: positive error should occur when the vehicle is to the right of the center line

        # TODO: calculate the lane heading error
        # Hint: use the lane heading a few meters before the vehicle to avoid oscillation 

        lateral_error, lane_theta = self.get_weighted_err(ret, self.speed)
        
      else:
        print("Unable to detect lanes")
      
      return combine_fit_img, bird_fit_img, lateral_error, lane_theta


  def cleanup(self):
        print ("Shutting down vision node.")
        cv2.destroyAllWindows() 

""" from MP4 """
def main(args):       
  try:
    VehiclePerception()
    rospy.spin()
  except KeyboardInterrupt:
    print ("Shutting down vision node.")
    cv2.destroyAllWindows()

if __name__ == '__main__':
  main(sys.argv)

""" FINAL project """
# def main(args):       

#   try:
#     ImageConverter()
#     rospy.spin()
#   except KeyboardInterrupt:
#     print ("Shutting down vision node.")
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#   main(sys.argv)


