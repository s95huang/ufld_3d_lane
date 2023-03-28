#!/usr/bin/env python3

# general imports
import cv2
import numpy as np
import time

# local imports
from ultrafastLaneDetector.utils import LaneModelType
from ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector

# ros imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from lane_detection_msgs.msg import CNNLaneDetectionMsg # mvsl lane detection msg
from loopx_lane_detection_msgs.msg import lane_2d, lane_2d_array # loopx lane detection msg

class ufld_v1_ros:
    def __init__(self):
        self.bridge = CvBridge()
        # input image topic
        input_img_topic = rospy.get_param('~input_image_topic', '/camera/color/image_raw')
        # output visualization topic
        output_visu_topic = rospy.get_param('~output_image_topic', '/lane_2d/visual')
        mvsl_lane_output_topic = rospy.get_param('~mvsl_lane_output_topic', '/lane_2d/lane_msg')
        loopx_lane_output_topic = rospy.get_param('~loopx_lane_output_topic', '/lane_2d/lane_msg')

        # By default, lane detection results are published using both lane_detection_msgs and loopx_lane_detection_msgs
        use_mvsl_msg = rospy.get_param('~use_mvsl_msg', True) # True for mvsl lane detection msg
        use_loopx_msg = rospy.get_param('~use_loopx_msg', True) # True for loopx lane detection msg

        self.image_sub = rospy.Subscriber(input_img_topic,Image,self.callback,queue_size=10)
        self.pub_visu = rospy.Publisher(output_visu_topic,Image,queue_size=10)
        
        use_culane = rospy.get_param('~use_culane', False) # True for CULane, False for TuSimple
        # get model path
        model_path = rospy.get_param('~model_path', 'tusimple_288x800.onnx')

        # set up lane_detection msg publisher
        if use_mvsl_msg:
            self.pub_lane_msg = rospy.Publisher(mvsl_lane_output_topic, CNNLaneDetectionMsg, queue_size=10)
        if use_loopx_msg:
            self.pub_lane_msg = rospy.Publisher(loopx_lane_output_topic, lane_2d_array, queue_size=10)
                
        if use_culane:
            file_name = 'culane_288x800.onnx'
            # replace tusimple_288x800 with culane_288x800
            model_path = model_path.replace('tusimple_288x800.onnx', file_name)
            self.lane_detector = UltrafastLaneDetector(model_path, LaneModelType.UFLD_CULANE)
        else:
            # no change needed, default is TuSimple
            self.lane_detector = UltrafastLaneDetector(model_path, LaneModelType.UFLD_TUSIMPLE)
        # self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.vout = cv2.VideoWriter('highway-10364_out.mp4', self.fourcc , 30.0, (800, 320))
        self.fps = 0
        self.frame_count = 0
        self.start = time.time()
        self.end = time.time()
        self.frame = None

    def callback(self,data):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Detect the lanes
        # output is network output, output_img is the visualization image with lanes drawn on it
        output, output_img, lanes_points, lanes_detected, lanes_confidence = self.lane_detector.AutoDrawLanes(self.frame)

        # print("=====================================")
        # print(lanes_points)
        # print(lanes_detected)

        # get output dimensions
        # print(output[0].shape)
        # (1, 101, 56, 4)

        print(lanes_confidence)


        self.pub_visu.publish(self.bridge.cv2_to_imgmsg(output_img, "bgr8"))

        # CNNLaneDetectionMsg
        mvsl_lane_msg = CNNLaneDetectionMsg()
        # mvsl_lane_msg.header = data.header # copy header from input image, most important is the timestamp

        # get index of the lane 


if __name__ == '__main__':
    rospy.init_node('ufldv1_ros', anonymous=True)
    ufld_v1_ros()
    rospy.spin()