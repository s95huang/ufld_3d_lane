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

class ufld_v1_ros:
    def __init__(self):
        self.bridge = CvBridge()
        # input image topic
        input_img_topic = rospy.get_param('~input_image_topic', '/camera/color/image_raw')
        # output visualization topic
        output_visu_topic = rospy.get_param('~output_image_topic', '/lane_2d/visual')
        output_lane_msg_topic = rospy.get_param('~output_lane_2d_topic', '/lane_2d/lane_msg')

        
        self.image_sub = rospy.Subscriber(input_img_topic,Image,self.callback,queue_size=10)
        self.pub_visu = rospy.Publisher(output_visu_topic,Image,queue_size=10)
        self.lane_msg_pub = rospy.Publisher(output_lane_msg_topic,Image,queue_size=10)
        
        use_culane = rospy.get_param('~use_culane', False) # True for CULane, False for TuSimple
        # get model path
        model_path = rospy.get_param('~model_path', 'tusimple_288x800.onnx')
                
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
        output_img = self.lane_detector.AutoDrawLanes(self.frame)

        self.pub_visu.publish(self.bridge.cv2_to_imgmsg(output_img, "bgr8"))


if __name__ == '__main__':
    rospy.init_node('ufldv1_ros', anonymous=True)
    ufld_v1_ros()
    rospy.spin()