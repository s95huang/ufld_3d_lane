import cv2
import numpy as np
from ultrafastLaneDetector.utils import LaneModelType
from ultrafastLaneDetector.ultrafastLaneDetectorV2 import UltrafastLaneDetectorV2
import time
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ufld_v2_ros:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)
        use_culane = rospy.get_param('~use_culane', False) # True for CULane, False for TuSimple
        if use_culane:
            self.lane_detector = UltrafastLaneDetectorV2("ufldv2_culane_res18_320x1600.onnx", LaneModelType.UFLDV2_CULANE)
        else:
            self.lane_detector = UltrafastLaneDetectorV2("ufldv2_tusimple_res18_320x800.onnx", LaneModelType.UFLDV2_TUSIMPLE)
        self.pub_visu = rospy.Publisher('ufldv2_ros', Image, queue_size=10)
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
    rospy.init_node('ufldv2_ros', anonymous=True)
    ufld_v2_ros()
    rospy.spin()