#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:23:55 2022

@author: minghao
"""

import os, cv2, sys
import numpy as np
from data.constant import culane_row_anchor, tusimple_row_anchor
import time

from tracker.LaneUpdaterWC import Lane_updater

###ROS Things
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

#import tf
from std_msgs.msg import ColorRGBA, Float32

from lane_detection_msgs.msg import CNNLaneDetectionMsg, LaneDetectionMsgFull


def draw_one_line_strip(x_wc_to_draw, y_wc_to_draw, rgba=(1.0,0,0,1), lane_id=0, is_line_strip = True, color_weight=[]):
    lane_line_strip = Marker()
    lane_line_strip.header.frame_id = 'rslidar_front'
    lane_line_strip.action = Marker.ADD
    lane_line_strip.pose.orientation.w = 1
    lane_line_strip.id = lane_id
    if is_line_strip:
        lane_line_strip.type = Marker.LINE_STRIP
    else:
        lane_line_strip.type = Marker.POINTS
    lane_line_strip.scale.x = 0.1
    lane_line_strip.color.r = rgba[0]
    lane_line_strip.color.g = rgba[1]
    lane_line_strip.color.b = rgba[2]
    lane_line_strip.color.a = rgba[3]
    
    for i in range( len(y_wc_to_draw) ):
        i_point = Point()
        i_wcx = x_wc_to_draw[i]
        i_wcy = y_wc_to_draw[i]
        i_point.x = i_wcx
        i_point.y = i_wcy
        lane_line_strip.points.append(i_point)
    
        if len(color_weight):
            i_color = ColorRGBA()
            i_color.r = rgba[0] * color_weight[i]
            i_color.g = rgba[1] * color_weight[i]
            i_color.b = rgba[2] * color_weight[i]
            i_color.a = rgba[3]
            lane_line_strip.colors.append(i_color)
    return lane_line_strip
    
def rot2bus(x, y, rot_heading_deg=-3, half_length=2):
    if np.all(x==0) or (rot_heading_deg==0 and half_length==0):
        return x, y
    else:
        rot_heading = np.deg2rad(rot_heading_deg)
        cos_head = np.cos(rot_heading)
        sin_head = np.sin(rot_heading)
        rot_matrix = np.array( [ [cos_head, sin_head], [-sin_head, cos_head] ] )
        xy = np.vstack([x,y]).T
        rot_xy = xy @ rot_matrix
        rot_x = rot_xy[:,0]+half_length
        rot_y = rot_xy[:,1]
        poly_f = np.poly1d(np.polyfit(rot_x, rot_y, 2))
        new_x = np.linspace(0, 20 , 21)
        new_y = poly_f(new_x)
        return new_x, new_y
    
    
def draw_visualization_and_pub(vis, out_j, updater):    
    # rot_heading_deg = 2
    # half_length = 2
    
    rot_heading_deg = 0
    half_length = 0
    
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                    cv2.circle(vis,ppp,3,(255,0,0),-1)        
    
    # Draw horizon line
    h = int(updater.Horizon_pos.get_refined_result()[0,0])
    cv2.line(vis,(0,h), (img_w,h),(0,0,255),4)  
    
    if len(updater.Memo_Points_in_WC_left):
        x_wc_to_draw = updater.Memo_Points_in_WC_left[:,1] ##Memo coord: (0, right), (1, front)
        y_wc_to_draw = -updater.Memo_Points_in_WC_left[:,0]
        left_lane_msg = draw_one_line_strip(x_wc_to_draw, y_wc_to_draw, rgba=(1,0,0,1), lane_id=2, is_line_strip=False, color_weight=updater.Weight_left)
        lane_marker_pub.publish(left_lane_msg)
    if len(updater.Memo_Points_in_WC_right):
        x_wc_to_draw = updater.Memo_Points_in_WC_right[:,1]
        y_wc_to_draw = -updater.Memo_Points_in_WC_right[:,0]
        right_lane_msg = draw_one_line_strip(x_wc_to_draw, y_wc_to_draw, rgba=(0,1,0,1), lane_id=3, is_line_strip=False, color_weight=updater.Weight_right)
        lane_marker_pub.publish(right_lane_msg)
        
    if len(updater.bounded_left_CP) > 0:
        for i in range(len(updater.bounded_left_CP)):
            ppp = (int(updater.bounded_left_CP[i,0]), int(updater.bounded_left_CP[i,1]) )
            cv2.circle(vis,ppp,5,(0,0,255),-1)  
    if len(updater.bounded_right_CP) > 0:
        for i in range(len(updater.bounded_right_CP)):
            ppp = (int(updater.bounded_right_CP[i,0]), int(updater.bounded_right_CP[i,1]) )
            cv2.circle(vis,ppp,5,(0,0,255),-1)  
            
            
    x_wc_to_draw = np.linspace(0, 20 , 21)
    # y_wc_to_draw = np.linspace(0, 20, 21)
    
    ret_lane_detection_msg = LaneDetectionMsgFull()
    is_good_detection = updater.has_initialized & (updater.continuous_no_detect_distance < updater.max_no_detection_distance)
    #TODO: has initialized ? no detection distance < 5?
    if (updater.left_fit_wc is not None) and is_good_detection:
        x_wc_to_draw_ = x_wc_to_draw.copy()
        left_f_wc = np.poly1d(updater.left_fit_wc)
        y_wc_to_draw_ = -left_f_wc(x_wc_to_draw_-updater.z_for_keep_distance)
        ret_lane_detection_msg.PolyFitC2 = -updater.left_fit_wc[0]
        ret_lane_detection_msg.PolyFitC1 = -updater.left_fit_wc[1]
        ret_lane_detection_msg.PolyFitC0 = -updater.left_fit_wc[2]
        ret_lane_detection_msg.LateralDis = -updater.left_fit_wc[2]
        ret_lane_detection_msg.Confidence = updater.selected_left_percent
    else:
        y_wc_to_draw_ = np.zeros_like(x_wc_to_draw)
        x_wc_to_draw_ = np.zeros_like(x_wc_to_draw)
    rot_x_centerlane, rot_y_centerlane = rot2bus(x_wc_to_draw_, y_wc_to_draw_, rot_heading_deg=rot_heading_deg, half_length=half_length)
    ret_lane_detection_msg.CenterLaneX = rot_x_centerlane
    ret_lane_detection_msg.CenterLaneY = rot_y_centerlane

    
    ret_lane_detection_msg.CenterLineX = rot_x_centerlane
    if np.all(rot_x_centerlane==0):
        ret_lane_detection_msg.CenterLineY = rot_y_centerlane
    else:
        ret_lane_detection_msg.CenterLineY = rot_y_centerlane - 1.5
        
    left_lane_msg = draw_one_line_strip(x_wc_to_draw_, y_wc_to_draw_, rgba=(1,1,0,1), lane_id=0)
    lane_marker_pub.publish(left_lane_msg)
        
    center_line_msg = draw_one_line_strip(x_wc_to_draw_, y_wc_to_draw_-1.5, rgba=(0,0,1,1), lane_id=9)
    lane_marker_pub.publish(center_line_msg)
    
    # ### applanix coord
    # rot_center_line_msg = draw_one_line_strip(rot_y_centerlane - 1.5, rot_x_centerlane, rgba=(1,0,0,1), lane_id=10)
    # rot_center_line_msg.header.frame_id = 'applanix'
    # lane_marker_pub.publish(rot_center_line_msg)

    if (updater.right_fit_wc is not None) and is_good_detection:
        x_wc_to_draw_ = x_wc_to_draw.copy()
        right_f_wc = np.poly1d(updater.right_fit_wc)
        y_wc_to_draw_ = -right_f_wc(x_wc_to_draw_-updater.z_for_keep_distance)
    else:
        y_wc_to_draw_ = np.zeros_like(x_wc_to_draw)
        x_wc_to_draw_ = np.zeros_like(x_wc_to_draw)
    
    rot_x_rightlane, rot_y_rightlane = rot2bus(x_wc_to_draw_, y_wc_to_draw_, rot_heading_deg=rot_heading_deg, half_length=half_length)
    ret_lane_detection_msg.RightCurbX = rot_x_rightlane
    ret_lane_detection_msg.RightCurbY = rot_y_rightlane
    
    right_lane_msg = draw_one_line_strip(x_wc_to_draw_, y_wc_to_draw_, rgba=(0,1,0,1), lane_id=1)
    lane_marker_pub.publish(right_lane_msg)
    
    lane_detection_pub.publish(ret_lane_detection_msg)
    
    if is_good_detection:
        if len(updater.Points_in_CP_left_for_draw) > 0:
            left_x_estimated = updater.Points_in_CP_left_for_draw[:,0]
            left_y_estimated = updater.Points_in_CP_left_for_draw[:,1]
            for i in range(len(left_y_estimated)):
                ppp = (int(left_x_estimated[i]), int(left_y_estimated[i]) )
                cv2.circle(vis,ppp,5,(0,255,0),-1)    
                
        if len(updater.Points_in_CP_right_for_draw) > 0:
            right_x_estimated = updater.Points_in_CP_right_for_draw[:,0]
            right_y_estimated = updater.Points_in_CP_right_for_draw[:,1]
            for i in range(len(right_y_estimated)):
                ppp = (int(right_x_estimated[i]), int(right_y_estimated[i]) )
                cv2.circle(vis,ppp,5,(0,255,0),-1)   
    
    cur = np.nan
    direct = np.nan
    left_pos = np.nan
    right_pos = np.nan
    
    if is_good_detection:
        cur_direction = updater.Lane_cur_direction.get_refined_result()
        cur_direction_cov = updater.Lane_cur_direction.kf.errorCovPost
        if cur_direction is not None:
            cur = cur_direction[0,0]
            direct = cur_direction[1,0]
            
        left_pos = updater.Left_lane_pos.get_refined_result()[0,0]
        left_pos_cov = updater.Left_lane_pos.kf.errorCovPost[0,0]
        right_pos = updater.Right_lane_pos.get_refined_result()[0,0]
        right_pos_cov = updater.Right_lane_pos.kf.errorCovPost[0,0]
    
    # print('Curve: %.2f,   Angle: %.2f,    DisL: %.2f,    DisR: %.2f '%( cur * 1e3, np.rad2deg(direct), left_pos, right_pos) )
    pub_plot_cur.publish(-cur * 1e3)
    pub_plot_dir.publish(-np.rad2deg(direct))
    pub_plot_posl.publish(-left_pos)
    pub_plot_posr.publish(-right_pos)
    
    # print('Left  Pos and Cov', left_pos, left_pos_cov)
    # print('Right Pos and Cov', right_pos, right_pos_cov)
    # print('Cur Direction and Cov', cur_direction, cur_direction_cov)
    
    #publish visualization
    # cv2.imshow('LaneDetectionResult', vis)
    # cv2.waitKey(2)
    if is_compress_the_image:
        out_msg = cvbridge.cv2_to_compressed_imgmsg(vis)
    else:
        out_msg = cvbridge.cv2_to_imgmsg(vis)
    out_pub.publish(out_msg)
    

cnn_out_msg_test = None
def cnn_out_callback(cnn_out_msg):
    global cnn_out_msg_test
    cnn_out_msg_test = cnn_out_msg
    print('get one cnn result')
    img_msg = cnn_out_msg.front_camera_image
    if img_msg.height > 0:
        image = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
    else:
        image = np.zeros( (img_h, img_w, 3), np.uint8 )
    
    phi_imu = cnn_out_msg.phi_imu
    delta_heading = cnn_out_msg.delta_heading
    delta_distance = cnn_out_msg.delta_distance
    start_time = cnn_out_msg.start_time
    cnn_time = cnn_out_msg.cnn_time
    
    ros_trans_time = rospy.get_rostime().to_sec()
    out_j = np.array(cnn_out_msg.out_cnn).reshape( (cnn_out_msg.out_cnn_height, cnn_out_msg.out_cnn_width) )

    updater.update(out_j, delta_distance, delta_heading, phi_imu)  #need change
    
    tracker_time = rospy.get_rostime().to_sec()
    vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Draw visualization things and publish them
    draw_visualization_and_pub(vis, out_j, updater)
    end_time = rospy.get_rostime().to_sec()
    
    pub_runtime_plot_all.publish(1e3*(end_time-start_time))
    pub_runtime_plot_cnn.publish(1e3*(cnn_time-start_time))
    pub_runtime_plot_ros_trans.publish(1e3*(ros_trans_time-cnn_time))
    pub_runtime_plot_tracker.publish(1e3*(tracker_time-ros_trans_time))
    

    

if __name__ == "__main__":    
    #### Start Ros Communication:
    cvbridge = CvBridge()
    rospy.init_node("Post_Camera_Lane_Detection_Node")
    print("initial lane detection post process node")
    
    # is_compress_the_image = True
    is_compress_the_image = False

    
    row_anchor = np.array(tusimple_row_anchor) 
    col_sample = np.linspace(0, 800 - 1, 100)
    col_sample_w = col_sample[1] - col_sample[0]
    cls_num_per_lane = 56
            
    img_w, img_h = 1920, 1200
    updater = Lane_updater( (img_h, img_w), col_sample_w, row_anchor, init_wait_length=3, max_no_detection_distance=5)
    

    cnn_out_sub = rospy.Subscriber("/lane_detection_result/CNN", CNNLaneDetectionMsg, cnn_out_callback, queue_size=1, buff_size=2**24)
    
    lane_detection_pub = rospy.Publisher("/Lane_Detection_Result", LaneDetectionMsgFull, queue_size=10)
    
    if is_compress_the_image:
        out_pub = rospy.Publisher("/lane_detection_result/PostProcess/compressed", CompressedImage, queue_size=1)
    else:
        out_pub = rospy.Publisher("/lane_detection_result/PostProcess", Image, queue_size=1)
    
    
    lane_marker_pub = rospy.Publisher('/lane_marker', Marker, queue_size=10)
    pub_runtime_plot_all = rospy.Publisher("/LaneDetectionRunTimePlot_all", Float32 ,queue_size=1)
    pub_runtime_plot_cnn = rospy.Publisher("/LaneDetectionRunTimePlot_CNN", Float32 ,queue_size=1)
    pub_runtime_plot_ros_trans = rospy.Publisher("/LaneDetectionRunTimePlot_RosTrans", Float32 ,queue_size=1)
    pub_runtime_plot_tracker = rospy.Publisher("/LaneDetectionRunTimePlot_tracker", Float32 ,queue_size=1)
    
    pub_plot_cur = rospy.Publisher("/LaneDetection/Curve", Float32 ,queue_size=1)
    pub_plot_dir = rospy.Publisher("/LaneDetection/Direction", Float32 ,queue_size=1)
    pub_plot_posl = rospy.Publisher("/LaneDetection/PositionLeft", Float32 ,queue_size=1)
    pub_plot_posr = rospy.Publisher("/LaneDetection/PositionRight", Float32 ,queue_size=1)
    rospy.spin()
    
    
    
    