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
import rosbag
### Tracker
cur_file_dir = os.path.dirname(os.path.realpath(__file__))
### RingroadMap
sys.path.append(os.path.join(cur_file_dir, 'tracker'))
#sys.path.append('tracker')
from tracker.LaneUpdaterWC import Lane_updater

def draw_visualization_and_show(vis, out_list, updater):    
    rot_heading_deg = 1.5
    half_length = 3.2
    for lane in out_list:
        for col_row_confi in lane:
            ppp = (int(col_row_confi[0]), int(col_row_confi[1]))
            cv2.circle(vis, ppp, 3, (255, 0, 0), -1)
    
    
    # Draw horizon line
    h = int(updater.Horizon_pos.get_refined_result()[0,0])
    cv2.line(vis,(0,h), (img_w,h),(0,0,255),4)  
    
    if len(updater.bounded_left_CP) > 0:
        for i in range(len(updater.bounded_left_CP)):
            ppp = (int(updater.bounded_left_CP[i,0]), int(updater.bounded_left_CP[i,1]) )
            cv2.circle(vis,ppp,5,(0,0,255),-1)  
    if len(updater.bounded_right_CP) > 0:
        for i in range(len(updater.bounded_right_CP)):
            ppp = (int(updater.bounded_right_CP[i,0]), int(updater.bounded_right_CP[i,1]) )
            cv2.circle(vis,ppp,5,(0,0,255),-1)  
            
    is_good_detection = updater.has_initialized & (updater.continuous_no_detect_distance < updater.max_no_detection_distance)
    
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
    cv2.imshow("vis", vis)  
    cv2.waitKey(1)
    

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
    
    pitch_imu = cnn_out_msg.phi_imu
    delta_heading = cnn_out_msg.delta_heading # make sure the heading is positive when turning right
    delta_distance = cnn_out_msg.delta_distance
    start_time = cnn_out_msg.start_time
    cnn_time = cnn_out_msg.cnn_time
    
    out_j = np.array(cnn_out_msg.out_cnn).reshape( (cnn_out_msg.out_cnn_height, cnn_out_msg.out_cnn_width) )
    out_list = preprocess_cnn_result(out_j)
    updater.update(out_list, delta_distance, delta_heading, pitch_imu)  #need change
    if image_np is not None:
        draw_visualization_and_show(image_np, out_list, updater)
    
    
def preprocess_cnn_result(out_cnn):
    # out_cnn: (56, 4) means 4 lanes, each lane has 56 column points in (288,800) coordinate
    # output: a list of 4 lanes, each lane is a list of (col, row, confidence) in (1200, 1920) coordinate
    output_list = []
    for i in range(4):
        lane_col = out_cnn[:, i]
        lane_list = []
        for j in range(cls_num_per_lane):
            if lane_col[j] > 0:
                lane_list.append( (colum_factor_to_original_image*lane_col[j], row_anchor_in_original_image[j], 1) )
            lane_arr = np.array(lane_list)
        output_list.append(lane_arr)
    return output_list


if __name__ == "__main__":    
    ## Load one rosbag
    bag = rosbag.Bag('/Users/minghao/Downloads/small1.bag')
    # is_compress_the_image = True
    is_compress_the_image = False

    # image detection result transformation parameters
    img_w, img_h = 1920, 1200
    row_anchor = np.array(tusimple_row_anchor, dtype=np.float32) 
    row_anchor_in_original_image = row_anchor[::-1] * img_h / 288.0
    col_sample = np.linspace(0, 800 - 1, 100)
    col_sample_w = col_sample[1] - col_sample[0]
    colum_factor_to_original_image = col_sample_w * img_w / 800.0
    cls_num_per_lane = 56
    
    # lane updater
    updater = Lane_updater( (img_h, img_w), init_wait_length=3, max_no_detection_distance=5)

    count = 0
    cnn_msg = None
    img_msg = None
    image_np = None
    for topic, msg, t in bag.read_messages():
        count += 1
        if topic == 'lane_detection_result/CNN':
            cnn_msg = msg
            cnn_out_callback(cnn_msg)

        elif topic == 'pylon_camera_node_center/image_rect/compressed':
            img_msg = msg
            np_arr = np.fromstring(img_msg.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        



        
    

    