#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:51:41 2021

@author: minghao

LaneUpdaterWC: in this version, I will try to fit the left and right lanes on the world coordinate.
And there is no another Kalman Filter to limit the change of hyperbola
"""

import cv2
import numpy as np
from math import *
# fix ModuleNotFoundError: No module named 'CameraConfig'
# from Camera_Config import Camera
# from KalmanFilter import Kalman
from sklearn import linear_model
import time

class Kalman(object):
    def __init__(self, num_para, proc_noise_scale_arr, meas_noise_scale_arr, use_control=False):
        self.state_size = num_para
        self.meas_size = num_para
        if use_control:
            self.contr_size = num_para
        else:
            self.contr_size = 0
        
        self.proc_noise_scale_arr = proc_noise_scale_arr
        self.meas_noise_scale_arr = meas_noise_scale_arr
        self.kf = cv2.KalmanFilter(self.state_size, self.meas_size, self.contr_size)
        self.kf.transitionMatrix = np.eye( self.state_size, dtype=np.float32 )
        self.kf.measurementMatrix = np.eye( self.state_size, dtype=np.float32 )
        self.kf.controlMatrix = np.eye( self.contr_size, dtype=np.float32 ) if self.contr_size > 0 else None
        
        self.processNoiseBase = np.diag( self.proc_noise_scale_arr ).astype(np.float32)
        self.kf.processNoiseCov = self.processNoiseBase.copy()
        
        self.measurementNoiseBase = np.diag( self.meas_noise_scale_arr ).astype(np.float32)
        self.kf.measurementNoiseCov = self.measurementNoiseBase.copy()
        
        self.kf.errorCovPre = np.diag( self.meas_noise_scale_arr ).astype(np.float32)
        self.kf.errorCovPost = np.diag( self.meas_noise_scale_arr ).astype(np.float32)
        
        self.first_detected = False
        
    def _first_detect(self, _vars):
        self.kf.statePost = _vars.copy()
        self.first_detected = True
        
    def update(self, _vars, proc_noise_arr = None, meas_noise_arr = None, u=None):
        self.kf.predict(u)
        if proc_noise_arr is not None:
            tmp_proc_noise = self.kf.processNoiseCov.copy()
            self.kf.processNoiseCov = np.diag( proc_noise_arr ).astype(np.float32)
        if meas_noise_arr is not None:
            tmp_meas_noise = self.kf.measurementNoiseCov.copy()
            self.kf.measurementNoiseCov = np.diag( meas_noise_arr ).astype(np.float32)
            
        if self.first_detected:
            self.kf.correct(_vars.copy())
        else:
            self._first_detect( _vars )
            
        if proc_noise_arr is not None:
            self.kf.processNoiseCov = tmp_proc_noise
        if meas_noise_arr is not None:
            self.kf.measurementNoiseCov = tmp_meas_noise
        
    def get_refined_result(self):
        if self.first_detected:
            return np.array( self.kf.statePost )
        return None

class Camera(object):
    def __init__(self, input_shape):        
        self.input_shape = input_shape  #height, width
        self.xc = 926.05
        self.yc = 511.34
        self.fx = 1288.175
        self.fy = 1444.18
        # self.fy = 1344.18
        
        self.K = np.matrix( [ [self.fx, 0, self.xc], [0, self.fy, self.yc], [0,0,1] ] )
        self.h = 2.1      #unit: m
        self.phi_road = None     #unit: degree  right hand cood
        self.phi_t_offset = -3 #CHANGE:the camera is installed a little downtowards
        self.phi_t = self.phi_t_offset
        
        # self.gamma_offset = 0.4 #CHANGE
        self.gamma_offset = 0 #CHANGE
        self.gamma = self.gamma_offset
        cos_gamma = cos(radians(self.gamma))
        sin_gamma = sin(radians(self.gamma))
        self.Ry_matrix_still = np.matrix( [ [cos_gamma, 0 , -sin_gamma],
                                [0, 1, 0], [sin_gamma, 0, cos_gamma] ] )
        
        self.vector_c = np.matrix( [0, -self.h, 0] ).T
        self.Projection_matrix = self.calculate_projection_matrix()
        self.Projection_matrix_still = self.Projection_matrix
        self.phi_road_update_rate = 0.06
        

    def update_phi_road(self, phi_imu):
        if self.phi_road is None:
            self.phi_road = phi_imu - self.phi_t_offset
        else:
            self.phi_road = (1-self.phi_road_update_rate)*self.phi_road + self.phi_road_update_rate*(phi_imu-self.phi_t_offset)
           
            
    def update_camera_phi(self, phi_imu, horizon):
        self.update_phi_road(phi_imu)
        phi_t_from_imu = phi_imu - self.phi_road
        phi_t_from_horizon = degrees( atan( (horizon-self.yc)/self.fy ) )
        self.phi_t = 0.7*phi_t_from_horizon + 0.3*phi_t_from_imu
        refined_horizon = tan( radians(self.phi_t) )*self.fy + self.yc
        return refined_horizon
    
    
    def update_camera_state(self, delta_gamma, delta_x, delta_z):
        self.gamma = self.gamma_offset + delta_gamma
        self.vector_c[0,0] = delta_x
        self.vector_c[2,0] = delta_z
        self.Projection_matrix = self.calculate_projection_matrix()
    
    
    def calculate_projection_matrix(self):
        T_matrix = np.eye(4)[:3]
        for i in range(3): 
            T_matrix[i,3] = -self.vector_c[i,0]
        T_matrix = np.matrix(T_matrix)
        
        cos_gamma = cos(radians(self.gamma))
        sin_gamma = sin(radians(self.gamma))
        Ry_matrix = np.matrix( [ [cos_gamma, 0 , -sin_gamma],
                                [0, 1, 0], [sin_gamma, 0, cos_gamma] ] )
        
        cos_phi_t = cos(radians(self.phi_t))
        sin_phi_t = sin(radians(self.phi_t))
        Rx_matrix = np.matrix( [ [1, 0, 0], [0, cos_phi_t, sin_phi_t], [0, -sin_phi_t, cos_phi_t] ] )
        
        Projection_matrix = self.K * Rx_matrix * Ry_matrix * T_matrix
        
        T_matrix_still = np.eye(4)[:3]
        T_matrix_still[1,3] = -self.vector_c[1,0]
        T_matrix_still = np.matrix(T_matrix_still)
        self.Projection_matrix_still = self.K * Rx_matrix * self.Ry_matrix_still * T_matrix_still
        return Projection_matrix
    
    
    def project_from_WC_to_CP_still(self, points):
        #  point[0] = x, point[1] = z
        Projection_matrix = self.Projection_matrix_still
        if len(points) == 0:
            return np.array([[],[]]).T
        else:
            uv_list = []
            for point in points:
                homo_point = np.matrix([point[0], 0, point[1], 1]).T
                uv_vector = Projection_matrix * homo_point
                u,v = uv_vector[0,0]/uv_vector[2,0], uv_vector[1,0]/uv_vector[2,0]
                uv_list.append( (u,v) )
            return np.array(uv_list)
    
    
    def project_from_WC_to_CP(self, points):
        #  point[0] = x, point[1] = z
        Projection_matrix = self.Projection_matrix
        if len(points) == 0:
            return np.array([[],[]]).T
        else:
            uv_list = []
            for point in points:
                homo_point = np.matrix([point[0], 0, point[1], 1]).T
                uv_vector = Projection_matrix * homo_point
                u,v = uv_vector[0,0]/uv_vector[2,0], uv_vector[1,0]/uv_vector[2,0]
                uv_list.append( (u,v) )
            return np.array(uv_list)
    
    
    def project_from_CP_to_WC(self, points):
        # point[0] = u, point[1] = v
        K_inv = self.K.I
        cos_phi_t = cos(radians(self.phi_t))
        sin_phi_t = sin(radians(self.phi_t))
        Rx_matrix = np.matrix( [ [1, 0, 0], [0, cos_phi_t, sin_phi_t], [0, -sin_phi_t, cos_phi_t] ] )
        R_inv = (Rx_matrix * self.Ry_matrix_still).I
        R_inv_K_inv = R_inv * K_inv
        
        if len(points) == 0:
            return np.array([[],[]]).T
        else:
            xz_list = []
            for point in points:
                M_vector = R_inv_K_inv * np.matrix( [point[0], point[1], 1] ).T
                s = self.h / M_vector[1,0]
                x,z = s * M_vector[0,0], s * M_vector[2,0]
                xz_list.append( (x,z) )
            return np.array(xz_list)


class Lane_updater(object):   
    def __init__(self,shape, col_sample_w, row_anchor, init_wait_length=4, max_no_detection_distance=5):
        self.input_shape = shape #h,w
        self.col_sample_w = col_sample_w
        self.row_anchor_in_detection = row_anchor
        self.row_anchor_in_original_image = row_anchor * self.input_shape[0] / 288
        
        # parameters for scanline search and blob detection to find candidate points
        # self.scanline_y_start = self.input_shape[0]*0.8 #CHANGE
        # self.scanline_y_end = self.input_shape[0]*0.45 #CHANGE
        self.scanline_y_start = max(self.row_anchor_in_original_image) #CHANGE
        self.scanline_y_end = min(self.row_anchor_in_original_image) #CHANGE
        
        self.Margin_for_search_around_poly = 0.6 #1 meter
        self.Number_of_scanline = len(row_anchor)
        self.Stride_between_each_line = (self.scanline_y_start - self.scanline_y_end) / self.Number_of_scanline
        
        self.Reset_no_detect_count_threshold = 3
        self.init_wait_length = init_wait_length
        self.max_no_detection_distance = max_no_detection_distance
        self.reset()
        
    def reset(self):
        # camera class, used to apply perspective transform based on movement
        self.cam = Camera(self.input_shape)
        self.horizon_init_pos = np.matrix( self.input_shape[0] * 0.34 ,dtype=np.float32) #CHANGE
        self.Horizon_pos = Kalman(1, [1], [10])
        self.Horizon_pos.update( self.horizon_init_pos )
        
        # WC:stored lane points in world coordiantes to car; column 0:x,lateral distance; column 1:z,longitudal distance
        # CP:stored lane points in Camera Plane; column 0:x, pixel number from left to right; column 1:y, pixel number from top to down
        self.Memo_Points_in_WC_right = np.array([[],[]]).T
        self.Memo_Points_in_WC_left = np.array([[],[]]).T
        self.Memo_Points_in_CP_left = np.array([[],[]]).T
        self.Memo_Points_in_CP_right = np.array([[],[]]).T
        
        self.Points_in_CP_left_for_draw = np.array([[],[]]).T
        self.Points_in_CP_right_for_draw = np.array([[],[]]).T
        
        # weights used when fitting hyperbola
        self.Weight_left = np.array([])
        self.Weight_right = np.array([])
        
        # from motion: old lane points in world coordiantes to the car transformed using car's movement
        self.Points_from_motion_left = np.array([[],[]]).T
        self.Points_from_motion_right = np.array([[],[]]).T
        self.Points_from_motion_left_draw = np.array([[],[]]).T
        self.Points_from_motion_right_draw = np.array([[],[]]).T
        
        
        # f:hyperbola function, used when reduce the searching region, and update points for draw
        self.left_f = None
        self.right_f = None
        self.left_fit_wc = None
        self.right_fit_wc = None
        
        # some parameters for rejecting unconfident detection
        self.Min_num_for_update_horizon = 16
        self.Min_num_for_update_hyperbola = 30
        self.Min_wc_distance_for_update_hyperbola = 3
        self.Max_detect_error_rate = self.Number_of_scanline / self.Min_num_for_update_hyperbola
        self.Min_num_for_store_points = 20
        
        # some parameters for limiting outlier's influence
        self.Max_update_step_for_horizon = 50
        self.Max_update_step_for_lane_pos = 0.5
        self.Max_update_step_for_cur = 1/300
        self.Max_update_step_for_direction = np.deg2rad(1.2)
        
        self.Hyperbola_para = Kalman( 4, [0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5] )
        self.Left_lane_pos = Kalman(1, [0.002], [0.04])
        self.Left_lane_pos.update( np.matrix(-2 ,dtype=np.float32) )
        self.Right_lane_pos = Kalman(1, [0.003], [0.04])
        self.Right_lane_pos.update( np.matrix(2.3 ,dtype=np.float32) )
        self.Lane_cur_direction = Kalman(2, [0.01, 0.003], [0.3, 0.3])
        
        self.z_for_keep_distance = 2 # when fitting parabola, minus x with this constant can help get a stable lateral distance
        
        self.Lane_detection_noise_scale_base = np.array( [0.2, 0.2, 0.1] ) * 0.15 # decide how confident to trust this detection
        
        self.travel_distance_for_not_detect_whole_lanes = 0
        self.travel_distance_for_not_detect_left_lane = 10
        self.travel_distance_for_not_detect_left_lane_count = 0
        self.travel_distance_for_not_detect_right_lane = 10
        
        self.has_initialized = False
        self.recent_detect_list = [0] * self.init_wait_length
        self.continuous_no_detect_distance = 0
        self.selected_left_percent = 0
        
        self.bounded_left_CP = np.array([[],[]]).T
        self.bounded_right_CP = np.array([[],[]]).T
        
        # self.left_lane_fitting_model = linear_model.HuberRegressor(epsilon=1.1, warm_start=True, max_iter=20, tol=0.1)
        # self.right_lane_fitting_model = linear_model.HuberRegressor(epsilon=1.1, warm_start=True, max_iter=20, tol=0.1)
        self.left_lane_fitting_model = linear_model.HuberRegressor(epsilon=1.3, warm_start=True, max_iter=20, tol=0.1)
        self.right_lane_fitting_model = linear_model.HuberRegressor(epsilon=1.3, warm_start=True, max_iter=20, tol=0.1)
        
        
    def remove_outlier(self, detected_y, detected_x, flag):
        diff_y = np.diff(detected_y)
        diff_x = np.diff(detected_x)
        dx_over_dy = diff_x / diff_y
        if flag == 'LEFT':
            maybe_outlier = ~( (dx_over_dy<=0) & (dx_over_dy>=-3) )
        else:
            maybe_outlier = ~( (dx_over_dy<=3) & (dx_over_dy>=0) )
        maybe_outlier_inds = np.where(maybe_outlier)[0]
        if len(maybe_outlier_inds) <= 1:
            return detected_y, detected_x
        else:
            next_inds = list(range(len(detected_y)))
            next_inds.remove(maybe_outlier_inds[0]+1)
            return self.remove_outlier(detected_y[next_inds], detected_x[next_inds], flag)
        
        
    def bounded_search(self, out_of_one_lane, row_anchor_original_image, flag='LEFT'):
        out_of_one_lane = out_of_one_lane * self.col_sample_w * self.input_shape[1] / 800
        detected_inds = out_of_one_lane > 0
        detected_x = out_of_one_lane[detected_inds]
        detected_y = row_anchor_original_image[::-1][detected_inds]
        
        detected_xy_CP = np.array(np.vstack( (detected_x,detected_y) ).T)
        detected_xy_WC = self.cam.project_from_CP_to_WC( detected_xy_CP )
        
        if flag == 'LEFT':
            if self.left_f is not None:
                left_f_wc = np.poly1d(self.change_cp_f_to_wc_f(self.left_f))
                x_mid = left_f_wc(detected_xy_WC[:,1] - self.z_for_keep_distance)
                x_search_left = x_mid - self.Margin_for_search_around_poly*(1+self.travel_distance_for_not_detect_left_lane/2)
                x_search_right = x_mid + self.Margin_for_search_around_poly*(1+self.travel_distance_for_not_detect_left_lane/2)
            else:
                x_search_left = -4
                x_search_right = -1
        
        else:
            if self.right_f is not None:
                right_f_wc = np.poly1d(self.change_cp_f_to_wc_f(self.right_f))
                x_mid = right_f_wc(detected_xy_WC[:,1] - self.z_for_keep_distance)
                x_search_left = x_mid - self.Margin_for_search_around_poly*(1+self.travel_distance_for_not_detect_right_lane/2)
                x_search_right = x_mid + self.Margin_for_search_around_poly*(1+self.travel_distance_for_not_detect_right_lane/2)
            else:
                x_search_left = 1
                x_search_right = 4
        
        selected_inds_front = (detected_xy_WC[:,1] > 2) & (detected_xy_WC[:,1] < 30)  # only use front distance from 2 to 30
        # print('in front selected nums %d'%selected_inds_front.sum(), 'all %d'%len(row_anchor_original_image))
        selected_inds = ( detected_xy_WC[:,0] > x_search_left ) & (detected_xy_WC[:,0] < x_search_right) & selected_inds_front
        if np.sum( selected_inds ) > 0:
            selected_y = detected_y[selected_inds]
            selected_x = detected_x[selected_inds]
            
            selected_vs_front_ratio = np.clip(np.sum(selected_inds)/ 40, 0, 1)
            # detected_vs_all_ratio = np.clip(np.sum(selected_inds_front) / 40, 0, 1)    
            # selected_vs_front_ratio = np.sum(selected_inds)/ np.sum(selected_inds_front)
            detected_vs_all_ratio = np.sum(detected_inds) / len(row_anchor_original_image)
            weight_of_select = 0.6
            detection_quality = weight_of_select*selected_vs_front_ratio + (1-weight_of_select)*detected_vs_all_ratio
            return True, selected_y, selected_x, detection_quality
        else:
            return False, [], [], 0
        
                            
    def update_horizon(self, lefty, leftx, righty, rightx):
        left_inds = (lefty >= self.input_shape[0]*0.75)
        right_inds = (righty >= self.input_shape[0]*0.75)
        lefty = lefty[left_inds]
        leftx = leftx[left_inds]
        righty = righty[right_inds]
        rightx = rightx[right_inds]
        if len(lefty) > self.Min_num_for_update_horizon and len(righty)> self.Min_num_for_update_horizon:
            left_ransac = linear_model.RANSACRegressor()
            left_ransac.fit(lefty.reshape(-1,1), leftx)
            left_coef_ = left_ransac.estimator_.coef_
            left_intercept_ = left_ransac.estimator_.intercept_
            
            right_ransac = linear_model.RANSACRegressor()
            right_ransac.fit(righty.reshape(-1,1), rightx)
            right_coef_ = right_ransac.estimator_.coef_
            right_intercept_ = right_ransac.estimator_.intercept_
            
            horizon_new = -(right_intercept_ - left_intercept_) / (right_coef_ - left_coef_)
            #print('Horizon New is : %.2f'%horizon_new)
            mean_horizon = self.Horizon_pos.get_refined_result()[0,0]
            #print('Horizon Mean is : %.2f'%mean_horizon)
            
            update_horizon_min = mean_horizon - self.Max_update_step_for_horizon
            update_horizon_max = mean_horizon + self.Max_update_step_for_horizon
            self.Horizon_pos.update( np.matrix(horizon_new, dtype=np.float32).clip(update_horizon_min, update_horizon_max) )
        return 0
    
    
    def update_state_with_motion(self, phi_imu, delta_heading, delta_x, delta_z):
        #add measurament left distance: self.meas_left_distance
        horizon_refine = self.cam.update_camera_phi(phi_imu, self.Horizon_pos.get_refined_result()[0,0])
        
        self.cam.update_camera_state(delta_heading, delta_x, delta_z)
        self.Horizon_pos.kf.statePost = np.matrix(horizon_refine, dtype=np.float32)
        
        if len(self.Memo_Points_in_WC_left) > 0:
            self.Points_from_motion_left = self.cam.project_from_WC_to_CP(self.Memo_Points_in_WC_left)
        else:
            self.Points_from_motion_left = np.array([[],[]]).T
            
        if len(self.Memo_Points_in_WC_right) > 0:
            self.Points_from_motion_right = self.cam.project_from_WC_to_CP(self.Memo_Points_in_WC_right)
        else:
            self.Points_from_motion_right = np.array([[],[]]).T
            
        if len(self.Points_from_motion_left) > 0:
            lefty_wc = self.Memo_Points_in_WC_left[:,1]
            left_inds = (lefty_wc<25) & (lefty_wc>2)
        else:
            left_inds = []
        self.Memo_Points_in_CP_left = self.Points_from_motion_left[left_inds]
        self.Memo_Points_in_WC_left = self.cam.project_from_CP_to_WC( self.Memo_Points_in_CP_left )
        self.Weight_left = self.Weight_left[left_inds]
        
        if len(self.Points_from_motion_right) > 0:
            righty_wc = self.Memo_Points_in_WC_right[:,1]
            right_inds = (righty_wc<25) & (righty_wc>2)
        else:
            right_inds = []
        self.Memo_Points_in_CP_right = self.Points_from_motion_right[right_inds]
        self.Memo_Points_in_WC_right = self.cam.project_from_CP_to_WC( self.Memo_Points_in_CP_right )
        self.Weight_right = self.Weight_right[right_inds]
            
                    
    def gaussian(self, x, mu, sig):
        gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        ret = 1 - 0.95*gauss
        return ret


    def update_hyperbola(self, h, tmp_lefty, tmp_leftx, tmp_righty, tmp_rightx, distanceAndhead):
        scale = 1
        
        if (len(tmp_lefty) + len(tmp_righty)) >= self.Min_num_for_store_points:
            if ( len(self.Weight_left) == 0 ):
                lefty = tmp_lefty.copy()
                leftx = tmp_leftx.copy()
                self.Weight_left = np.ones( (len(lefty)) )
                self.Memo_Points_in_CP_left = np.array(np.vstack( (leftx,lefty) ).T)
            else:
                memo_points_leftx = self.Memo_Points_in_CP_left.copy()[:,0]
                memo_points_lefty = self.Memo_Points_in_CP_left.copy()[:,1]
                memo_weight_left = self.Weight_left - self.travel_distance_for_not_detect_whole_lanes / 8
                
                gaussian_weight = np.ones( (len(memo_weight_left)) )
                for tmpy in tmp_lefty:
                    gaussian_weight *= self.gaussian(memo_points_lefty, tmpy, self.Stride_between_each_line)
                
                combined_weight = memo_weight_left * gaussian_weight
                keep_points_inds = (combined_weight>0.01)
                
                lefty = np.concatenate( (memo_points_lefty[keep_points_inds], tmp_lefty) )
                leftx = np.concatenate( (memo_points_leftx[keep_points_inds], tmp_leftx) )
                self.Weight_left = np.concatenate( (combined_weight[keep_points_inds], np.ones(len(tmp_lefty))) )
                self.Memo_Points_in_CP_left = np.array(np.vstack( (leftx,lefty) ).T)
                
            self.Memo_Points_in_WC_left = self.cam.project_from_CP_to_WC( self.Memo_Points_in_CP_left )
            
            
            if ( len(self.Weight_right) == 0 ):
                righty = tmp_righty.copy()
                rightx = tmp_rightx.copy()
                self.Weight_right = np.ones( (len(righty)) )
                self.Memo_Points_in_CP_right = np.array(np.vstack( (rightx,righty) ).T)
            else:
                memo_points_rightx = self.Memo_Points_in_CP_right.copy()[:,0]
                memo_points_righty = self.Memo_Points_in_CP_right.copy()[:,1]
                memo_weight_right = self.Weight_right - self.travel_distance_for_not_detect_whole_lanes / 8
                
                gaussian_weight = np.ones( (len(memo_weight_right)) )
                for tmpy in tmp_righty:
                    gaussian_weight *= self.gaussian(memo_points_righty, tmpy, self.Stride_between_each_line)
                
                combined_weight = memo_weight_right * gaussian_weight
                keep_points_inds = (combined_weight>0.01)
                
                righty = np.concatenate( (memo_points_righty[keep_points_inds], tmp_righty) )
                rightx = np.concatenate( (memo_points_rightx[keep_points_inds], tmp_rightx) )
                self.Weight_right = np.concatenate( (combined_weight[keep_points_inds], np.ones(len(tmp_righty))) )
                self.Memo_Points_in_CP_right = np.array(np.vstack( (rightx,righty) ).T)
                
            self.Memo_Points_in_WC_right = self.cam.project_from_CP_to_WC( self.Memo_Points_in_CP_right )
            
            self.travel_distance_for_not_detect_whole_lanes = 0
        
        ## TODO: replace them with the RANSAC based polyfit
        detect_error_rate = (  self.Number_of_scanline**2/(2* (1e-4+ (sum(self.Weight_left))**2 + (sum(self.Weight_right))**2)  )  )**0.5
        if detect_error_rate <= self.Max_detect_error_rate:
            scale = min( (1, 0.8+0.4/(1e-5+self.travel_distance_for_not_detect_whole_lanes) ) )
            scale = scale * detect_error_rate#TODO
        
        self.filter_WC_and_PC_based_on_lane_width(scale, distanceAndhead)
       
        
    def change_cp_f_to_wc_f(self, f_cp):
        if f_cp is not None:
            ploty = np.linspace(self.scanline_y_end+5, self.scanline_y_start-30, 15)
            fitx = f_cp(ploty)
            CP = np.array(np.vstack( (fitx,ploty) ).T)
            WC = self.cam.project_from_CP_to_WC(CP)
            lane_fit = np.polyfit(WC[:,1]-self.z_for_keep_distance, WC[:,0], 2)
            return lane_fit
        else:
            return None
        
        
    def get_filtered_CP_points(self, flag='LEFT'):
        cur_direction = self.Lane_cur_direction.get_refined_result()
        if cur_direction is not None:
            cur, direction = cur_direction[:,0]
            if flag=='LEFT':
                lane_fit = np.array( [cur, direction, self.Left_lane_pos.get_refined_result()[0,0] ] )
            else:
                lane_fit = np.array( [cur, direction, self.Right_lane_pos.get_refined_result()[0,0] ] )
            lane_f_refine = np.poly1d(lane_fit)
            
            WC_z = np.linspace(4, 24, 20)
            WC_x = lane_f_refine(WC_z - self.z_for_keep_distance)
            WC = np.vstack( (WC_x,WC_z) ).T
            ret_CP = self.cam.project_from_WC_to_CP_still(WC)
            ret_CP_f = np.poly1d( np.polyfit(ret_CP[:,1], ret_CP[:,0], 2) )
            return ret_CP, ret_CP_f,lane_fit
        
        return np.array([[],[]]).T, None, None
    

    def resample_wc_points(self, y_wc):
        sample_start_y = max(-3, min(y_wc))
        sample_end_y = min(7, max(y_wc))
        sample_num = max(1, min(16, int(np.sum(y_wc<7)/1.5)))
        
        sample_y_pos = np.linspace( sample_start_y, sample_end_y, sample_num )
        dis_matrix = np.abs(sample_y_pos[:,None] - y_wc[None])
        resampled_inds = dis_matrix.argmin(0)
        resampled_inds = np.append(resampled_inds, np.where(y_wc>7))
        return resampled_inds
    
    def fit_poly_ransac(self, lane='LEFT'):
        x_wc = []
        y_wc = []
        if lane=='LEFT':
            if len(self.Memo_Points_in_WC_left) > 0:
                x_wc = self.Memo_Points_in_WC_left[:,0] 
                y_wc = self.Memo_Points_in_WC_left[:,1] - self.z_for_keep_distance
                weight_arr = self.Weight_left
        else:
            if len(self.Memo_Points_in_WC_right) > 0:
                x_wc = self.Memo_Points_in_WC_right[:,0]
                y_wc = self.Memo_Points_in_WC_right[:,1] - self.z_for_keep_distance
                weight_arr = self.Weight_right
        if len(x_wc) > 0:
            # print('x_wc\n', x_wc)
            # print('y_wc\n', y_wc)
            resampled_inds = self.resample_wc_points(y_wc)
            y_wc = y_wc[resampled_inds]
            x_wc = x_wc[resampled_inds]
            weight_arr = weight_arr[resampled_inds]
            
            X_in = np.array(np.vstack( (y_wc**2,y_wc) ).T)
            y_in = x_wc 
            # lane_ransac = linear_model.RANSACRegressor()
            if lane == 'LEFT':
                lane_ransac = self.left_lane_fitting_model
            else:
                lane_ransac = self.right_lane_fitting_model
            #start_time = time.time()
            lane_ransac.fit(X_in, y_in, weight_arr)
            #print('one time fitting time: %.4f ms'%( (time.time()-start_time)*1e3) )
            # lane_wc_fit = np.array( (lane_ransac.estimator_.coef_[0], lane_ransac.estimator_.coef_[1], lane_ransac.estimator_.intercept_))
            new_cur = lane_ransac.coef_[0]
            new_direction = lane_ransac.coef_[1]
            
            # Judge the confidence of fitting when the curve or direction is large
            cur_dir_pos_score = np.array([1, 1, 1])
            curve_quality = min(1, (1e-2/(np.abs(new_cur)+1e-9))**2)
            dir_quality = min(1, (0.14/(np.abs(new_direction)+1e-9))**2)
            cur_dir_pos_score = cur_dir_pos_score * curve_quality * dir_quality
            old_cur_direction = self.Lane_cur_direction.get_refined_result()
            if old_cur_direction is not None:
                old_cur, old_direction = old_cur_direction[:,0]
                new_cur = np.clip(new_cur, old_cur-self.Max_update_step_for_cur, old_cur+self.Max_update_step_for_cur)
                new_direction = np.clip(new_direction, old_direction-self.Max_update_step_for_direction, old_direction+self.Max_update_step_for_direction)
            # still can't go out of this region
            new_cur = np.clip(new_cur, -5*self.Max_update_step_for_cur, 5*self.Max_update_step_for_cur)
            new_direction = np.clip(new_direction, -5*self.Max_update_step_for_direction, 5*self.Max_update_step_for_direction)
            lane_wc_fit = np.array( (new_cur, new_direction, lane_ransac.intercept_))
            cur_fit_score = lane_ransac.score(X_in, y_in)
            if np.isnan(cur_fit_score):
                cur_fit_score = 1e-6
            lane_fit_score = max( cur_fit_score, 1e-6) * cur_dir_pos_score
            return lane_wc_fit, lane_fit_score
        else:
            return None, 1e-6

    def filter_WC_and_PC_based_on_lane_width(self, scale, distanceAndhead):
        refine_cur_direction = None
        #start_time = time.time()
        left_fit_wc, left_score = self.fit_poly_ransac('LEFT')
        right_fit_wc, right_score = self.fit_poly_ransac('RIGHT')
        #print('lane width based fitting time: %.4f ms'%( (time.time()-start_time)*1e3) )
        # print('LEFT  :    Curve: %.2f,   Angle: %.2f,    Dis: %.2f,    Score: %.2f '%( 1/(1e-9+left_fit_wc[0]), np.rad2deg(left_fit_wc[1]), left_fit_wc[2], left_score[0]) )
        # print('RIGHT :    Curve: %.2f,   Angle: %.2f,    Dis: %.2f,    Score: %.2f '%( 1/(1e-9+right_fit_wc[0]), np.rad2deg(right_fit_wc[1]), right_fit_wc[2], right_score[0]) )
        
        #TODO:propose a method to judge the confidence of polyfit from lane detection
        pos_precess_noise = self.Left_lane_pos.proc_noise_scale_arr[0] * max(0.1, distanceAndhead[0] / 0.5)
        if left_fit_wc is not None:
            left_fit_wc_std = self.Lane_detection_noise_scale_base * 1/left_score
#            print('left_fit_wc_std:', left_fit_wc_std, '     left_fit_gps_std:',left_fit_gps_std)
            left_fit_refine = left_fit_wc.copy()
            left_fit_refine_std = left_fit_wc_std.copy()
            old_pos_left = self.Left_lane_pos.get_refined_result()[0,0]
            update_lane_pos = np.clip(left_fit_refine[2], old_pos_left-self.Max_update_step_for_lane_pos, old_pos_left+self.Max_update_step_for_lane_pos)
            self.Left_lane_pos.update( np.matrix([update_lane_pos], dtype=np.float32), proc_noise_arr=[pos_precess_noise], meas_noise_arr=[left_fit_refine_std[2]] )
            refine_cur_direction = left_fit_refine[:2] 
            refine_cur_direction_std = left_fit_refine_std[:2]
            #print('left fit meas_std: %.4f, fixed: %.4f'%(left_fit_refine_std[2], self.Left_lane_pos.meas_noise_scale_arr[0]) )
            
            
        if right_fit_wc is not None:
            right_fit_wc_std = self.Lane_detection_noise_scale_base * 1/right_score
#            print('right_fit_wc_std:', right_fit_wc_std, '     right_fit_gps_std:',right_fit_gps_std)
            right_fit_refine = right_fit_wc.copy()
            right_fit_refine_std = right_fit_wc_std.copy()
            old_pos_right = self.Right_lane_pos.get_refined_result()[0,0]
            update_lane_pos = np.clip(right_fit_refine[2], old_pos_right-self.Max_update_step_for_lane_pos, old_pos_right+self.Max_update_step_for_lane_pos)
            self.Right_lane_pos.update( np.matrix([update_lane_pos], dtype=np.float32), proc_noise_arr=[pos_precess_noise], meas_noise_arr=[right_fit_refine_std[2]] )
            refine_cur_direction = right_fit_refine[:2] 
            refine_cur_direction_std = right_fit_refine_std[:2]
        
        if (right_fit_wc is not None) and (left_fit_wc is not None):
            refine_cur_direction = ( left_fit_refine[:2] * right_fit_refine_std[:2] + right_fit_refine[:2] * left_fit_refine_std[:2] ) / (left_fit_refine_std[:2] + right_fit_refine_std[:2])
            refine_cur_direction_std = 1 / (1/left_fit_refine_std[:2] + 1/right_fit_refine_std[:2])
    #        print('refine_cur_direction_std:', refine_cur_direction_std)
        if refine_cur_direction is not None:
            self.Lane_cur_direction.update(np.matrix(refine_cur_direction, dtype=np.float32).T, meas_noise_arr=refine_cur_direction_std)          
            

        filtered_left_CP, filtered_left_CP_f, refined_wc_left = self.get_filtered_CP_points(flag='LEFT')
        filtered_right_CP, filtered_right_CP_f, refined_wc_right = self.get_filtered_CP_points(flag='RIGHT')
        
        #self.left_fit_wc = refined_wc_left.copy()
        #self.right_fit_wc = refined_wc_right.copy()
        self.left_fit_wc = refined_wc_left
        self.right_fit_wc = refined_wc_right
        
        self.Points_in_CP_left_for_draw = filtered_left_CP
        self.Points_in_CP_right_for_draw = filtered_right_CP
        
        self.left_f = filtered_left_CP_f
        self.right_f = filtered_right_CP_f
            


    def find_lane_and_fit(self, out_j, row_anchor_original_image, phi_imu, delta_heading, delta_x, delta_z):
        distance = ( delta_x**2 + delta_z**2 )**0.5
        self.travel_distance_for_not_detect_whole_lanes += distance
        
        if np.sum(out_j[:, 2] != 0) > 2: #2 is the right lane channel
            right_flag, righty, rightx, selected_right_percent = self.bounded_search(out_j[:, 2], row_anchor_original_image, 'RIGHT')    
            #print('right percent: %.3f'%selected_right_percent)
            if right_flag:
                self.bounded_right_CP = np.array(np.vstack( (rightx,righty) ).T)
            if selected_right_percent < 0.6:
                self.travel_distance_for_not_detect_right_lane += distance
            elif selected_right_percent > 0.8:
                self.travel_distance_for_not_detect_right_lane -= distance
        else:
            self.travel_distance_for_not_detect_right_lane += distance
            right_flag, righty, rightx, selected_right_percent = False, [], [], 0
        self.travel_distance_for_not_detect_right_lane = np.clip(self.travel_distance_for_not_detect_right_lane, 0, 12)   
        
        
        if np.sum(out_j[:, 1] != 0) > 2: #1 is the left lane channel
            left_flag, lefty, leftx, selected_left_percent = self.bounded_search(out_j[:, 1], row_anchor_original_image, 'LEFT')
            self.selected_left_percent = selected_left_percent
            if left_flag:
                self.bounded_left_CP = np.array(np.vstack( (leftx,lefty) ).T)
            if selected_left_percent < 0.6:
                self.travel_distance_for_not_detect_left_lane += distance
            elif selected_left_percent > 0.8:
                self.travel_distance_for_not_detect_left_lane -= distance
                
            if selected_left_percent < 0.6 and selected_right_percent < 0.6:
                self.continuous_no_detect_distance += distance
            else:
                self.continuous_no_detect_distance = 0
            
            if selected_left_percent > 0.8:
                self.recent_detect_list.append(1)
            else:
                self.recent_detect_list.append(0)
                
        else:
            self.continuous_no_detect_distance += distance
            self.travel_distance_for_not_detect_left_lane += distance
            left_flag, lefty, leftx = False, [], []
            self.recent_detect_list.append(0)
        
        self.recent_detect_list.pop(0)
        if sum(self.recent_detect_list) == self.init_wait_length:
            self.has_initialized = True
        
        if self.continuous_no_detect_distance > self.max_no_detection_distance:
            self.recent_detect_list = [0] * self.init_wait_length
            self.has_initialized = False
            
        if self.travel_distance_for_not_detect_left_lane > 12:
            self.travel_distance_for_not_detect_left_lane_count += 1
        else:
            self.travel_distance_for_not_detect_left_lane_count = 0
        self.travel_distance_for_not_detect_left_lane = np.clip(self.travel_distance_for_not_detect_left_lane, 0, 12)
            
        # print('distance: %.4f'%distance)
        print('no left lane distance: %.3f and count: %d'%(self.travel_distance_for_not_detect_left_lane, self.travel_distance_for_not_detect_left_lane_count))
        if self.travel_distance_for_not_detect_left_lane_count > self.Reset_no_detect_count_threshold:
            self.reset()
            print('Reset')
        else:
            detect_any_flag = left_flag | right_flag
            detect_all_flag = left_flag & right_flag
            
            if detect_all_flag:
                self.update_horizon(lefty, leftx, righty, rightx)
        
            #start_time = time.time()
            self.update_state_with_motion(phi_imu, delta_heading, delta_x, delta_z)
            #print('motion time: %.4f ms'%( (time.time()-start_time)*1e3) )
            if detect_any_flag and (not detect_all_flag):
                if not self.Hyperbola_para.first_detected:
                    if left_flag:
                        ## add right_point manually
                        righty = np.array([self.input_shape[0]])
                        rightx = np.array([self.input_shape[1]*0.8])
                    else:
                        ## add left point manually
                        lefty = np.array([self.input_shape[0]])
                        leftx = np.array([self.input_shape[1]*0.2])
                        
            h = self.Horizon_pos.get_refined_result()[0,0]
            # start_time = time.time()
            self.update_hyperbola(h, lefty, leftx, righty, rightx, [distance,delta_heading])
            #print('fitting time: %.4f ms'%( (time.time()-start_time)*1e3) )
        
        fit_para = None
        if self.Hyperbola_para.first_detected:      
            fit_para = self.Hyperbola_para.get_refined_result()
            
        return fit_para
    
    # def find_lane_and_fit(self, out_j, row_anchor_original_image, phi_imu, delta_heading, delta_x, delta_z):
    #     distance = ( delta_x**2 + delta_z**2 )**0.5
    #     self.travel_distance_for_not_detect_whole_lanes += distance
        
    #     if np.sum(out_j[:, 1] != 0) > 2: #1 is the left lane channel
    #         left_flag, lefty, leftx, selected_left_percent = self.bounded_search(out_j[:, 1], row_anchor_original_image, 'LEFT')
    #         #print('left percent: %.3f'%selected_left_percent)
    #         if left_flag:
    #             self.bounded_left_CP = np.array(np.vstack( (leftx,lefty) ).T)
    #         if selected_left_percent < 0.6:
    #             self.travel_distance_for_not_detect_left_lane += distance
    #         elif selected_left_percent > 0.8:
    #             self.travel_distance_for_not_detect_left_lane -= distance
            
    #         if selected_left_percent < 0.6:
    #             self.continuous_no_detect_distance += distance
    #         else:
    #             self.continuous_no_detect_distance = 0
            
    #         if selected_left_percent > 0.8:
    #             self.recent_detect_list.append(1)
    #         else:
    #             self.recent_detect_list.append(0)
                
    #     else:
    #         self.continuous_no_detect_distance += distance
    #         self.travel_distance_for_not_detect_left_lane += distance
    #         left_flag, lefty, leftx = False, [], []
    #         self.recent_detect_list.append(0)
        
    #     self.recent_detect_list.pop(0)
    #     if sum(self.recent_detect_list) == self.init_wait_length:
    #         self.has_initialized = True
            
    #     if self.travel_distance_for_not_detect_left_lane > 12:
    #         self.travel_distance_for_not_detect_left_lane_count += 1
    #     else:
    #         self.travel_distance_for_not_detect_left_lane_count = 0
    #     self.travel_distance_for_not_detect_left_lane = np.clip(self.travel_distance_for_not_detect_left_lane, 0, 12)
            
        
    #     if np.sum(out_j[:, 2] != 0) > 2: #2 is the right lane channel
    #         right_flag, righty, rightx, selected_right_percent = self.bounded_search(out_j[:, 2], row_anchor_original_image, 'RIGHT')    
    #         #print('right percent: %.3f'%selected_right_percent)
    #         if right_flag:
    #             self.bounded_right_CP = np.array(np.vstack( (rightx,righty) ).T)
    #         if selected_right_percent < 0.6:
    #             self.travel_distance_for_not_detect_right_lane += distance
    #         elif selected_right_percent > 0.8:
    #             self.travel_distance_for_not_detect_right_lane -= distance
    #     else:
    #         self.travel_distance_for_not_detect_right_lane += distance
    #         right_flag, righty, rightx = False, [], []
    #     self.travel_distance_for_not_detect_right_lane = np.clip(self.travel_distance_for_not_detect_right_lane, 0, 12)   
        
    #     print('no left lane distance: %.3f and count: %d'%(self.travel_distance_for_not_detect_left_lane, self.travel_distance_for_not_detect_left_lane_count))
    #     if self.travel_distance_for_not_detect_left_lane_count > self.Reset_no_detect_count_threshold:
    #         self.reset()
    #         print('Reset')
    #     else:
    #         detect_any_flag = left_flag | right_flag
    #         detect_all_flag = left_flag & right_flag
            
    #         if detect_all_flag:
    #             self.update_horizon(lefty, leftx, righty, rightx)
        
    #         #start_time = time.time()
    #         self.update_state_with_motion(phi_imu, delta_heading, delta_x, delta_z)
    #         #print('motion time: %.4f ms'%( (time.time()-start_time)*1e3) )
    #         if detect_any_flag and (not detect_all_flag):
    #             if not self.Hyperbola_para.first_detected:
    #                 if left_flag:
    #                     ## add right_point manually
    #                     righty = np.array([self.input_shape[0]])
    #                     rightx = np.array([self.input_shape[1]*0.8])
    #                 else:
    #                     ## add left point manually
    #                     lefty = np.array([self.input_shape[0]])
    #                     leftx = np.array([self.input_shape[1]*0.2])
                        
    #         h = self.Horizon_pos.get_refined_result()[0,0]
    #         # start_time = time.time()
    #         self.update_hyperbola(h, lefty, leftx, righty, rightx, [distance,delta_heading])
    #         #print('fitting time: %.4f ms'%( (time.time()-start_time)*1e3) )
        
    #     fit_para = None
    #     if self.Hyperbola_para.first_detected:      
    #         fit_para = self.Hyperbola_para.get_refined_result()
            
    #     return fit_para
    
    
    def change_para_to_function(self, hyperbola_para):
        if hyperbola_para is None:
            return None, None
        k, b_l, b_r, c = hyperbola_para[:,0]
        h = self.Horizon_pos.get_refined_result()[0,0]
        def f_left(y):
            return k/(y-h) + b_l*(y-h) + c
        def f_right(y):
            return k/(y-h) + b_r*(y-h) + c
        return f_left, f_right
    
    
    def update(self,out_j, delta_distance, delta_heading, phi_imu):
        delta_x = delta_distance * np.sin( np.radians(delta_heading/2) )
        delta_z = delta_distance * np.cos( np.radians(delta_heading/2) )

        self.find_lane_and_fit(out_j, self.row_anchor_in_original_image, 
                                                phi_imu, delta_heading, delta_x, delta_z)
        