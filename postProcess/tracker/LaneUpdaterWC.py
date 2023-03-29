#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:51:41 2021

@author: minghao

LaneUpdaterWC: in this version, I will try to fit the left and right lanes on the world coordinate.
And there is no another Kalman Filter to limit the change of hyperbola
"""


import numpy as np
from CameraConfig import Camera, imu_motion_to_cam_motion
from KalmanFilter import Kalman
from sklearn import linear_model
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

# Filter out ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Lane_updater(object):
    def __init__(self, shape, init_wait_length=4, max_no_detection_distance=5):
        self.input_shape = shape  # h,w

        # parameters for scanline search and blob detection to find candidate points
        # self.scanline_y_start = self.input_shape[0]*0.8 #CHANGE
        # self.scanline_y_end = self.input_shape[0]*0.45 #CHANGE

        self.Margin_for_search_around_poly = 0.3  # 1 meter
        self.Reset_no_detect_count_threshold = 3
        self.init_wait_length = init_wait_length
        self.max_no_detection_distance = max_no_detection_distance
        self.reset()

    def reset(self):
        # camera class, used to apply perspective transform based on movement
        self.cam = Camera(self.input_shape)
        self.horizon_init_pos = np.matrix(
            self.input_shape[0] * 0.34, dtype=np.float32)  # CHANGE
        # maybe there is no need for this filter
        self.Horizon_pos = Kalman(1, [1], [1])
        self.Horizon_pos.update(self.horizon_init_pos)

        # WC:stored lane points in world coordiantes to car; column 0:x,lateral distance; column 1:z,longitudal distance
        # CP:stored lane points in Camera Plane; column 0:x, pixel number from left to right; column 1:y, pixel number from top to down
        self.Memo_Points_in_WC_right = np.array([[], []]).T
        self.Memo_Points_in_WC_left = np.array([[], []]).T
        self.Memo_Points_in_CP_left = np.array([[], []]).T
        self.Memo_Points_in_CP_right = np.array([[], []]).T

        self.Points_in_CP_left_for_draw = np.array([[], []]).T
        self.Points_in_CP_right_for_draw = np.array([[], []]).T

        # weights used when fitting hyperbola
        self.Weight_left = np.array([])
        self.Weight_right = np.array([])

        # from motion: old lane points in world coordiantes to the car transformed using car's movement
        self.Points_from_motion_left = np.array([[], []]).T
        self.Points_from_motion_right = np.array([[], []]).T
        self.Points_from_motion_left_draw = np.array([[], []]).T
        self.Points_from_motion_right_draw = np.array([[], []]).T

        # f:hyperbola function, used when reduce the searching region, and update points for draw
        self.left_f = None
        self.right_f = None
        self.left_fit_wc = None
        self.right_fit_wc = None

        # some parameters for rejecting unconfident detection
        self.Min_num_for_update_horizon = 16
        self.Min_wc_distance_for_update_hyperbola = 3
        self.Min_num_for_store_points = 14

        # some parameters for limiting outlier's influence
        self.Max_update_step_for_horizon = 50
        self.Max_update_step_for_lane_pos = 0.5
        self.Max_update_step_for_cur = 1/300
        self.Max_update_step_for_direction = np.deg2rad(1.2)

        self.Left_lane_pos = Kalman(1, [0.002], [0.04])
        self.Left_lane_pos.update(np.matrix(-2, dtype=np.float32))
        self.Right_lane_pos = Kalman(1, [0.003], [0.04])
        self.Right_lane_pos.update(np.matrix(2, dtype=np.float32))
        self.Lane_cur_direction = Kalman(2, [0.01, 0.003], [0.3, 0.3])

        # when fitting parabola, minus x with this constant can help get a stable lateral distance
        self.z_for_keep_distance = 2

        # decide how confident to trust this detection
        self.Lane_detection_noise_scale_base = np.array([0.2, 0.2, 0.1]) * 0.15

        self.travel_distance_for_not_detect_left_lane = 10
        self.travel_distance_for_not_detect_left_lane_count = 0
        self.travel_distance_for_not_detect_right_lane = 10

        self.has_initialized = False
        self.recent_detect_list = [0] * self.init_wait_length
        self.continuous_no_detect_distance = 0
        self.left_confidence = 0

        self.bounded_left_CP = np.array([[], []]).T
        self.bounded_right_CP = np.array([[], []]).T

        # self.left_lane_fitting_model = linear_model.HuberRegressor(epsilon=1.1, warm_start=True, max_iter=20, tol=0.1)
        # self.right_lane_fitting_model = linear_model.HuberRegressor(epsilon=1.1, warm_start=True, max_iter=20, tol=0.1)
        self.left_lane_fitting_model = linear_model.HuberRegressor(
            epsilon=1.35, warm_start=True, max_iter=40, tol=0.1)
        self.right_lane_fitting_model = linear_model.HuberRegressor(
            epsilon=1.35, warm_start=True, max_iter=40, tol=0.1)

        # RANSAC model for calibrating vanishing point
        self.left_ransac = linear_model.RANSACRegressor()
        self.right_ransac = linear_model.RANSACRegressor()

    def bounded_search(self, detected_col_row_confidence, flag='LEFT'):
        # detected points in camera plane
        detected_xy_CP = detected_col_row_confidence[:, :2]
        detected_xz_WC = self.cam.project_from_CP_to_WC(
            detected_xy_CP)  # detected points in world coordinates
        # x:right, y:down, z:forward (in world coordinates)

        if flag == 'LEFT':
            if self.left_fit_wc is not None:
                left_f_wc = np.poly1d(self.left_fit_wc)
                x_mid = left_f_wc(
                    detected_xz_WC[:, 1] - self.z_for_keep_distance)
                uncertainty_from_distance = np.clip(
                    detected_xz_WC[:, 1]/10, 1, 3)  # furthur away, less confident
                x_search_left = x_mid - self.Margin_for_search_around_poly * \
                    (1+self.travel_distance_for_not_detect_left_lane/3) * \
                    uncertainty_from_distance
                x_search_right = x_mid + self.Margin_for_search_around_poly * \
                    (1+self.travel_distance_for_not_detect_left_lane/3) * \
                    uncertainty_from_distance
            else:
                x_search_left = -4
                x_search_right = -1

        else:
            if self.right_fit_wc is not None:
                right_f_wc = np.poly1d(self.right_fit_wc)
                x_mid = right_f_wc(
                    detected_xz_WC[:, 1] - self.z_for_keep_distance)
                uncertainty_from_distance = np.clip(
                    detected_xz_WC[:, 1]/10, 1, 3)
                x_search_left = x_mid - self.Margin_for_search_around_poly * \
                    (1+self.travel_distance_for_not_detect_right_lane/3) * \
                    uncertainty_from_distance
                x_search_right = x_mid + self.Margin_for_search_around_poly * \
                    (1+self.travel_distance_for_not_detect_right_lane/3) * \
                    uncertainty_from_distance
            else:
                x_search_left = 1
                x_search_right = 4
            # x_search_left = 1
            # x_search_right = 4

        selected_inds_front = (detected_xz_WC[:, 1] > 2) & (
            detected_xz_WC[:, 1] < 30)  # only use front distance from 2 to 30
        # print('in front selected nums %d'%selected_inds_front.sum(), 'all %d'%len(row_anchor_original_image))
        selected_inds = (detected_xz_WC[:, 0] > x_search_left) & (
            detected_xz_WC[:, 0] < x_search_right) & selected_inds_front
        if np.sum(selected_inds) > 0:
            selected_col_row_confidence = detected_col_row_confidence[selected_inds]
            selected_confidence_ratio = np.clip(
                np.sum(selected_col_row_confidence[:, 2]) / 20, 0, 1)
            return True, selected_col_row_confidence, selected_confidence_ratio
        else:
            return False, np.zeros((0, 3)), 0

    def update_horizon(self, left_col_row_confidence, right_col_row_confidence):
        lefty = left_col_row_confidence[:, 1]
        leftx = left_col_row_confidence[:, 0]
        righty = right_col_row_confidence[:, 1]
        rightx = right_col_row_confidence[:, 0]

        left_inds = (lefty >= self.input_shape[0]*0.75)
        right_inds = (righty >= self.input_shape[0]*0.75)
        lefty = lefty[left_inds]
        leftx = leftx[left_inds]
        righty = righty[right_inds]
        rightx = rightx[right_inds]
        if len(lefty) > self.Min_num_for_update_horizon and len(righty) > self.Min_num_for_update_horizon:

            self.left_ransac.fit(lefty.reshape(-1, 1), leftx)
            left_coef_ = self.left_ransac.estimator_.coef_
            left_intercept_ = self.left_ransac.estimator_.intercept_

            self.right_ransac.fit(righty.reshape(-1, 1), rightx)
            right_coef_ = self.right_ransac.estimator_.coef_
            right_intercept_ = self.right_ransac.estimator_.intercept_

            horizon_new = -(right_intercept_ - left_intercept_) / \
                (right_coef_ - left_coef_)
            #print('Horizon New is : %.2f'%horizon_new)
            mean_horizon = self.Horizon_pos.get_refined_result()[0, 0]
            #print('Horizon Mean is : %.2f'%mean_horizon)

            update_horizon_min = mean_horizon - self.Max_update_step_for_horizon
            update_horizon_max = mean_horizon + self.Max_update_step_for_horizon
            self.Horizon_pos.update(np.matrix(horizon_new, dtype=np.float32).clip(
                update_horizon_min, update_horizon_max))

    def update_state_with_motion(self, pitch_imu, delta_heading, delta_front, delta_right):
        # add measurament left distance: self.meas_left_distance
        horizon_refine = self.cam.update_camera_phi(
            pitch_imu, self.Horizon_pos.get_refined_result()[0, 0])

        self.cam.update_camera_state(delta_heading, delta_front, delta_right)
        self.Horizon_pos.kf.statePost = np.matrix(
            horizon_refine, dtype=np.float32)

        # first project the old points in world coordinate to camera coordinate
        self.Points_from_motion_left = self.cam.project_from_WC_to_CP(
            self.Memo_Points_in_WC_left)

        self.Points_from_motion_right = self.cam.project_from_WC_to_CP(
            self.Memo_Points_in_WC_right)

        # then project the points in camera coordinate to new world coordinate
        if len(self.Points_from_motion_left) > 0:
            leftz_wc = self.Memo_Points_in_WC_left[:, 1]
            left_inds = (leftz_wc < 25) & (leftz_wc > 2)
        else:
            left_inds = []
        self.Memo_Points_in_CP_left = self.Points_from_motion_left[left_inds]
        self.Memo_Points_in_WC_left = self.cam.project_from_CP_to_WC(
            self.Memo_Points_in_CP_left)
        self.Weight_left = self.Weight_left[left_inds]

        if len(self.Points_from_motion_right) > 0:
            rightz_wc = self.Memo_Points_in_WC_right[:, 1]
            right_inds = (rightz_wc < 25) & (rightz_wc > 2)
        else:
            right_inds = []
        self.Memo_Points_in_CP_right = self.Points_from_motion_right[right_inds]
        self.Memo_Points_in_WC_right = self.cam.project_from_CP_to_WC(
            self.Memo_Points_in_CP_right)
        self.Weight_right = self.Weight_right[right_inds]

    def gaussian(self, x, mu, sig):
        gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        ret = 1 - 0.95*gauss
        return ret

    def update_lane_fit(self, left_col_row_confidence, right_col_row_confidence, distanceAndhead):
        if (len(left_col_row_confidence) + len(right_col_row_confidence)) >= self.Min_num_for_store_points:
            if (len(self.Weight_left) == 0):
                self.Weight_left = left_col_row_confidence[:, 2]
                self.Memo_Points_in_CP_left = left_col_row_confidence[:, :2]
            else:
                memo_weight_left = 0.8*self.Weight_left - \
                    distanceAndhead[0] / 5
                keep_points_inds = (memo_weight_left > 0.1)
                self.Weight_left = np.concatenate(
                    (memo_weight_left[keep_points_inds], left_col_row_confidence[:, 2]))
                self.Memo_Points_in_CP_left = np.concatenate(
                    (self.Memo_Points_in_CP_left[keep_points_inds], left_col_row_confidence[:, :2]))

            self.Memo_Points_in_WC_left = self.cam.project_from_CP_to_WC(
                self.Memo_Points_in_CP_left)

            if (len(self.Weight_right) == 0):
                self.Weight_right = right_col_row_confidence[:, 2]
                self.Memo_Points_in_CP_right = right_col_row_confidence[:, :2]
            else:
                memo_weight_right = 0.8*self.Weight_right - \
                    distanceAndhead[0] / 5
                keep_points_inds = (memo_weight_right > 0.1)
                self.Weight_right = np.concatenate(
                    (memo_weight_right[keep_points_inds], right_col_row_confidence[:, 2]))
                self.Memo_Points_in_CP_right = np.concatenate(
                    (self.Memo_Points_in_CP_right[keep_points_inds], right_col_row_confidence[:, :2]))

            self.Memo_Points_in_WC_right = self.cam.project_from_CP_to_WC(
                self.Memo_Points_in_CP_right)

        self.filter_WC_and_PC_based_on_lane_width(distanceAndhead)

    def get_filtered_CP_points(self, flag='LEFT'):
        cur_direction = self.Lane_cur_direction.get_refined_result()
        if cur_direction is not None:
            cur, direction = cur_direction[:, 0]
            if flag == 'LEFT':
                lane_fit = np.array(
                    [cur, direction, self.Left_lane_pos.get_refined_result()[0, 0]])
            else:
                lane_fit = np.array(
                    [cur, direction, self.Right_lane_pos.get_refined_result()[0, 0]])
            lane_f_refine = np.poly1d(lane_fit)

            WC_z = np.linspace(4, 24, 20)
            WC_x = lane_f_refine(WC_z - self.z_for_keep_distance)
            WC = np.vstack((WC_x, WC_z)).T
            ret_CP = self.cam.project_from_WC_to_CP_still(WC)
            return ret_CP, lane_fit
        return np.array([[], []]).T, None

    def resample_wc_points(self, z_wc):
        sample_start_z = max(-3, min(z_wc))
        sample_end_z = min(7, max(z_wc))
        sample_num = max(1, min(16, int(np.sum(z_wc < 7)/1.5)))

        sample_z_pos = np.linspace(sample_start_z, sample_end_z, sample_num)
        dis_matrix = np.abs(sample_z_pos[:, None] - z_wc[None])
        resampled_inds = dis_matrix.argmin(0)
        resampled_inds = np.append(resampled_inds, np.where(z_wc > 7))
        return resampled_inds

    def fit_poly_ransac(self, lane='LEFT'):
        x_wc = []
        z_wc = []
        if lane == 'LEFT':
            if len(self.Memo_Points_in_WC_left) > 0:
                x_wc = self.Memo_Points_in_WC_left[:, 0]
                z_wc = self.Memo_Points_in_WC_left[:,
                                                   1] - self.z_for_keep_distance
                weight_arr = self.Weight_left
        else:
            if len(self.Memo_Points_in_WC_right) > 0:
                x_wc = self.Memo_Points_in_WC_right[:, 0]
                z_wc = self.Memo_Points_in_WC_right[:,
                                                    1] - self.z_for_keep_distance
                weight_arr = self.Weight_right
        if len(x_wc) > 0:
            resampled_inds = self.resample_wc_points(z_wc)
            z_wc = z_wc[resampled_inds]
            x_wc = x_wc[resampled_inds]
            weight_arr = weight_arr[resampled_inds]

            X_in = np.array(np.vstack((z_wc**2, z_wc)).T)
            y_in = x_wc
            # lane_ransac = linear_model.RANSACRegressor()
            if lane == 'LEFT':
                lane_ransac = self.left_lane_fitting_model
            else:
                lane_ransac = self.right_lane_fitting_model
            #start_time = time.time()
            try:
                lane_ransac.fit(X_in, y_in)
            except Exception as e:
                print("An error occurred during fitting " +
                      lane + " lane :{e}")
                return None, 1e-6
            # lane_ransac.fit(X_in, y_in, weight_arr)
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
                old_cur, old_direction = old_cur_direction[:, 0]
                new_cur = np.clip(
                    new_cur, old_cur-self.Max_update_step_for_cur, old_cur+self.Max_update_step_for_cur)
                new_direction = np.clip(new_direction, old_direction-self.Max_update_step_for_direction,
                                        old_direction+self.Max_update_step_for_direction)
            # still can't go out of this region
            new_cur = np.clip(
                new_cur, -5*self.Max_update_step_for_cur, 5*self.Max_update_step_for_cur)
            new_direction = np.clip(
                new_direction, -5*self.Max_update_step_for_direction, 5*self.Max_update_step_for_direction)
            lane_wc_fit = np.array(
                (new_cur, new_direction, lane_ransac.intercept_))
            cur_fit_score = lane_ransac.score(X_in, y_in)
            if np.isnan(cur_fit_score):
                cur_fit_score = 1e-6
            lane_fit_score = max(cur_fit_score, 1e-6) * cur_dir_pos_score
            return lane_wc_fit, lane_fit_score
        else:
            return None, 1e-6

    def filter_WC_and_PC_based_on_lane_width(self, distanceAndhead):
        refine_cur_direction = None
        #start_time = time.time()
        left_fit_wc, left_score = self.fit_poly_ransac('LEFT')
        right_fit_wc, right_score = self.fit_poly_ransac('RIGHT')
        #print('lane width based fitting time: %.4f ms'%( (time.time()-start_time)*1e3) )
        # print('LEFT  :    Curve: %.2f,   Angle: %.2f,    Dis: %.2f,    Score: %.2f '%( 1/(1e-9+left_fit_wc[0]), np.rad2deg(left_fit_wc[1]), left_fit_wc[2], left_score[0]) )
        # print('RIGHT :    Curve: %.2f,   Angle: %.2f,    Dis: %.2f,    Score: %.2f '%( 1/(1e-9+right_fit_wc[0]), np.rad2deg(right_fit_wc[1]), right_fit_wc[2], right_score[0]) )

        # TODO:propose a method to judge the confidence of polyfit from lane detection
        pos_precess_noise = self.Left_lane_pos.proc_noise_scale_arr[0] * max(
            0.1, distanceAndhead[0] / 0.5)
        if left_fit_wc is not None:
            left_fit_wc_std = self.Lane_detection_noise_scale_base * 1/left_score
#            print('left_fit_wc_std:', left_fit_wc_std, '     left_fit_gps_std:',left_fit_gps_std)
            left_fit_refine = left_fit_wc.copy()
            left_fit_refine_std = left_fit_wc_std.copy()
            old_pos_left = self.Left_lane_pos.get_refined_result()[0, 0]
            update_lane_pos = np.clip(
                left_fit_refine[2], old_pos_left-self.Max_update_step_for_lane_pos, old_pos_left+self.Max_update_step_for_lane_pos)
            self.Left_lane_pos.update(np.matrix([update_lane_pos], dtype=np.float32), proc_noise_arr=[
                                      pos_precess_noise], meas_noise_arr=[left_fit_refine_std[2]])
            refine_cur_direction = left_fit_refine[:2]
            refine_cur_direction_std = left_fit_refine_std[:2]
            #print('left fit meas_std: %.4f, fixed: %.4f'%(left_fit_refine_std[2], self.Left_lane_pos.meas_noise_scale_arr[0]) )

        if right_fit_wc is not None:
            right_fit_wc_std = self.Lane_detection_noise_scale_base * 1/right_score
#            print('right_fit_wc_std:', right_fit_wc_std, '     right_fit_gps_std:',right_fit_gps_std)
            right_fit_refine = right_fit_wc.copy()
            right_fit_refine_std = right_fit_wc_std.copy()
            old_pos_right = self.Right_lane_pos.get_refined_result()[0, 0]
            update_lane_pos = np.clip(
                right_fit_refine[2], old_pos_right-self.Max_update_step_for_lane_pos, old_pos_right+self.Max_update_step_for_lane_pos)
            self.Right_lane_pos.update(np.matrix([update_lane_pos], dtype=np.float32), proc_noise_arr=[
                                       pos_precess_noise], meas_noise_arr=[right_fit_refine_std[2]])

        if refine_cur_direction is not None:
            self.Lane_cur_direction.update(np.matrix(
                refine_cur_direction, dtype=np.float32).T, meas_noise_arr=refine_cur_direction_std)

        filtered_left_CP, refined_wc_left = self.get_filtered_CP_points(
            flag='LEFT')
        filtered_right_CP, refined_wc_right = self.get_filtered_CP_points(
            flag='RIGHT')

        #self.left_fit_wc = refined_wc_left.copy()
        #self.right_fit_wc = refined_wc_right.copy()
        self.left_fit_wc = refined_wc_left
        self.right_fit_wc = refined_wc_right

        self.Points_in_CP_left_for_draw = filtered_left_CP
        self.Points_in_CP_right_for_draw = filtered_right_CP

    def find_lane_and_fit(self, out_list, pitch_imu, delta_heading, delta_front, delta_right):
        distance = (delta_front**2 + delta_right**2)**0.5
        if len(out_list[2]) > 2:  # 2 is the right lane channel
            right_flag, right_col_row_confidence, right_confidence = self.bounded_search(
                out_list[2], 'RIGHT')
            #print('right percent: %.3f'%selected_right_percent)
            if right_flag:
                self.bounded_right_CP = right_col_row_confidence[:, :2]
            if right_confidence < 0.6:
                self.travel_distance_for_not_detect_right_lane += distance
            elif right_confidence > 0.8:
                self.travel_distance_for_not_detect_right_lane -= distance
        else:
            self.travel_distance_for_not_detect_right_lane += distance
            right_flag, right_col_row_confidence, right_confidence = False, np.zeros(
                (0, 3)), 0
        self.travel_distance_for_not_detect_right_lane = np.clip(
            self.travel_distance_for_not_detect_right_lane, 0, 12)

        if len(out_list[1]) > 2:  # 1 is the left lane channel
            left_flag, left_col_row_confidence, left_confidence = self.bounded_search(
                out_list[1], 'LEFT')
            self.left_confidence = left_confidence
            if left_flag:
                self.bounded_left_CP = left_col_row_confidence[:, :2]
            if left_confidence < 0.6:
                self.travel_distance_for_not_detect_left_lane += distance
            elif left_confidence > 0.8:
                self.travel_distance_for_not_detect_left_lane -= distance

            if left_confidence < 0.6 and right_confidence < 0.6:
                self.continuous_no_detect_distance += distance
            else:
                self.continuous_no_detect_distance = 0

            # use recent_detect_list to count the recent detection performance
            if left_confidence > 0.8:
                self.recent_detect_list.append(1)
            else:
                self.recent_detect_list.append(0)

        else:
            self.continuous_no_detect_distance += distance  # care more about the left lane
            self.travel_distance_for_not_detect_left_lane += distance
            left_flag, left_col_row_confidence = False, np.zeros((0, 3))
            self.recent_detect_list.append(0)

        self.recent_detect_list.pop(0)
        if sum(self.recent_detect_list) == self.init_wait_length:
            # wait for init_wait_length successful frames to initialize publishing the lane
            self.has_initialized = True

        if self.continuous_no_detect_distance > self.max_no_detection_distance:
            self.recent_detect_list = [0] * self.init_wait_length
            self.has_initialized = False

        if self.travel_distance_for_not_detect_left_lane > 12:
            self.travel_distance_for_not_detect_left_lane_count += 1
        else:
            self.travel_distance_for_not_detect_left_lane_count = 0
        self.travel_distance_for_not_detect_left_lane = np.clip(
            self.travel_distance_for_not_detect_left_lane, 0, 12)

        # print('distance: %.4f'%distance)
        print('no left lane distance: %.3f and count: %d' % (
            self.travel_distance_for_not_detect_left_lane, self.travel_distance_for_not_detect_left_lane_count))
        if self.travel_distance_for_not_detect_left_lane_count > self.Reset_no_detect_count_threshold:
            self.reset()
            print('Reset')
        else:
            detect_any_flag = left_flag | right_flag
            detect_all_flag = left_flag & right_flag

            if detect_all_flag:
                self.update_horizon(left_col_row_confidence,
                                    right_col_row_confidence)

            #start_time = time.time()
            self.update_state_with_motion(
                pitch_imu, delta_heading, delta_front, delta_right)
            #print('motion time: %.4f ms'%( (time.time()-start_time)*1e3) )
            # This may not be a good idea to add points manually when only one lane is detected
            if detect_any_flag and (not detect_all_flag):
                if not self.has_initialized:
                    if left_flag:
                        # add right_point manually
                        right_col_row_confidence = np.array(
                            [[self.input_shape[1]*0.8, self.input_shape[0], 0.1]])
                    else:
                        # add left point manually
                        left_col_row_confidence = np.array(
                            [[self.input_shape[1]*0.2, self.input_shape[0], 0.1]])

            # start_time = time.time()
            # Use the detected points to update the lane
            self.update_lane_fit(left_col_row_confidence, right_col_row_confidence, [
                distance, delta_heading])
            #print('fitting time: %.4f ms'%( (time.time()-start_time)*1e3) )

    def update(self, out_list, delta_distance, delta_heading, pitch_imu):
        # turn right has positive delta_heading
        delta_front, delta_right = imu_motion_to_cam_motion(
            delta_distance, delta_heading)
        self.find_lane_and_fit(out_list, pitch_imu,
                               delta_heading, delta_front, delta_right)
