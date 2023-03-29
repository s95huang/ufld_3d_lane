#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 05:24:10 2019

@author: minghao
"""

import numpy as np
from math import *

# estimate the camera motion based on the chasis motion
rot_heading_deg = 1.5  # yaw angle difference between the camera and the chasis
half_length = 3.2  # longitudinal distance from the camera to the chasis
c_theta = np.cos(np.deg2rad(rot_heading_deg))
s_theta = np.sin(np.deg2rad(rot_heading_deg))
T_AL = np.array([
    [c_theta, s_theta, -half_length],
    [-s_theta, c_theta, 0],
    [0, 0, 1]
])  # T_AL: transform matrix from the chasis to the camera
# T_AL: transform matrix from the camera to the chasis
T_LpAp = np.linalg.inv(T_AL)


def imu_motion_to_cam_motion(delta_distance, delta_heading):
    ds = delta_distance
    dyaw = delta_heading
    c_yaw = np.cos(np.deg2rad(dyaw))
    s_yaw = np.sin(np.deg2rad(dyaw))
    T_ApA = np.array([
        [c_yaw, -s_yaw, ds],
        [s_yaw, c_yaw, ds*np.deg2rad(dyaw)/2],
        [0, 0, 1]
    ])  # chasis motion between two frames. dx=ds, dy=ds*np.deg2rad(dyaw)/2

    # camera motion between two frames
    T_LpL = np.dot(np.dot(T_LpAp, T_ApA), T_AL)

    # front direction, corresponding to z axis in the camera frame
    delta_front = T_LpL[0, -1]
    # right direction, corresponding to x axis in the camera frame
    delta_right = T_LpL[1, -1]
    return delta_front, delta_right


class Camera(object):
    """
    there are two coordinate systems:
    1. the camera coordinate system, in pixel coordinate: x is horizontal right, y is vertical down, z is depth front
    2. the world coordinate system, almost the same as the camera coordinate system. The origin is the cross point between the vertical line passing camera center and the ground plane. 
        The x axis is horizontal right, the y axis is vertical down, the z axis is front.
        This world coordinate system may not be the same as the one used in the vehicle, where x is the front, y is the left, z is the up.
        Also, there could be an angle offset between the used world coordinate system and the one used in the vehicle. This angle offset is called as gamma_offset.
        There could also be an angle offset between the imu motion and the camera motion. This is calibrated using imu_motion_to_cam_motion function.
    """

    def __init__(self, input_shape):
        self.input_shape = input_shape  # height, width
        self.xc = 926.05
        self.yc = 511.34
        self.fx = 1288.175
        self.fy = 1444.18

        self.K = np.array(
            [[self.fx, 0, self.xc], [0, self.fy, self.yc], [0, 0, 1]])
        self.h = 2.1  # unit: m
        self.phi_road = None  # unit: degree, positive value means going uphill
        self.phi_t_offset = -3  # CHANGE:the camera is installed a little downtowards
        self.phi_t = self.phi_t_offset

        self.gamma_offset = 0  # CHANGE
        self.gamma = self.gamma_offset
        cos_gamma = cos(radians(self.gamma))
        sin_gamma = sin(radians(self.gamma))
        self.Ry_matrix_still = np.array([[cos_gamma, 0, -sin_gamma],
                                         [0, 1, 0], [sin_gamma, 0, cos_gamma]])

        self.vector_c = np.array([[0, -self.h, 0]]).T
        self.Projection_matrix = self.calculate_projection_matrix()
        self.phi_road_update_rate = 0.06

    def update_phi_road(self, phi_imu):
        if self.phi_road is None:
            self.phi_road = phi_imu  # First time assignment
        else:
            self.phi_road = (1-self.phi_road_update_rate)*self.phi_road + \
                self.phi_road_update_rate*phi_imu  # Exponential moving average

    def update_camera_phi(self, phi_imu, horizon):
        self.update_phi_road(phi_imu)
        phi_t_from_imu = phi_imu - self.phi_road + self.phi_t_offset
        phi_t_from_horizon = degrees(atan((horizon-self.yc)/self.fy))
        # arount -3 degree, it help avoid large change of phi_t
        phi_t_default = degrees(atan((440-self.yc)/self.fy))
        self.phi_t = 0.4*phi_t_from_horizon + 0.4*phi_t_from_imu + 0.2*phi_t_default
        refined_horizon = tan(radians(self.phi_t))*self.fy + self.yc
        return refined_horizon

    def update_camera_state(self, delta_gamma, delta_front, delta_right):
        self.gamma = self.gamma_offset + delta_gamma
        self.vector_c[0, 0] = delta_right
        self.vector_c[2, 0] = delta_front
        self.Projection_matrix = self.calculate_projection_matrix()

    def calculate_projection_matrix(self):
        T_matrix = np.eye(4)[:3]
        T_matrix[:, 3] = -self.vector_c[:, 0]

        cos_gamma = cos(radians(self.gamma))
        sin_gamma = sin(radians(self.gamma))
        Ry_matrix = np.array([[cos_gamma, 0, -sin_gamma],
                              [0, 1, 0], [sin_gamma, 0, cos_gamma]])

        cos_phi_t = cos(radians(self.phi_t))
        sin_phi_t = sin(radians(self.phi_t))
        Rx_matrix = np.array(
            [[1, 0, 0], [0, cos_phi_t, sin_phi_t], [0, -sin_phi_t, cos_phi_t]])

        Projection_matrix = np.dot(
            np.dot(np.dot(self.K, Rx_matrix), Ry_matrix), T_matrix)

        T_matrix_still = np.eye(4)[:3]
        T_matrix_still[1, 3] = -self.vector_c[1, 0]
        self.Projection_matrix_still = np.dot(
            np.dot(np.dot(self.K, Rx_matrix), self.Ry_matrix_still),  T_matrix_still)
        return Projection_matrix

    def project_from_WC_to_CP_still(self, points):
        # still means the camera is not moving, ignoring the motion betweem two time steps
        #  point[0] = x, point[1] = z
        Projection_matrix = self.Projection_matrix_still
        if len(points) == 0:
            return np.array([[], []]).T
        else:
            # [point[0], 0, point[1], 1]
            homo_points = np.insert(
                np.insert(points, 1, 0, axis=1), 3, 1, axis=1)
            projected_vector = np.dot(Projection_matrix, homo_points.T).T
            uv_array = projected_vector[:, :2]/projected_vector[:, [2]]
            return uv_array

    def project_from_WC_to_CP(self, points):
        # consider the motion betweem two time steps
        #  point[0] = x, point[1] = z
        Projection_matrix = self.Projection_matrix
        if len(points) == 0:
            return np.array([[], []]).T
        else:
            # [point[0], 0, point[1], 1]
            homo_points = np.insert(
                np.insert(points, 1, 0, axis=1), 3, 1, axis=1)
            projected_vector = np.dot(Projection_matrix, homo_points.T).T
            uv_array = projected_vector[:, :2]/projected_vector[:, [2]]
            return uv_array

    def project_from_CP_to_WC(self, points):
        # point[0] = u, point[1] = v
        K_inv = np.linalg.inv(self.K)
        cos_phi_t = cos(radians(self.phi_t))
        sin_phi_t = sin(radians(self.phi_t))
        Rx_matrix = np.array(
            [[1, 0, 0], [0, cos_phi_t, sin_phi_t], [0, -sin_phi_t, cos_phi_t]])
        R_inv = np.linalg.inv(np.dot(Rx_matrix, self.Ry_matrix_still))
        R_inv_K_inv = np.dot(R_inv, K_inv)

        if len(points) == 0:
            return np.array([[], []]).T
        else:
            homo_points = np.insert(points, 2, 1, axis=1)
            M_vector = np.dot(R_inv_K_inv, homo_points.T).T
            s = self.h / M_vector[:, 1]
            x, z = s * M_vector[:, 0], s * M_vector[:, 2]
            return np.array([x, z]).T
