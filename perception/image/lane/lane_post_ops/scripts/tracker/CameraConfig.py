#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 05:24:10 2019

@author: minghao
"""

import numpy as np
from math import *

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
