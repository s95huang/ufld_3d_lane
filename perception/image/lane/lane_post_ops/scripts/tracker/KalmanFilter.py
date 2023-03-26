#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:43:19 2019

@author: minghao
"""
import cv2
import numpy as np

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