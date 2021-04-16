# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:11:36 2021

@author: Henke
"""
import numpy as np

class PowerControl:
    
    def __init__(self, num_dev, sigma_w, rtx):
        self.n = num_dev
        self.sigma_w = sigma_w
        self.rtx = rtx; #Number of retransmissions. rtx=0 means single transmission
        self.P_max = num_dev*np.ones((self.n,1))
        self.h = self.generateH()
        self.w = np.random.normal(0, self.sigma_w)
        
    def generateW(self):
        self.w = np.random.normal(0, self.sigma_w)
        
    def generateH(self):
        #Generate h according to rayleigh fading
        h_real = np.random.normal(0, 0.5, (self.n, 1))
        h_complex = np.random.normal(0, 0.5, (self.n, 1))
        h = h_real+1j*h_complex
        #Sort h
        sort_index = np.argsort(abs(h), axis=0)
        sorted_h = np.zeros((self.n,1)) + 1j*np.zeros((self.n,1))
        i = 0
        for ind in sort_index:
            sorted_h[i] = h[ind]
            i = i + 1
        return sorted_h
    
    def xiaowenEta(self):
        #Solve subproblems to find a list of eta_tildes
        eta_tilde = np.zeros((self.n,1))
        for k in range(self.n):
            sum1 = 0
            sum2 = 0
            for i in range(k+1):
                sum1 = sum1 + self.P_max[i]*np.abs(self.h[i])**2
                sum2 = sum2 + np.sqrt(self.P_max[i])*np.abs(self.h[i])
            eta_tilde[k] = ((self.sigma_w**2+sum1)/sum2)**2
        return np.min(eta_tilde)
    
    def xiaowenB(self):
        b_x = np.zeros((self.n,1))+1j*np.zeros((self.n,1))
        for i in range(self.n):
            if self.P_max[i] > self.eta_x/np.abs(self.h[i])**2:
                b_x[i] = np.conj(self.h[i])*np.sqrt(self.eta_x)/np.abs(self.h[i])**2
            else:
                b_x[i] = np.conj(self.h[i])*np.sqrt(self.P_max[i])/np.abs(self.h[i])
        
        return b_x
    
    def henrikEta(self):
        #Solve subproblems to find a list of eta_tildes
        eta_tilde = np.zeros((self.n,1))
        T = self.rtx+1 #Number of uplink transmissions in each communication round
        for k in range(self.n):
            sum1 = 0
            sum2 = 0
            for i in range(k+1):
                sum1 = sum1 + self.P_max[i]*np.abs(self.h[i])**2
                sum2 = sum2 + np.sqrt(self.P_max[i])*np.abs(self.h[i])
            eta_tilde[k] = ((self.sigma_w**2+T*sum1)/(T*sum2))**2
        return np.min(eta_tilde)
    
    def henrikB(self):
        b_h = np.zeros((self.n,1))+1j*np.zeros((self.n,1))
        for i in range(self.n):
            if self.P_max[i] > self.eta_h/np.abs(self.h[i])**2:
                b_h[i] = np.conj(self.h[i])*np.sqrt(self.eta_h)/np.abs(self.h[i])**2
            else:
                b_h[i] = np.conj(self.h[i])*np.sqrt(self.P_max[i])/np.abs(self.h[i])
        
        return b_h
    
    def truncateB(self, s):
        s = s.reshape(self.n,1)
        b_trunc = np.zeros((self.n,1))+1j*np.zeros((self.n,1))
        for i in range(self.n):
            #If s = 0, there's no reason to transmit anything
            if float(np.abs(s[i])) == 0:
                b_trunc[i] = 0
            #Power constraint exceeded
            elif float(np.abs(self.b_x[i])) > float(np.sqrt(self.P_max[i]))/float(np.abs(s[i])):
                b_trunc[i] = np.conj(self.h[i])/np.abs(self.h[i])*np.sqrt(self.P_max[i])/np.abs(s[i])
                print("Power constraint exceeded! s=", s[i])
            else:
                b_trunc[i] = self.b_x[i]
        
        return b_trunc
    
    #Assumes rows is nxm matrix
    #n is number of devices and m is number of weights in row
    def estLayerXiaowen(self, rows):
        m = rows.shape[1]
        sum_est = np.zeros((m,1))
        for i in range(m):
            est_s = 0
            for j in range(self.rtx+1):
                #Generate new noise
                self.generateW()
                #Calculate sum OTA
                s = rows[:,i].reshape(1,self.n)
                est_s = est_s + (np.dot(s, np.multiply(self.h, self.b_x)) + self.w)/np.sqrt(self.eta_x)
            sum_est[i] = np.real(est_s)/(self.rtx+1)
        return sum_est
    
    #Assumes rows is nxm matrix
    #n is number of devices and m is number of weights in row
    def estLayerHenrik(self, rows):
        m = rows.shape[1]
        sum_est = np.zeros((m,1))
        for i in range(m):
            est_s = 0
            for j in range(self.rtx+1):
                #Generate new noise
                self.generateW()
                #Calculate sum OTA
                s = rows[:,i].reshape(1,self.n)
                est_s = est_s + (np.dot(s, np.multiply(self.h, self.b_h)) + self.w)/np.sqrt(self.eta_h)
            sum_est[i] = np.real(est_s)/(self.rtx+1)
        return sum_est
    
    def estLayer(self, rows, method):
        if method == "xiaowen":
            self.eta_x = self.xiaowenEta()
            self.b_x = self.xiaowenB()
            return self.estLayerXiaowen(rows)
        elif method == "henrik":
            self.eta_h = self.henrikEta()
            self.b_h = self.henrikB()
            return self.estLayerHenrik(rows)
        else:
            print("Incorrect power control method selected, returning zeros.")
            return np.zeros((rows.shape[1], 1))