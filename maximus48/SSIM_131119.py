#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:01:47 2018

@author: mpolikarpov, Maxim Polikarpov EMBL HH


compares images with the flatfield with the algorithm described in 
https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
os.environ['OMP_NUM_THREADS'] ='1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
from numpy import sqrt, square 
#import time
        
           

    

class SSIM_const(object):
    __slots__ = ['shape', 'mu', 'delta', 'sigma']
    
    """ Calculate flatfield-related constants for the 2D image
    
    Parameters
    __________
    image : ndarray 
        input image data 2D array, 
        can be in (X,Y,N) shape, where X&Y - pixels of one image, N - different images
    mu, delta, sigma: ndarray
        parameters from the article, can be 2D if there were N of images
    
    """
    
    def __init__(self, image):
        """ Constructor """
        
        #self.data = image
        self.shape = image.shape
        self.mu = np.sum(image, axis = (0,1))/(self.shape[0]*self.shape[1])
        self.delta = image-self.mu
        
        razn = square(image - self.mu)
        if len(self.shape) == 2:        
            self.sigma = sqrt(np.sum(razn)/(razn.shape[0]*razn.shape[1]-1))
        
        elif len(self.shape) == 3:
            self.sigma = sqrt(np.sum(razn, axis = (0,1))/(razn.shape[0]*razn.shape[1]-1))
        
        pass


    

class SSIM(object):
    __slots__ = ['data', 'ff']
    """ 
    Calculate SSIM-index for 2D/3D images
    __________
    image : class Image2D 
        can work for 2D and 3D images
       
    out: ndarray
        calculate SSIM indexes for the images
    
    
    
    
    """
    
    def __init__(self, data, flatfield):
        self.data = data
        self.ff = flatfield 
        
        pass
            

    def ssim(self):
        ''' this function calculates SSIM for 2D or 3D case'''
        
        if len(self.ff.shape) == 2:
            #2D case
            C1=0
            C2=0
            
            #claculate sigmaxy first
            k = self.data.delta * self.ff.delta
            k1 = np.sum(k)
            sxy = k1/(k.shape[0]*k.shape[1]-1)
            
            #calculate SSIM
            nominator = (2 * self.data.mu * self.ff.mu + C1) * (2 * sxy + C2)
            denominator1 = (square(self.data.mu) + square(self.ff.mu) + C1) 
            denominator2 = (square(self.data.sigma) + square(self.ff.sigma) + C2)
            
            out = nominator / (denominator1 * denominator2)    
            
                                                
            
        elif len(self.ff.shape) == 3:
            #3D case
            out = np.zeros(self.ff.shape[2])
            
            for i in range(self.ff.shape[2]):
            
                C1=0
                C2=0
                
                #calculate sigmaxy first
                try:
                    k = self.data.delta * self.ff.delta[:,:,i]
                except ValueError:
                    print('\n####WARNING: The dimensions of the ff-data is wrong. Please make sure that images are stacked along the LAST dimension of the 3D array\n')
                    raise
                k1 = np.sum(k)
                sxy = k1/(k.shape[0]*k.shape[1]-1)
                
                #calculate SSIM
                nominator = (2 * self.data.mu * self.ff.mu[i] + C1) * (2 * sxy + C2)
                denominator1 = (square(self.data.mu) + square(self.ff.mu[i]) + C1) 
                denominator2 = (square(self.data.sigma) + square(self.ff.sigma[i]) + C2)
                
                out[i] = nominator / (denominator1 * denominator2)    
        
        return out
            
        
        
        
    
            
           
    
    

    
# =============================================================================
#     def ssim2D(self):
#         #works!
#         if len(self.ff.shape) != 2:
#             raise Exception('Your flatfield.shape is %s but you try to use SSIM function for 2D flatfield-array' %(len(self.ff.shape)))
# 
#         C1=0
#         C2=0
#         
#         #calculate sigmaxy first
#         k = self.data.delta * self.ff.delta
#         k1 = np.sum(k)
#         sxy = k1/(k.shape[0]*k.shape[1]-1)
#         
#         #calculate SSIM
#         nominator = (2 * self.data.mu * self.ff.mu + C1) * (2 * sxy + C2)
#         denominator1 = (square(self.data.mu) + square(self.ff.mu) + C1) 
#         denominator2 = (square(self.data.sigma) + square(self.ff.sigma) + C2)
#         
#         out = nominator / (denominator1 * denominator2)
#         
#         return out  
# =============================================================================
    
    
# =============================================================================
# class SSIM(object):
#     """ Calculates the SSIM and divides the image""" 
#     
#     def __init__(self, data, flatfield):
#         self.data = data
#         self.ff = flatfield
#         pass
#                   
#     
#     def ssim(self):
#         C1=0
#         C2=0
#         
#         #claculate sigmaxy first
#         k = self.data.delta * self.ff.delta
#         k1 = np.sum(k)
#         sxy = k1/(k.shape[0]*k.shape[1]-1)
#         
#         #calculate SSIM
#         nominator = (2 * self.data.mu * self.ff.mu + C1) * (2 * sxy + C2)
#         denominator1 = (square(self.data.mu) + square(self.ff.mu) + C1) 
#         denominator2 = (square(self.data.sigma) + square(self.ff.sigma) + C2)
#         
#         SSIM = nominator / (denominator1 * denominator2)
#         
#         return SSIM   
# =============================================================================
    
    
    
# =============================================================================
# def ssim(ux, uy, sx, sy, deltaX, deltaY):
#     #Calculate similarity index, based on constants#
#     
#     C1=0
#     C2=0
#     
#     #calculate sigmaxy first
#     k = deltaX * deltaY
#     k1 = np.sum(k)
#     sxy = k1/(k.shape[0]*k.shape[1]-1)
#     
#     SSIM = (2*ux*uy+C1)*(2*sxy+C2)/((square(ux)+square(uy)+C1)*(square(sx)+square(sy)+C2))
#     return SSIM;
# =============================================================================

    
    
    
    
    
    
    
    
 



