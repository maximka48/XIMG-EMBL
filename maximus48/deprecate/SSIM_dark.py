#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:01:47 2018

@author: mpolikarpov
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy
from numpy import sqrt, square 
from joblib import Parallel, delayed
import multiprocessing

        
           

def divide(image, flat, dark, ROI=None):
    
    """
    compares images with the flatfield with the algorithm described in 
    https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    
    then divides (image-dark_current) by (the most corresponding flatfield - dark_current) and returns result
    
    Parameters
    __________
    image : ndarray 
        input image data 3D array
    flat : ndarray
        input flatfield data 3D array
    dark : ndarray
        input dark current data 3D array
    ROI : tuple
        image ROI where comparison will take place
    """
    
    
    #functions that are described in the article (use the link above)
    def mu(image):
        u = numpy.sum(image, axis = (0,1))/(image.shape[0]*image.shape[1])
        return u;
    
    def delta(image,mu):
        out = image-mu
        return out;

    def sigma(image, u):
        razn = square(image - u)
        razn1 = numpy.sum(razn) 
        s = sqrt(razn1/(razn.shape[0]*razn.shape[1]-1))
        return s;

    def sigmaxy(delta1, delta2):
        k = delta1*delta2
        k1 = numpy.sum(k)
        out = k1/(k.shape[0]*k.shape[1]-1)
        return out;
           
    def ssim_func(ux, uy, sx, sy, deltaX, deltaY):
        C1=0
        C2=0
    
        sxy = sigmaxy(deltaX, deltaY)
        SSIM = (2*ux*uy+C1)*(2*sxy+C2)/((square(ux)+square(uy)+C1)*(square(sx)+square(sy)+C2))
        return SSIM;
   
    def ff_corr(ux, uy, sx, sy, deltaX, deltaY, image, flat, dark=0):
        best = [0,0]
        for f in range(flat.shape[2]):
            SSIM = ssim_func(ux, uy[f], sx, sy[f], deltaX, deltaY[:,:,f])
            if SSIM > best[0]:
                best = [SSIM, f]
        
        arr = (image-dark)/(flat[:,:,best[1]]-dark)
        return arr;
      
    
    
    # body of the program 
    num_cores = multiprocessing.cpu_count() 
    image = image.transpose(1,2,0)
    flat = flat.transpose(1,2,0)
    
    if ROI:
        a = ROI[0]
        b = ROI[1]
        c = ROI[2]
        d = ROI[3]
    else:
        a=b=c=d=None
   
    
    ux = mu(image[b:d,a:c,:])
    deltaX=delta(image[b:d,a:c,:], ux)

    uy = mu(flat[b:d,a:c,:])
    deltaY=delta(flat[b:d,a:c,:], uy)

    sx = numpy.asarray(Parallel(n_jobs=num_cores)(delayed(sigma)(image[b:d,a:c,i], ux[i]) for i in range(image.shape[2])))
    sy = numpy.asarray(Parallel(n_jobs=num_cores)(delayed(sigma)(flat[b:d,a:c,i], uy[i]) for i in range(flat.shape[2])))
    
    dark = numpy.average(dark, 0)
    filtered = numpy.asarray(Parallel(n_jobs=num_cores)(delayed(ff_corr)(ux[i], uy, sx[i], sy, deltaX[:,:,i], deltaY, image[:,:,i], flat, dark) for i in range(image.shape[2])))
    
    return filtered
