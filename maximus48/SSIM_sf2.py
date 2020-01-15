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
                   

def divide(image, flat, ROI=None):
    
    """
    compares images with the flatfield with the algorithm described in 
    https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    
    then divides image by the most corresponding flatfield and returns result
    
    Parameters
    __________
    image : ndarray 
        input image data 2D array
    flat : ndarray
        input flatfield data 3D array
    ROI : tuple
        image ROI where comparison will take place
    ncore: int
        number of cores assigned to job 
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
   
    def ff_corr(ux, uy, sx, sy, deltaX, deltaY):
        best = [0,0]
        for f in range(flat.shape[2]):
            SSIM = ssim_func(ux, uy[f], sx, sy[f], deltaX, deltaY[:,:,f])
            if SSIM > best[0]:
                best = [SSIM, f]
        return best[1] 
      
        
# =============================================================================
# Old version - the file is corrected on 010819
#     def ff_corr(ux, uy, sx, sy, deltaX, deltaY, image, flat):
#         best = [0,0]
#         for f in range(flat.shape[2]):
#             SSIM = ssim_func(ux, uy[f], sx, sy[f], deltaX, deltaY[:,:,f])
#             if SSIM > best[0]:
#                 best = [SSIM, f]
#         arr = image/flat[:,:,best[1]]
#         return arr;
# =============================================================================
    
    
    # body of the program 
    flat = flat.transpose(1,2,0)
    
    if ROI:
        a = ROI[0]
        b = ROI[1]
        c = ROI[2]
        d = ROI[3]
    else:
        a=b=c=d=None
   
    
    ux = mu(image[b:d,a:c])
    deltaX=delta(image[b:d,a:c], ux)

    uy = mu(flat[b:d,a:c])
    deltaY=delta(flat[b:d,a:c], uy)

    sx = sigma(image[b:d,a:c], ux)
    
    sy = []
    for i in range(flat.shape[2]):
        sy.append(sigma(flat[b:d,a:c,i], uy[i]))
    sy = numpy.asarray(sy)
    
    #filtered = numpy.asarray(ff_corr(ux, uy, sx, sy, deltaX, deltaY, image, flat))
    
    out = ff_corr(ux, uy, sx, sy, deltaX, deltaY)
    filtered = numpy.asarray(image/flat[:,:,out])
    
    return filtered




