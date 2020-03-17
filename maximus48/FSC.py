#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:44:41 2020

@author: mpolikarpov

Computes the Fourier Shell Correlation (FSC) between two given 2d- or
3d-images. In the 2D-setting this is also known as the Fourier Ring
correlation (FRC).

By comparing the FSC between two indepedent reconstructions
of an object from two different data sets to the 1/2-bit-threshold
curve, the achieved resolution can be estimated.

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2923553/

"""
import numpy as np
from numpy.fft import fftn, ifftshift
from numpy import floor, ceil, real, divide, multiply, sqrt, size




def FSC(im1, im2, beta = 7):
    """
    Calculate Fourier Shell correlation

    Input arguments:
        im1: 2d- or 3d-array containing the first image to be correlated
        im2: size(im1)-array containing the second image to be correlated
        beta: width for the Kaiser Bessel window 
                standard value is beta = 7

    Output arguments:
        frc: length ceil(norm(size(im1))/2) vector containing the computed values
             of the Fourier shell correlation of the images im1, im2, in ascending
             order of frequencies.
        T_hbit: size(fsc)-vector containing the values of the 1/2-bit-threshold
                    curve for the FSC.
        T_bit:  size(fsc)-vector containing the values of the 1-bit-threshold
                    curve for the FSC.
        nu:     size(fsc)-vector containing the Fourier frequencies 
                    corresponding to the values of fsc and t_hbit.
        no_r:   number of bits per ring/shell
    """
    

    N = im1.shape
    ndim = size(N)    

    # Compute Fourier transforms
    # Default case: apply Kaiser-Bessel window prior to Fourier transform
    if beta > 0:
            window = kaiser_bessel(N,beta)
            im1_fft = fftn(im1 * window)
            im2_fft = fftn(im2 * window)
            window = None
    
    # Special case beta == 0: no Kaiser-bessel window applied
    else :  
            im1_fft = fftn(im1)
            im2_fft = fftn(im2)

    # Compute norm of the wave vector for all points in Fourier space
    for jj in range(ndim):
        axis = np.arange(-floor(0.5*N[jj]),  ceil(0.5*N[jj]))**2
        if jj == 0:
            xi = axis
        else:
            xi = xi[..., None] + axis[None, ...] 
    xi = sqrt(ifftshift(xi))

    # Round values to integers for distribution to different Fourier shells
    shells = (np.around(xi)+1).astype('int64')
    
    # Number of Fourier frequencies in each cell
    n_xi = np.bincount(shells.flatten(), 
                       np.ones((size(shells)), dtype = 'int64'))[1:]           # тут получатется нулевой элемент которого нет в матлабе 
    num_xi = size(n_xi)
    
    # Compute correlation on shells
    FRC = divide(
            np.bincount(shells.flatten(), 
                        real(im1_fft.flatten() * np.conj(im2_fft.flatten())),
                        num_xi),
            sqrt(multiply(
                    np.bincount(shells.flatten(),
                                abs(im1_fft.flatten())**2, num_xi),
                    np.bincount(shells.flatten(),
                                abs(im2_fft.flatten())**2, num_xi)
                    )))[1:]
    
    # Restrict to values on shells that are fully contained within the image
    FRC  =  FRC[:int(ceil((min(N)+1)/2))]
    n_xi = n_xi[:int(ceil((min(N)+1)/2))]
    
    # frequencies
    num_xi = size(n_xi)
    delta_nu = 1/(2*num_xi)
    nu = delta_nu * np.arange(num_xi)
    
    return nu, FRC
    
    
    
     
            
def kaiser_bessel(N, beta):
    """
    make multi-dimensional kaiser-bessel window 
    size(N) corresponds to the number of dimensions in your data
    """
    
    method = np.kaiser
    
    
    # 1D case
    if size(N) == 1:
        window = method(N, beta)
      
    # 2D case
    elif size(N) == 2:
        dim1 = method(N[0],beta)
        dim2 = method(N[1], beta)
        window = dim1[...,None] * dim2[None, ...]
    
    # 3D case
    elif size(N) == 3:
        dim1 = method(N[0],beta)
        dim2 = method(N[1], beta)
        dim3 = method(N[2], beta)
        window = dim1[...,None, None] * dim2[None, ..., None] * dim3[None, None, ...]
    
    else: 
        print('Please check the number of dimensions (N)')
        
    return window












    

