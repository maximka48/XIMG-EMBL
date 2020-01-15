"""
Created on Fri Oct  12 15:46:49 2018

@author: mpolikarpov
"""


from numpy import floor, meshgrid, multiply, divide, square, sin, cos, sqrt, mean, pi, fliplr, flipud, rot90, ceil, arange, concatenate
from numpy.fft import ifftshift, fft2, ifft2
from scipy.special import erfc





def single_distance_CTF(image, beta_delta, fresnelNumber, zero_compensation):
    """
    This function makes the CTF phase-retrieval from the single-distance image.
    Please refer to the book of Tim Salditt "Biomedical imaging", 2017, ISBN = 978-3-11-042668-7, ISBN (PDF)= 978-3-11-0426694
    equation 6.144
    
    Parameters
    __________
    image : float32
        2D image
    beta_delta: int
        absorption divided by refraction  (decrements). Normally small, for ex. 0.08 
    fresnel Number: int
        can be calculated as fresnelN = pixelsize**2/(wavelength*object_detector_distance)
    zero_compensation: int
        just a parameter, should be small, for ex. 0.01
    """
    
    ctf = ctf_mono(image.shape, fresnelNumber, fresnelNumber, beta_delta, zero_compensation)
    out1 = multiply(fft2(image - 1),ctf)
    reco = ifft2(out1).real
    return reco

    # =============================================================================
    # in case of multiple distance please rewrite function reco as:
    # a1 = sum over all images(fft2(projection - 1)*nominator)
    # a2 = sum over all images(denominator)
    # out1 = a1/a2 
    # 
    # functions nominator and denominator can be found in the monochromaticCTF.m file
    # also when you will align images one to each other - first calculate phase_retrieved_images at different distances
    #                                                     then detect the shift
    #                                                     then correct original images for this shift and do several-distance-CTF 
    # =============================================================================


def circle(x,y,r):
    """
    This function creates a mask - 2D array with zeros outside a circular region and ones inside the circle.
    x is the xaxis of the coordinate system, y is the y axis. The third argument is the radius of the circle. 
    
    Parameters
    __________
    x : 1D array
        x axis
    y : 1D array
        y axis
    r: int
        radius of the mask circle    
    """
    
    X,Y = meshgrid(x,y)
    c = sqrt(square(X)+square(Y))
    c[c<=r]=1
    c[c>r]=0
    return c 


def ctf_mono(imageshape, fresnelx, fresnely, beta_delta, zero_compensation):
    """
    This is the main function which calculates nominator/denominator. Please refer to the book of Salditt
    """
    # set fourier coordinate system
    Ny = imageshape[0]
    Nx = imageshape[1] 
    dqx = 1/Nx
    dqy = 1/Ny
    
    xx = arange(1,Nx+1)-floor(Nx/2)-1 
    yy = arange(1,Ny+1)-floor(Ny/2)-1
    Qx, Qy = meshgrid(xx*dqx, yy*dqy)
      
    # formulate the contrast transfer function
    #argument = ifftshift(pi/fresnelx*square(Qx)+pi/fresnely*square(Qy))
    argument = ifftshift(pi/fresnelx*square(Qx)+pi/fresnely*square(Qy))         # Салдит в своей книге по другому задает Qx, в 2пи раз больше
    ctf_mono_nominator = sin(argument)+ beta_delta * cos(argument)
    ctf_mono_denominator =  2 * square(ctf_mono_nominator)

    # regularization    
    # cutoff matrix using the function from P. Cloetens
    cutoff_sigma = 0.01
    cutoff = sqrt(mean(fresnelx)*0.5)
    r = filt2(Ny, cutoff*Ny, cutoff_sigma*Ny, Nx, cutoff*Nx)  
    regularization = multiply(zero_compensation,(1-r))              
    ctf_mono_denominator = ctf_mono_denominator + regularization    
    
    # aliasing free region f
    N_correct = min(floor(fresnelx / dqx**2), floor(fresnely / dqy**2)) 
    f  =  ifftshift(circle(xx,yy,floor(N_correct/2)))
    
    ctf_mono = multiply(divide(ctf_mono_nominator,ctf_mono_denominator), f)     
    ctf_mono[abs(ctf_mono)<zero_compensation] = zero_compensation 

    return ctf_mono



def filt2(n, cutn, sigman, m=None, cutm=None):
        
# =============================================================================
# # regularisation function
# =============================================================================

# =============================================================================
# % Copyright (C) 2007 P. Cloetens
# %
# % This program is free software; you can redistribute it and/or modify
# % it under the terms of the GNU General Public License as published by
# % the Free Software Foundation; either version 2 of the License, or
# % (at your option) any later version.
# %
# % This program is distributed in the hope that it will be useful,
# % but WITHOUT ANY WARRANTY; without even the implied warranty of
# % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# % GNU General Public License for more details.
# %
# % You should have received a copy of the GNU General Public License
# % along with this program; if not, write to the Free Software
# % Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
# %
# % filt2
# %		r = filt2(n,cutn,sigman,varargin)
# %		create low-pass 2D filter in Fourier domain
# %		erfc type
# %
# %		arguments:
# %		argument 1: number of pixels along dimension 1
# %		argument 2: cut_off along dimension 1
# %		argument 3: width (sigma) of transition along dimension 1
# %		argument 4: number of pixels along dimension 2 ( default : same as argument 1)
# %		argument 5: cut_off along dimension 2 ( default : isotropic )
# %
# %		See also: filt2_gaussian
# 
# % Author: P. Cloetens <cloetens@esrf.fr>
# %
# % 2007-03-26 P. Cloetens <cloetens@esrf.fr>
# % * Initial revision
# %
# % adapted to Matlab: Martin Krenkel, February 2013
# =============================================================================
    
    if not m:
        m = n
    if not cutm:    
        cutm = cutn*m/n
        
        
    x,y = meshgrid(arange(floor(m/2)+1),arange(floor(n/2)+1))
    r = sqrt(square(x)+square(y*(cutm/cutn)))
 
    # im Prinzip nur hier:
    r = erfc((r-cutn)/sigman)
    r=r/r[0,0]
    
    r1 = fliplr(r[:,1:int(ceil(m/2))])
    r2 = flipud(r[1:int(ceil(n/2)),:]) 
    r3 = rot90(r[1:int(ceil(n/2)),1:int(ceil(m/2))],2)
    
    out0 = concatenate((r,r1), axis=1)
    out1 = concatenate((r2,r3), axis=1)
    out  = concatenate((out0,out1), axis=0)
        
    return out






def CTF_function(image, fresnelN, beta_delta):
    """
    One-dimensional CTF function
     
    Parameters
    __________
    image : 2D array
        image
    beta_delta: int
        absorption divided by refraction  (decrements). Normally small, for ex. 0.08 
    fresnel Number: int
        can be calculated as fresnelN = pixelsize**2/(wavelength*object_detector_distance)
    """
    if image.shape[0] != image.shape[1]:
        raise Exception('The image is not square. Please make it square and try again')
    
    Nx = image.shape[0]
    dqx = 1/Nx
    xx = arange(1,Nx+1)-floor(Nx/2)-1 
    Qx = xx*dqx
      
    # formulate the contrast transfer function
    argument = ifftshift(pi/fresnelN*square(Qx))         
    ctf_mono_nominator = sin(argument) + beta_delta * cos(argument)            #uncomment if you want to take the absorbtion in account
    
    return abs(ctf_mono_nominator)**2


def CTF_function2D(image, fresnelN, beta_delta):
    """
    One-dimensional CTF function
     
    Parameters
    __________
    image : 2D array
        image
    beta_delta: int
        absorption divided by refraction  (decrements). Normally small, for ex. 0.08 
    fresnel Number: int
        can be calculated as fresnelN = pixelsize**2/(wavelength*object_detector_distance)
    """
    
    Ny = image.shape[0]
    Nx = image.shape[1]
    dqx = 1/Nx
    dqy = 1/Ny
    
    xx = arange(1,Nx+1)-floor(Nx/2)-1 
    yy = arange(1,Ny+1)-floor(Ny/2)-1
    Qx, Qy = meshgrid(xx*dqx, yy*dqy)
      
    # formulate the contrast transfer function
    argument = (pi/fresnelN*square(Qx)+pi/fresnelN*square(Qy))         
    ctf_mono_nominator = sin(argument)+ beta_delta * cos(argument)
    
    return abs(ctf_mono_nominator)




